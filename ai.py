"""Generate buy/sell/hold recommendations using Claude."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

from stocks import TechnicalAnalysis, transaction_fee, round_trip_fee, break_even_pct
from news import NewsArticle


@dataclass
class Recommendation:
    ticker: str
    action: str           # BUY | SELL | HOLD
    confidence: str       # HIGH | MEDIUM | LOW
    current_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    buy_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    shares: int = 0
    position_value: float = 0.0


@dataclass
class AnalysisReport:
    generated_at: str
    market_summary: str
    news_impact: str
    recommendations: list[Recommendation]
    top_buy: list[str]
    top_sell: list[str]
    total_portfolio_value: float = 0.0
    total_cost_basis: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_unrealized_pnl_pct: float = 0.0


client = anthropic.Anthropic()


def _format_stock(a: TechnicalAnalysis) -> str:
    pct = f"{a.price_change_1d_pct:+.2f}%"
    oversold   = " [OVERSOLD]"   if a.rsi14 < 30 else ""
    overbought = " [OVERBOUGHT]" if a.rsi14 > 70 else ""

    if a.shares > 0:
        one_way = transaction_fee(a.shares, a.current_price)
        full    = round_trip_fee(a.shares, a.current_price)
        be_pct  = break_even_pct(a.shares, a.current_price)
        fee_line = (
            f"  Fees: {a.shares} shares | sell fee=${one_way:.2f} | "
            f"round-trip=${full:.2f} | break-even move={be_pct:.3f}%\n"
        )
        if a.buy_price > 0:
            sign = "+" if a.unrealized_pnl >= 0 else ""
            pnl_line = (
                f"  P&L: bought @ ${a.buy_price:.2f} | "
                f"unrealized {sign}${a.unrealized_pnl:.2f} ({sign}{a.unrealized_pnl_pct:.2f}%)\n"
            )
        else:
            pnl_line = ""
    else:
        fee_line = ""
        pnl_line = ""

    return (
        f"**{a.ticker}** — ${a.current_price:.2f} ({pct} today)\n"
        f"  RSI(14): {a.rsi14:.1f}{oversold}{overbought}\n"
        f"  MACD histogram: {a.macd_histogram:.4f} → {a.macd_signal}\n"
        f"  Bollinger Bands: {a.bb_position} | upper=${a.bb_upper:.2f} lower=${a.bb_lower:.2f}\n"
        f"  SMA50=${a.sma50:.2f} SMA200=${a.sma200:.2f} → {a.ma_signal}\n"
        f"  Overall technical verdict: {a.overall_signal.upper().replace('_', ' ')}\n"
        f"{pnl_line}"
        f"{fee_line}"
    )


def _format_news(articles: list[NewsArticle]) -> str:
    if not articles:
        return "(No news available today)"
    lines = []
    for a in articles[:15]:
        desc = f": {a.description[:120]}" if a.description else ""
        lines.append(f"• [{a.source}] {a.title}{desc}")
    return "\n".join(lines)


def generate_report(
    analyses: list[TechnicalAnalysis],
    news: list[NewsArticle],
) -> AnalysisReport:
    from datetime import datetime, timezone

    stocks_text = "\n\n".join(_format_stock(a) for a in analyses)
    news_text   = _format_news(news)

    prompt = f"""You are an expert quantitative stock analyst. Analyze the portfolio stocks using the technical indicators and recent news below, then provide clear buy/sell/hold recommendations.

## Market News (Last 24 Hours)
{news_text}

## Portfolio Technical Analysis
{stocks_text}

## Instructions
Reply with ONLY valid JSON — no markdown fences, no prose outside the JSON:

{{
  "market_summary": "2-3 sentence overview of current market conditions",
  "news_impact": "How today's macro news may affect this specific portfolio",
  "recommendations": [
    {{
      "ticker": "AAPL",
      "action": "BUY",
      "confidence": "HIGH",
      "current_price": 192.50,
      "target_price": 210.00,
      "stop_loss": 182.00,
      "reasoning": "Cite RSI, MACD, BB, SMA signals and any news context (2-4 sentences)"
    }}
  ],
  "top_buy": ["TICKER1"],
  "top_sell": ["TICKER2"]
}}

Rules:
- action must be "BUY", "SELL", or "HOLD" (exact case)
- confidence must be "HIGH", "MEDIUM", or "LOW"
- target_price and stop_loss are optional — derive from BB/SMA levels when clear, else omit (null)
- target_price for any position with a buy_price must always exceed the buy_price (the investor needs to at least break even); if technicals don't support a target above buy_price, set action to HOLD or SELL instead
- Only include a ticker in top_buy/top_sell when there is a strong, multi-indicator signal
- Each stock shows its brokerage fee and break-even % — only recommend BUY or SELL when the expected move clearly exceeds the break-even threshold; otherwise prefer HOLD"""

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    text = "".join(
        b.text for b in message.content if b.type == "text"
    )

    # Extract the JSON object from the response
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Claude response contained no JSON:\n{text[:400]}")

    parsed = json.loads(match.group())
    analysis_map = {a.ticker: a for a in analyses}

    def to_rec(r: dict) -> Recommendation:
        a = analysis_map.get(r["ticker"])
        return Recommendation(
            ticker=r["ticker"],
            action=r["action"],
            confidence=r["confidence"],
            current_price=float(r["current_price"]),
            target_price=float(r["target_price"]) if r.get("target_price") else None,
            stop_loss=float(r["stop_loss"]) if r.get("stop_loss") else None,
            reasoning=r["reasoning"],
            buy_price=a.buy_price if a else 0.0,
            unrealized_pnl=a.unrealized_pnl if a else 0.0,
            unrealized_pnl_pct=a.unrealized_pnl_pct if a else 0.0,
            shares=a.shares if a else 0,
            position_value=a.shares * a.current_price if a else 0.0,
        )

    total_value    = sum(a.shares * a.current_price for a in analyses)
    total_cost     = sum(a.shares * a.buy_price for a in analyses if a.buy_price > 0)
    total_pnl      = total_value - total_cost if total_cost > 0 else 0.0
    total_pnl_pct  = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    return AnalysisReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        market_summary=parsed.get("market_summary", ""),
        news_impact=parsed.get("news_impact", ""),
        recommendations=[to_rec(r) for r in parsed.get("recommendations", [])],
        top_buy=parsed.get("top_buy", []),
        top_sell=parsed.get("top_sell", []),
        total_portfolio_value=total_value,
        total_cost_basis=total_cost,
        total_unrealized_pnl=total_pnl,
        total_unrealized_pnl_pct=total_pnl_pct,
    )
