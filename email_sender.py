"""Send the daily analysis report via Gmail."""

from __future__ import annotations

import os
import smtplib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ai import AnalysisReport, Recommendation
from charts import generate_1h_candlestick_chart


def _action_color(action: str) -> str:
    return {"BUY": "#16a34a", "SELL": "#dc2626"}.get(action, "#d97706")


def _action_bg(action: str) -> str:
    return {"BUY": "#dcfce7", "SELL": "#fee2e2"}.get(action, "#fef9c3")




def _sort_recs(recs: list[Recommendation]) -> list[Recommendation]:
    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    return sorted(recs, key=lambda r: order.get(r.action, 2))


def _subject(report: AnalysisReport) -> str:
    date = datetime.fromisoformat(report.generated_at).strftime("%b %-d")
    buys  = report.top_buy
    sells = report.top_sell
    if buys and sells:
        return f"📈 Stock Alert {date}: BUY {', '.join(buys[:2])} | SELL {', '.join(sells[:2])}"
    if buys:
        return f"🟢 Stock Alert {date}: Buy — {', '.join(buys[:3])}"
    if sells:
        return f"🔴 Stock Alert {date}: Sell — {', '.join(sells[:3])}"
    return f"📊 Daily Stock Analysis — {date}"


def _portfolio_summary_html(report: AnalysisReport) -> str:
    if report.total_portfolio_value == 0:
        return ""
    pnl   = report.total_unrealized_pnl
    pct   = report.total_unrealized_pnl_pct
    color = "#16a34a" if pnl >= 0 else "#dc2626"
    bg    = "#dcfce7" if pnl >= 0 else "#fee2e2"
    sign  = "+" if pnl >= 0 else ""
    arrow = "▲" if pnl >= 0 else "▼"
    return f"""
<div style="background:white;border-radius:8px;padding:20px;margin-bottom:16px;
            border:1px solid #e5e7eb;text-align:center;">
  <p style="margin:0 0 4px 0;font-size:13px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;">Total Portfolio Value</p>
  <p style="margin:0 0 8px 0;font-size:32px;font-weight:700;color:#111827;">${report.total_portfolio_value:,.2f}</p>
  <span style="background:{bg};color:{color};border-radius:6px;padding:4px 14px;
               font-size:14px;font-weight:600;">
    {arrow} {sign}${pnl:,.2f} ({sign}{pct:.2f}%) unrealized
  </span>
  <p style="margin:8px 0 0 0;font-size:12px;color:#9ca3af;">Cost basis: ${report.total_cost_basis:,.2f}</p>
</div>"""


def _html(report: AnalysisReport, chart_tickers: set[str] | None = None) -> str:
    date_str = datetime.fromisoformat(report.generated_at).strftime(
        "%A, %B %-d, %Y"
    )

    def rec_card(r: Recommendation) -> str:
        price_cells = f"<td style='padding:3px 10px;'><strong>Price:</strong> ${r.current_price:.2f}</td>"
        if r.shares > 0:
            price_cells += f"<td style='padding:3px 10px;'><strong>Value:</strong> ${r.position_value:,.2f} ({r.shares} shares)</td>"
        if r.buy_price > 0:
            price_cells += f"<td style='padding:3px 10px;'><strong>Bought:</strong> ${r.buy_price:.2f}</td>"

        pnl_block = ""
        if r.buy_price > 0:
            color  = "#16a34a" if r.unrealized_pnl >= 0 else "#dc2626"
            bg     = "#dcfce7" if r.unrealized_pnl >= 0 else "#fee2e2"
            sign   = "+" if r.unrealized_pnl >= 0 else ""
            arrow  = "▲" if r.unrealized_pnl >= 0 else "▼"
            pnl_block = (
                f"<div style='display:inline-block;background:{bg};color:{color};"
                f"border-radius:6px;padding:4px 12px;font-size:13px;font-weight:600;margin-bottom:8px;'>"
                f"{arrow} {sign}${r.unrealized_pnl:.2f} ({sign}{r.unrealized_pnl_pct:.2f}%)"
                f"</div>"
            )

        chart_block = ""
        if chart_tickers and r.ticker in chart_tickers:
            chart_block = (
                f"<img src='cid:chart_{r.ticker}' "
                f"style='width:100%;max-width:600px;border-radius:6px;"
                f"margin-bottom:12px;display:block;' alt='{r.ticker} candlestick chart'>"
            )
        return f"""
<div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:12px;background:#fff;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <h3 style="margin:0;font-size:17px;color:#111827;">
      <span style="background:{_action_bg(r.action)};color:{_action_color(r.action)};
                   padding:2px 10px;border-radius:4px;font-size:13px;font-weight:700;margin-right:8px;">{r.action}</span>
      {r.ticker}
    </h3>
    <span style="color:#6b7280;font-size:12px;background:#f3f4f6;padding:2px 8px;border-radius:12px;">{r.confidence} confidence</span>
  </div>
  {chart_block}
  <table style="font-size:13px;color:#374151;margin-bottom:10px;"><tr>{price_cells}</tr></table>
  {pnl_block}
  <p style="margin:0;font-size:14px;color:#4b5563;line-height:1.65;">{r.reasoning}</p>
</div>"""

    cards = "".join(rec_card(r) for r in _sort_recs(report.recommendations))

    buy_badges  = " ".join(
        f"<span style='background:#dcfce7;color:#16a34a;padding:2px 9px;border-radius:12px;"
        f"font-weight:600;margin:2px;'>{t}</span>"
        for t in report.top_buy
    )
    sell_badges = " ".join(
        f"<span style='background:#fee2e2;color:#dc2626;padding:2px 9px;border-radius:12px;"
        f"font-weight:600;margin:2px;'>{t}</span>"
        for t in report.top_sell
    )
    signals_block = ""
    if buy_badges or sell_badges:
        signals_block = f"""
<div style="background:white;border-radius:8px;padding:20px;margin-bottom:16px;border:1px solid #e5e7eb;">
  <h2 style="margin:0 0 12px 0;font-size:15px;color:#374151;">🎯 Key Signals</h2>
  {"<div style='margin-bottom:6px;'><strong style='color:#16a34a;'>🟢 Buy:</strong> " + buy_badges + "</div>" if buy_badges else ""}
  {"<div><strong style='color:#dc2626;'>🔴 Sell:</strong> " + sell_badges + "</div>" if sell_badges else ""}
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
             max-width:680px;margin:0 auto;padding:20px;background:#f9fafb;color:#111827;">

<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;
            border-radius:12px;padding:24px;margin-bottom:20px;">
  <h1 style="margin:0 0 4px 0;font-size:22px;">📈 Daily Stock Analysis</h1>
  <p style="margin:0;opacity:0.85;font-size:13px;">{date_str}</p>
</div>

{_portfolio_summary_html(report)}

<div style="background:white;border-radius:8px;padding:20px;margin-bottom:16px;border:1px solid #e5e7eb;">
  <h2 style="margin:0 0 10px 0;font-size:15px;color:#374151;">📊 Market Summary</h2>
  <p style="margin:0;line-height:1.7;color:#4b5563;">{report.market_summary}</p>
</div>

<div style="background:white;border-radius:8px;padding:20px;margin-bottom:16px;border:1px solid #e5e7eb;">
  <h2 style="margin:0 0 10px 0;font-size:15px;color:#374151;">📰 News Impact</h2>
  <p style="margin:0;line-height:1.7;color:#4b5563;">{report.news_impact}</p>
</div>

{signals_block}

<div style="background:white;border-radius:8px;padding:20px;margin-bottom:16px;border:1px solid #e5e7eb;">
  <h2 style="margin:0 0 16px 0;font-size:15px;color:#374151;">📋 Recommendations</h2>
  {cards}
</div>

<div style="background:#fef9c3;border:1px solid #fde68a;border-radius:8px;
            padding:14px;font-size:12px;color:#78350f;text-align:center;line-height:1.5;">
  ⚠️ For informational purposes only — not financial advice.
  Always do your own research before investing.
</div>

<p style="text-align:center;color:#9ca3af;font-size:11px;margin-top:12px;">
  Stock Analysis Agent · {report.generated_at}
</p>
</body>
</html>"""


def _text(report: AnalysisReport) -> str:
    date_str = datetime.fromisoformat(report.generated_at).strftime("%A, %B %-d, %Y")
    emoji = {"BUY": "🟢", "SELL": "🔴"}.get

    lines = [
        f"DAILY STOCK ANALYSIS — {date_str}",
        "=" * 50,
        "",
    ]
    if report.total_portfolio_value > 0:
        sign = "+" if report.total_unrealized_pnl >= 0 else ""
        lines += [
            f"PORTFOLIO VALUE: ${report.total_portfolio_value:,.2f}",
            f"P&L: {sign}${report.total_unrealized_pnl:,.2f} ({sign}{report.total_unrealized_pnl_pct:.2f}%)  |  Cost basis: ${report.total_cost_basis:,.2f}",
            "",
        ]
    lines += [
        "MARKET SUMMARY",
        report.market_summary,
        "",
        "NEWS IMPACT",
        report.news_impact,
    ]
    if report.top_buy:
        lines.append(f"\nBUY OPPORTUNITIES: {', '.join(report.top_buy)}")
    if report.top_sell:
        lines.append(f"SELL ALERTS: {', '.join(report.top_sell)}")

    lines.append("\nRECOMMENDATIONS\n")
    for r in _sort_recs(report.recommendations):
        e = emoji(r.action, "🟡")
        lines.append(f"{e} {r.ticker} — {r.action} ({r.confidence} confidence)")
        lines.append(f"  Price: ${r.current_price:.2f}" + (f"  |  Value: ${r.position_value:,.2f} ({r.shares} shares)" if r.shares > 0 else ""))
        if r.buy_price > 0:
            sign = "+" if r.unrealized_pnl >= 0 else ""
            lines.append(f"  Bought: ${r.buy_price:.2f}  |  P&L: {sign}${r.unrealized_pnl:.2f} ({sign}{r.unrealized_pnl_pct:.2f}%)")
        lines.append(f"  {r.reasoning}")
        lines.append("")

    lines += ["---", "⚠️ Not financial advice.", f"Generated: {report.generated_at}"]
    return "\n".join(lines)


def send_report(report: AnalysisReport) -> None:
    user = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    to = os.environ.get("EMAIL_TO", user)

    # Generate candlestick charts in parallel
    tickers = [r.ticker for r in report.recommendations]
    charts: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(generate_1h_candlestick_chart, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data:
                charts[ticker] = data

    html_content = _html(report, set(charts.keys()))

    # multipart/mixed
    #   multipart/alternative
    #     text/plain
    #     multipart/related        ← html + inline images
    #       text/html
    #       image/png (cid per ticker)
    msg = MIMEMultipart("mixed")
    msg["Subject"] = _subject(report)
    msg["From"] = f"Stock Agent 📈 <{user}>"
    msg["To"] = to

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(_text(report), "plain"))

    related = MIMEMultipart("related")
    related.attach(MIMEText(html_content, "html"))
    for ticker, img_bytes in charts.items():
        img = MIMEImage(img_bytes, "png")
        img.add_header("Content-ID", f"<chart_{ticker}>")
        img.add_header("Content-Disposition", "inline", filename=f"{ticker}_chart.png")
        related.attach(img)

    alt.attach(related)
    msg.attach(alt)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.sendmail(user, to, msg.as_string())


def send_error_email(error: Exception) -> None:
    """Send a plain error notification email."""
    user = os.environ["GMAIL_USER"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    to = os.environ.get("EMAIL_TO", user)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"⚠️ Stock Agent Error — {type(error).__name__} at {now}"
    body = (
        f"Stock Agent encountered an error at {now}:\n\n"
        f"Type: {type(error).__name__}\n"
        f"Message: {error}\n"
    )

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = f"Stock Agent 📈 <{user}>"
    msg["To"] = to
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.sendmail(user, to, msg.as_string())
