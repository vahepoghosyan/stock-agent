#!/usr/bin/env python3
"""Daily stock analysis agent — runs on a schedule or immediately with --run-now."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import schedule
import time
from dotenv import load_dotenv

from stocks import TechnicalAnalysis, analyze_stock
from news import fetch_market_news
from ai import generate_report
from email_sender import send_report, send_error_email


load_dotenv()
ROOT = Path(__file__).parent


def run_analysis() -> None:
    print(f"\n[{datetime.now().isoformat()}] Starting daily stock analysis...")

    portfolio = json.loads((ROOT / "portfolio.json").read_text())
    stocks_cfg = portfolio["stocks"]
    tickers = [s["ticker"] for s in stocks_cfg]
    shares_map    = {s["ticker"]: s.get("shares", 0)     for s in stocks_cfg}
    buy_price_map = {s["ticker"]: s.get("buy_price", 0.0) for s in stocks_cfg}
    print(f"Portfolio: {', '.join(tickers)}")

    # Fetch all stocks in parallel
    analyses: list[TechnicalAnalysis] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(analyze_stock, t, shares_map[t], buy_price_map[t]): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                analyses.append(result)
                sig = result.overall_signal.replace("_", " ").upper()
                print(f"  {ticker}: ${result.current_price:.2f}  RSI={result.rsi14:.1f}  → {sig}")
            except Exception as exc:
                print(f"  ⚠️  {ticker} skipped: {exc}")

    if not analyses:
        print("No stocks analyzed — aborting.")
        return

    # Fetch news
    print("Fetching market news...")
    news = fetch_market_news(tickers)
    print(f"  {len(news)} articles fetched")

    # Claude analysis
    print("Generating AI report with Claude Opus 4.7...")
    report = generate_report(analyses, news)

    # Email
    print("Sending email report...")
    send_report(report)

    recipient = os.environ.get("EMAIL_TO", os.environ.get("GMAIL_USER", "?"))
    print(f"✅ Report sent to {recipient}")
    if report.top_buy:
        print(f"   🟢 Buy:  {', '.join(report.top_buy)}")
    if report.top_sell:
        print(f"   🔴 Sell: {', '.join(report.top_sell)}")


def main() -> None:
    if "--run-now" in sys.argv:
        try:
            run_analysis()
        except Exception as exc:
            print(f"ERROR: {exc}")
            try:
                send_error_email(exc)
                print("Error notification sent by email.")
            except Exception as mail_exc:
                print(f"Failed to send error email: {mail_exc}")
        return

    market_open  = os.environ["RUN_TIME"]
    market_close = os.environ["MARKET_CLOSE"]

    print(f"Stock Analysis Agent started.")
    print(f"Sends reports every 1 minute from {market_open} to {market_close} (local time).")
    print("Run  python main.py --run-now  to trigger immediately.\n")

    def run_if_market_open() -> None:
        now = datetime.now()
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            print(f"[{now.isoformat()}] Weekend — market closed, skipping.")
            return
        time_str = now.strftime("%H:%M")
        if market_open <= time_str < market_close:
            try:
                run_analysis()
            except Exception as exc:
                print(f"[{now.isoformat()}] ERROR: {exc}")
                try:
                    send_error_email(exc)
                except Exception as mail_exc:
                    print(f"  Failed to send error email: {mail_exc}")
        else:
            print(f"[{now.isoformat()}] Outside market hours ({market_open}–{market_close}) — skipping.")

    schedule.every(60).minutes.do(run_if_market_open)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
