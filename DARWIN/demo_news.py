"""Live FinBERT sentiment demo for review sessions."""

from src.agents.news_agent import get_asset_sentiment

print('=' * 78)
print('  LIVE NEWS SENTIMENT — FinBERT (ProsusAI/finbert)')
print('=' * 78)

for ticker in ['AAPL', 'GOOGL', 'MSFT', 'SPY']:
    score, details = get_asset_sentiment(ticker, max_items=5)
    arrow = 'UP' if score > 0.1 else ('DN' if score < -0.1 else '--')
    verdict = 'BUY' if score > 0.15 else ('SELL' if score < -0.15 else 'HOLD')
    print(f"\n{ticker:6s}  [{arrow}]  aggregate={score:+.3f}  signal: {verdict}")
    for d in details[:3]:
        print(f"        {d['score']:+.2f}  {d['headline'][:78]}")