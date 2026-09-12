\---



\## News Sentiment Agent (FinBERT)



The `news` agent scores live financial headlines with \*\*FinBERT\*\* — a

transformer fine-tuned on financial text by Prosus AI. It understands

context, not just keywords: "Apple beats earnings" and "Apple recalls

iPhones" both contain "Apple" but score oppositely.



\### How it works



1\. Fetches today's headlines from Yahoo Finance RSS for each ticker

2\. Scores each headline with FinBERT → `\[-1, +1]` per headline

3\. Averages into a per-asset sentiment score

4\. Emits `BUY` if score > +0.15, `SELL` if < -0.15, else `HOLD`

5\. Adapts thresholds monthly based on performance (same as other agents)



\### Live demo



```bash

python demo\_news.py

