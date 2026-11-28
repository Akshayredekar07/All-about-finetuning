# **Leaderboard Snapshots**

*Stage 01. Date: 28 Nov 2026. No leaderboard was read for this stage, so every score cell is empty and marked not checked. Do not fill a cell from memory.*

## **1. How to Take a Snapshot**

1. Open the leaderboard page and write its address and the date below.
2. Copy only the rows for the candidates in `configs/candidates.yaml`.
3. Mark each row verified, and keep the page address next to it.
4. Treat a public score as a hint. The 50 prompt probe in `decision_framework.md` is the test that counts for your task.

## **2. Snapshot Table**

| date | leaderboard | model | score | status |
|---|---|---|---|---|
| | | Gemma 4 E4B | | not checked |
| | | Gemma 4 31B | | not checked |
| | | Qwen3.8 27B | | not checked |
| | | Muse Glimmer 30B | | not checked |
| | | Inkling 2 Mini | | not checked |

## **3. Your Own Probe**

| model | prompts | passes | $p$ | SE |
|---|---|---|---|---|
| Gemma 4 E4B | 50 | | | |
| Gemma 4 31B | 50 | | | |
| Qwen3.8 27B | 50 | | | |

SE is $\sqrt{p(1-p)/50}$. A gap under 0.20 between two models is noise.