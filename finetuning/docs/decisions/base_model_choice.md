# **Decision Record: Base Model Choice**

*Stage 01. Date: 28 Nov 2026. Status: provisional. Capability is not checked and no peak is measured on an L4, so this record names the leading pick and what would change it.*

## **1. The Decision**

| role | pick | runner up |
|---|---|---|
| Tier A, fast loops (16 GB T4) | Gemma 4 E4B | Gemma 4 E2B |
| Tier B, real target (24 GB L4) | Gemma 4 31B | Qwen3.8 27B |

## **2. Defense on Four Axes**

| axis | Gemma 4 E4B | Gemma 4 31B | Qwen3.8 27B |
|---|---|---|---|
| Capability | not checked | not checked | not checked |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| Memory | 10 GB need | 22 GB need, spare 2.00 or 0.49 GiB on L4 | 24 GB need, spare 0.14 GiB or no fit on L4 |
| Tooling | Unsloth | Unsloth | Unsloth |

**Why Gemma 4 31B leads.** On memory it has the best odds on the 22.49 GiB L4, and it fits under both readings of the quoted unit. It shares a family, a chat template and a tool path with the tier A pick, so a config proven on E4B carries over. Qwen3.8 27B fits only if its 24 GB is decimal, and then with 0.14 GiB to spare.

**Why not the others on this platform.** Muse Glimmer misses the L4 by 0.08 GiB and Inkling 2 Mini by 3.21 GiB. Both fit the 48 GB tier, so keep them for a later stage with a bigger GPU.

## **3. What Would Change the Pick**

| event | effect |
|---|---|
| Measured Gemma 4 31B peak at 1,024 tokens is above 21.5 GiB | move tier B to Gemma 4 12B, or to the 48 GB tier |
| Qwen3.8 27B beats Gemma 4 31B by more than 0.20 on the 50 prompt probe and fits the L4 | Qwen3.8 27B becomes the target |
| The probe gap is under 0.20 | keep Gemma 4 31B on memory and tooling |

## **4. Next Actions**

1. Run the peak memory cell on an L4 for Gemma 4 31B and Qwen3.8 27B at 1,024 tokens.
2. Run the 50 prompt probe and fill `results/leaderboard_snapshots.md`.
3. Close open items O1 to O7 in `results/claims_log.md`.
4. Change the status line above from provisional to final.