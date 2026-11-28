# **Model Matrix**

*Repo level summary for Stage 01. Date checked: 28 Nov 2026. Each row is marked verified, earlier draft or not checked. Details live in `01_model_selection/models/`.*

## **1. Trainable Candidates**

| model | released | size | license | smallest train path | tier | status |
|---|---|---|---|---|---|---|
| Gemma 4 E2B | Apr 2026 | small | Apache 2.0 | 8 GB | A | verified |
| Gemma 4 E4B | Apr 2026 | small | Apache 2.0 | 10 GB | A | verified |
| Gemma 4 12B | 3 Jun 2026 | 12B | Apache 2.0 | QLoRA about 17 | B | verified |
| Gemma 4 31B | Apr 2026 | 31B | Apache 2.0 | QLoRA 22 GB | B | verified |
| Gemma 4 26B A4B | Apr 2026 | 26B, 4B active | Apache 2.0 | LoRA above 40 GB | C | verified |
| Qwen3.8 27B | 14 Aug 2026 | 27B dense | Apache 2.0 | QLoRA 24 GB | B | verified |
| Muse Glimmer 30B | 10 Aug 2026 | 29.6B dense | Apache 2.0 plus usage policy | QLoRA peak 22.57 GiB | C | verified |
| Inkling 2 Mini | 20 Nov 2026 | 26.4B, 3.4B active | Apache 2.0 plus acceptable use policy | QLoRA peak 25.7 GiB | C | verified |
| Nemotron 3.5 Bolt E4B | 23 Oct 2026 | E4B | OpenMDW 1.1 over Apache 2.0 | Gemma path | A | path not confirmed |
| Nemotron 3.5 Lightning 30B A3B | 11 Aug 2026 | 30B, 3B active | OpenMDW 1.1 | not checked | C | not checked |
| Olala mini 7A1B | not checked | 7B | not checked | not checked | | earlier draft |
| Ripple 65B Preview | not checked | 65B | not checked | not checked | | earlier draft |

## **2. Serve Only**

| model | size | license | note | status |
|---|---|---|---|---|
| Inkling 2 Small | 594B, 17B active | Apache 2.0 | 4 bit about 310 GB | earlier draft |
| Kimi K3.1 Pro | 2.8T, 104B active | separate agreement above 20 million dollars | about 1.56 TB | verified |
| Muse Spark Open | 816B, 42B active | Muse Model License | attribution required | verified |
| Qwen3.8 2.4T | 2.4T | see model note | | verified |
| DeepSeek V4 Flash | 284B or 304B | MIT | size conflict open | open |

## **3. Memory Reminder**

Weights are $N \times b$ with $N$ the total parameters. Real 4 bit files run 0.55 to 0.75 bytes per parameter, so use the real file size. Active parameters lower compute, not memory.