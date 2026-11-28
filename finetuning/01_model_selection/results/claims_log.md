# **Claims Log**

*Stage 01. Last updated: 28 Nov 2026. One row per claim that the decision depends on. Status is verified, earlier draft or not checked.*

## **1. Claims That Decide the Pick**

| id | claim | status | source | date |
|---|---|---|---|---|
| C1 | The L4 reports 23,034 MiB, which is 22.49 GiB | verified | Docker GPU docs, AAU GPU guide | 28 Nov 2026 |
| C2 | 24 GB consumer cards report about 23.65 GiB in PyTorch | verified | PyTorch forum thread, NVIDIA forum thread | 28 Nov 2026 |
| C3 | Glimmer QLoRA peak is 22.57 GiB at 1,024 tokens | verified | models/glimmer.md | 28 Nov 2026 |
| C4 | Inkling 2 Mini QLoRA peak is 25.7 GiB at 1,024 tokens | verified | models/inkling.md | 28 Nov 2026 |
| C5 | Qwen3.8 27B QLoRA needs 24 GB, LoRA above 36 GB | verified | models/qwen.md | 28 Nov 2026 |
| C6 | Gemma 4 31B QLoRA needs 22 GB, E4B 10 GB, E2B 8 GB | verified | models/gemma.md | 28 Nov 2026 |
| C7 | Gemma 4 12B QLoRA is about 17 measured | verified | models/gemma.md | 28 Nov 2026 |
| C8 | All five tier candidates are Apache 2.0, two with an extra policy | verified | models/*.md | 28 Nov 2026 |

## **2. Open Items**

| id | item | status | what closes it |
|---|---|---|---|
| O1 | Unit (GB or GiB) of the vendor needs in C5, C6, C7 | not checked | run the peak memory cell on an L4 |
| O2 | Usable memory of T4, A10, L40S and 80 GB cards | not checked | print `total_memory` on each tier |
| O3 | Capability of every candidate | not checked | the 50 prompt probe |
| O4 | Gemma Unsloth path works for Nemotron Bolt E4B | not checked | one smoke test |
| O5 | Nemotron Lightning training memory | not checked | read the model card |
| O6 | Olala mini 7A1B and Ripple 65B Preview | earlier draft | search each |
| O7 | DeepSeek V4 Flash size, 284B or 304B | open conflict | read the model card |

## **3. Corrections Made**

| earlier statement | correction | why |
|---|---|---|
| Inkling QLoRA peak 21.1 GB | 25.7 GiB at 1,024 tokens | the 21.1 figure could not be found again |
| Glimmer has 0.43 GiB spare on 24 GB | negative, minus 0.08 GiB on an L4 | the 0.43 assumed a 23 GiB card, the L4 has 22.49 |
| The L4 has about 22.5 GiB usable | 22.49 GiB, verified | first value was from memory |