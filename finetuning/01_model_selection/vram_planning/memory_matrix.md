# **Memory Matrix**

*Stage 01. Date checked: 28 Nov 2026. Figures come from the model notes in `models/`. Rule from `memory_formulas.md`: weights are $N \times b$ with $N$ the total parameters, and active parameters lower compute, not memory.*

## **1. How to Read It**

A **need** is a figure a vendor quotes. A **peak** is a measured number at a stated sequence length. A peak is worth more than a need.

## **2. Training Memory by Model and Path**

| model | path | figure | kind | note |
|---|---|---|---|---|
| Gemma 4 E2B | train | 8 GB | need | tier A |
| Gemma 4 E4B | train | 10 GB | need | tier A |
| Gemma 4 12B | QLoRA | about 17 | measured | unit not recorded |
| Gemma 4 31B | QLoRA | 22 GB | need | tier B |
| Gemma 4 26B A4B | LoRA | above 40 GB | need | 4 bit not recommended |
| Qwen3.8 27B | QLoRA | 24 GB | need | tier B |
| Qwen3.8 27B | LoRA | above 36 GB, plan 56 GB for bf16 | need | the 48 GB tier is too small for the plan |
| Muse Glimmer 30B | QLoRA | 22.57 GiB at 1,024 tokens | measured | spare 0.43 GiB on a 23 GiB card |
| Muse Glimmer 30B | LoRA | above 40 GB | need | bf16 weights about 56 GB |
| Inkling 2 Mini | QLoRA | 22 GB need, 25.7 GiB peak at 1,024 tokens | need and measured | batch size 1 |
| Inkling 2 Mini | 16 bit LoRA | 59 GB | measured | merged 16 bit about 53 GB |
| Nemotron 3.5 Bolt E4B | Gemma path | not confirmed | not checked | |
| Nemotron 3.5 Lightning 30B A3B | any | not checked | not checked | |

## **3. Does It Fit the Platform Tiers**

Usable memory: L4 is 22.49 GiB (verified). T4, A10, L40S and 80 GB cards are labels only, since usable memory is not checked.

| model and path | 16 GB T4 | 24 GB L4 | 48 GB L40S | 80 GB |
|---|---|---|---|---|
| Gemma 4 E2B | yes | yes | yes | yes |
| Gemma 4 E4B | likely | yes | yes | yes |
| Gemma 4 12B QLoRA | no | yes, spare about 5.5 to 6.7 GiB | yes | yes |
| Gemma 4 31B QLoRA | no | open, spare 2.00 or 0.49 GiB | yes | yes |
| Qwen3.8 27B QLoRA | no | open, spare 0.14 GiB or no | yes | yes |
| Qwen3.8 27B LoRA bf16 | no | no | no | yes |
| Glimmer QLoRA | no | no, over by 0.08 GiB | yes | yes |
| Inkling 2 Mini QLoRA | no | no, over by 3.21 GiB | yes | yes |
| Inkling 2 Mini 16 bit LoRA | no | no | no, 59 GB | yes |

**Worked check for one cell.** Glimmer QLoRA on the L4:

| step | value |
|---|---|
| measured peak | 22.57 GiB |
| usable $U$ | 22.49 GiB |
| $S = 22.49 - 22.57$ | $\boxed{-0.08}$ |

Negative spare means it does not fit, even before the 1.0 GiB margin from `decision_framework.md`.

## **4. Sources**

L4 usable memory is verified (see `decision_framework.md` section 8). Every other figure is as marked in its model note. The unit of the 12B, Gemma 31B and Qwen figures is an open item.