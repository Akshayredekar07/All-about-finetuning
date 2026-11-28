# **Muse Glimmer**

**Date checked:** 28 Nov 2026

Muse Glimmer 30B is Meta's dense open weights model for local agent work, released on 10 Aug 2026 under Apache 2.0. It fits a 24 GB card for QLoRA, but with very little spare room.

## **1. Spec Table**

| Item | Value | Status |
|---|---|---|
| Model id | meta-models/Muse-Glimmer-30B | verified |
| Released | 10 Aug 2026 | verified |
| Type | Dense causal transformer plus a perception encoder | verified |
| Total parameters | about 29.6B, including a 1.8B ViT-G/14 vision encoder | verified |
| Layers | 52, in a repeating pattern of 3 local layers and 1 global layer | verified |
| Local attention window | 2,048 tokens | verified |
| Attention heads | 32 query, 2 key value | verified |
| Context | 131,072 and above, Unsloth lists up to 262,144 | verified |
| Input and output | text and images in, text out | verified |
| Helper model | 2.56B DFlash drafter for faster generation | verified |

With one global layer in four, 13 of the 52 layers keep a full attention cache, and the two key value heads keep that cache small.

## **2. License**

**Apache 2.0**, with a separate usage policy file from Meta. Card: https://huggingface.co/meta-models/Muse-Glimmer-30B

The earlier draft was wrong to say Muse Model License. That license belongs to Muse Spark Open, the 816B model released on 15 Oct 2026, covered in serve_only.md.

## **3. Memory and Fine Tune Path**

| Item | Value | Source | Date | Status |
|---|---|---|---|---|
| bf16 shards | about 56 GB | Unsloth guide and a community card | 28 Nov 2026 | verified |
| bf16 by formula | 59.2 GB | 29.6 × 2 | 28 Nov 2026 | verified, differs from files |
| Dynamic 4 bit from Unsloth | 21 GB | Unsloth guide | 28 Nov 2026 | verified |
| Q4_K_M GGUF, inference | 16.9 GB | community card | 28 Nov 2026 | one source |
| QLoRA | works on 24 GB | Unsloth guide | 28 Nov 2026 | verified |
| LoRA | above 40 GB | Unsloth guide | 28 Nov 2026 | verified |
| Measured QLoRA peak | 22.57 GiB reserved at 1,024 tokens, batch 1, rank 16 | Unsloth guide | 28 Nov 2026 | verified |
| Free GPUs | Kaggle 2 × T4 notebooks | Unsloth guide | 28 Nov 2026 | verified |

My earlier draft listed 58 to 60 GB and 17 to 18 GB as published sizes. The sources I found today say 56 GB and 21 GB, so those replace it.

### **3.1 Worked example: does the measured peak fit a 24 GiB card**

| Step | Calculation | Result |
|---|---|---|
| 1. Card size | given | 24 GiB |
| 2. Keep free | 1 GiB reserve | 23 GiB usable |
| 3. Measured peak | published | 22.57 GiB |
| 4. Spare | 23 − 22.57 | 0.43 GiB |

$$
\boxed{\text{Fits with 0.43 GiB spare at 1,024 tokens, so keep batch size 1 and short context}}
$$

Unsloth itself says to reserve a 24 GiB card as the realistic minimum.

## **4. Traps**

| Trap | What to do |
|---|---|
| 29.6B includes a 1.8B vision encoder | Text only training freezes it, so the trained part is smaller |
| Little spare memory on 24 GiB | Start at short context, Unsloth advises this too |
| License mix up with Spark Open | Glimmer is Apache 2.0, Spark Open is not |
| Context is shorter than Inkling or Qwen | 131K here, against 1M and 262K |
| Read the usage policy | It sits next to the license file |

## **5. Pre-flight Answers**

| Question | Answer |
|---|---|
| Tier | Tier B, tight |
| Dense or MoE | Dense, so no expert training problem in 4 bit |
| Commercial use | Apache 2.0 with no user cap, plus the usage policy |
| Free GPU path | Yes, Kaggle 2 × T4 notebooks |

## **6. Sources**

| Source | Used for | Date |
|---|---|---|
| https://unsloth.ai/docs/models/muse-glimmer/train | QLoRA, LoRA, measured peak | read 28 Nov 2026 |
| https://huggingface.co/blog/muse-glimmer | Architecture, release | read 28 Nov 2026 |
| https://huggingface.co/keras/muse_glimmer_30b_assistant | Layer pattern, heads, drafter | read 28 Nov 2026 |
| https://venturebeat.com/technology/meta-returns-to-open-source-with-muse-glimmer-an-apache-2-0-licensed-30b-parameter-ai-model-optimized-for-agents-available-now | License, 29.6B, 52 layers | read 28 Nov 2026 |
| https://mer.vin/news/muse-glimmer-30b-is-metas-first-open-weight-model-since-llama-4/ | Usage policy | read 28 Nov 2026 |
| https://huggingface.co/EryriLabs/Glimmer-Sentry-30B | 4 bit loaded size, GGUF sizes | read 28 Nov 2026 |