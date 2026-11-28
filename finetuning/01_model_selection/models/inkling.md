# **Inkling 2 Mini**

**Date checked:** 28 Nov 2026

Inkling 2 Mini is an open weights looped mixture of experts model from Thinking Machines Lab, released on 20 Nov 2026. It is small enough to train with QLoRA on a 24 GB class card, which makes it the main tier B candidate in this stage.

## **1. Spec Table**

| Item | Value | Status |
|---|---|---|
| Model id | thinkingmachines/Inkling-2-Mini | verified |
| Released | 20 Nov 2026 | verified |
| Total parameters | 26.4B | verified |
| Active per token | 3.4B on one pass, 5.4B counting loop passes | verified |
| Unique weights used per text token | 3.1B | verified, Artificial Analysis |
| Layers | 24 stored, middle 12 run 3 times, 48 layer visits | verified |
| Experts | 128 per layer, 8 picked per token | verified |
| Context | 1M tokens | verified |
| Input and output | text and image in, text out | verified |
| Official checkpoint | about 20 GB, most weights NVFP4, last four layers MXFP8 | verified |
| Memory advice on the card | 32 GB or more for running | verified |

Each extra loop pass adds about 1.0B active parameters, which gives 3.4B plus 2 times 1.0B, or 5.4B. The loop design is explained in architecture_facts.md.

## **2. License**

**Apache 2.0**, plus a separate Model Acceptable Use Policy published by Thinking Machines. Card: https://huggingface.co/thinkingmachines/Inkling-2-Mini

Read the policy before shipping a product. Inkling 2 Small, the bigger sibling, is also Apache 2.0.

## **3. Memory and Fine Tune Path**

| Item | Value | Source | Date | Status |
|---|---|---|---|---|
| bf16 weights by formula | 52.8 GB | 26.4 × 2 | 28 Nov 2026 | verified |
| Merged 16 bit file on disk | about 53 GB | Unsloth guide | 28 Nov 2026 | verified |
| Official checkpoint | about 20 GB | launch coverage of the model card | 28 Nov 2026 | verified |
| QLoRA, local training | 22 GB VRAM | Unsloth guide | 28 Nov 2026 | verified |
| QLoRA benchmark peak | 25.7 GiB at 1,024 tokens on RTX PRO 6000, against 55.4 GiB with transformers, PEFT and TRL | Unsloth guide | 28 Nov 2026 | verified |
| 16 bit LoRA peak | 59 GB, 2.54 s per step, against 25 GB and 3.05 s per step for QLoRA | Unsloth guide | 28 Nov 2026 | verified |
| Free GPUs | Kaggle 2 × T4 works at about 26 s per step, one 15 GB T4 does not | Unsloth release notes | 28 Nov 2026 | verified |
| QLoRA peak on a 24 GB card | 21.1 GB | my earlier research | earlier draft | not found again |

The benchmark setup used batch size 1 with gradient accumulation 4, LoRA rank 16 on attention and the routed experts, and the adamw_8bit optimizer.

### **3.1 Worked example: does QLoRA fit a 24 GB card**

| Step | Calculation | Result |
|---|---|---|
| 1. Card size | given | 24 GB |
| 2. Keep free | 1 GB reserve | 23 GB usable |
| 3. Need, Unsloth minimum | published | 22 GB |
| 4. Need, benchmark peak | published | 25.7 GiB |
| 5. Spare at the minimum | 23 − 22 | 1 GB |

$$
\boxed{\text{Fits at short context with 1 GB spare, but the 25.7 GiB benchmark peak does not fit}}
$$

Test with your own sequence length before you commit to this card.

## **4. Traps**

| Trap | What to do |
|---|---|
| Speed follows 5.4B, not 3.4B | Use 5.4B when you plan time |
| Official checkpoint is NVFP4, not bf16 | Start from the Unsloth model id for training |
| Batch size above 1 | Keep batch size 1 and raise gradient accumulation |
| Gate layers in 4 bit | Keep them out of 4 bit (earlier research, not rechecked) |
| vLLM fast inference in training | Not supported for Inkling 2 yet |
| One 15 GB T4 | Does not fit, use two T4 cards |
| Expecting Small level quality | The card says Mini is less capable than Small on most evaluations |

## **5. Pre-flight Answers**

| Question | Answer |
|---|---|
| Can I train it on a free GPU | Yes, on Kaggle 2 × T4, not on one T4 |
| Smallest local card for QLoRA | 22 GB class, so tier B |
| Is it safe for commercial use | Apache 2.0, read the acceptable use policy |
| Does it fit tier A | Inference may, training does not |

## **6. Sources**

| Source | Used for | Date |
|---|---|---|
| https://unsloth.ai/docs/models/inkling-2-mini/fine-tuning | Training memory, speeds, traps | read 28 Nov 2026 |
| https://github.com/unslothai/unsloth/releases | Kaggle 2 × T4 result | read 28 Nov 2026 |
| https://artificialanalysis.ai/articles/inkling-2-mini | 3.4B, 5.4B, 3.1B, 1M context | read 28 Nov 2026 |
| https://venturebeat.com/technology/thinking-machines-wows-with-inkling-2-mini-looped-transformer-model-that-runs-locally-on-laptops | 24 layers, 8 of 128 experts, license | read 28 Nov 2026 |
| https://letsdatascience.com/news/thinking-machines-lab-releases-inkling-2-mini-an-open-weight-170080f4 | 1.0B per extra pass, 48 visits | read 28 Nov 2026 |
| https://casrai.org/news/thinking-machines-inkling-2-mini-open-looped-model-laptop-2026 | NVFP4 and MXFP8 checkpoint, 20 GB | read 28 Nov 2026 |