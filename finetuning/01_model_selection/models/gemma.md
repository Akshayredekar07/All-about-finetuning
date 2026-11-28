# **Gemma 4**

**Date checked:** 28 Nov 2026

Gemma 4 is Google DeepMind's open model family, released in April 2026, with a 12B size added on 3 Jun 2026. It has the cleanest license of the candidates and one size for each memory tier, so it is the main tier A candidate in this stage.

## **1. Spec Table**

| Size | Type | Parameters | Context | Status |
|---|---|---|---|---|
| E2B | Dense, edge | 2.3B effective, 5.1B with embeddings | 128K | verified |
| E4B | Dense, edge | 4.5B effective, 8B with embeddings | 128K | verified |
| 12B Unified | Dense, no separate vision or audio encoder | 11.95B | 256K | verified |
| 26B A4B | Mixture of experts, 128 experts, 8 picked | 25.2B total, 3.8B active | 256K | verified |
| 31B | Dense | about 30.7B | 256K | verified |

Embedding tables are large but only used for lookups, which is why the effective count is lower than the stored count for E2B and E4B. Global attention layers use shared keys and values and p-RoPE to save memory at long context.

## **2. License**

**Apache 2.0** for the whole family. Card: https://huggingface.co/unsloth/gemma-4-12B-it

## **3. Memory and Fine Tune Path**

| Item | Value | Source | Date | Status |
|---|---|---|---|---|
| Inference bf16 and Q4_0, E4B | 17.9 GB and 4.5 GB | Google Gemma docs | 28 Nov 2026 | verified |
| Inference bf16 and Q4_0, 12B | 26.7 GB and 6.7 GB | Google Gemma docs | 28 Nov 2026 | verified |
| Inference bf16 and Q4_0, 26B A4B | 57.7 GB and 14.4 GB | Google Gemma docs | 28 Nov 2026 | verified |
| Inference bf16 and Q4_0, 31B | 69.9 GB and 17.5 GB | Google Gemma docs | 28 Nov 2026 | verified |
| Training, E2B | 8 GB | Unsloth guide | 28 Nov 2026 | verified |
| Training, E4B | 10 GB, LoRA 17 GB | Unsloth guide | 28 Nov 2026 | verified |
| Training, 12B QLoRA | about 17 GB peak on one RTX 5090, 9.5 GB per card on 2 × T4 | two community model cards | 28 Nov 2026 | one run each |
| Training, 31B | QLoRA 22 GB | Unsloth guide | 28 Nov 2026 | verified |
| Training, 26B A4B | LoRA above 40 GB, 4 bit not recommended | Unsloth guide | 28 Nov 2026 | verified |

The open item from earlier is closed: for 26B A4B, Unsloth's own recipe loads in 16 bit and says MoE QLoRA is not recommended.

### **3.1 Worked example: QLoRA on the 12B**

This uses the formula from memory_formulas.md with 200M adapter parameters and 4 GB extra.

| Step | Calculation | Result |
|---|---|---|
| 1. Weights at 4 bit | Google Q4_0 figure | 6.7 GB |
| 2. Adapters | 200 × 16 / 1000 | 3.2 GB |
| 3. Activations and buffers | starting guess | 4.0 GB |
| 4. Total guess | 6.7 + 3.2 + 4.0 | 13.9 GB |
| 5. Measured run | RTX 5090, rank 16, context 512 | about 17 GB |

$$
\boxed{\text{Guess 13.9 GB, measured about 17 GB, so plan with 17 GB}}
$$

The guess is low here, so use the measured number. That puts the 12B at the top of tier A or the bottom of tier B.

## **4. Traps**

| Trap | What to do |
|---|---|
| E2B and E4B size labels | Memory follows the stored 5.1B and 8B, compute follows the effective counts |
| 12B needs a new transformers | Two projects report version 5.10 and 5.14 as the minimum, install the newer one |
| The tokenizer is a multimodal processor | Pass text with the text= keyword |
| 4 bit training for 26B A4B | Use 16 bit LoRA or pick another model |
| float16 on T4 cards | Use Unsloth, which carries fixes for this |
| 12B memory claim of 16 GB | That is for inference at short context, not training |

## **5. Pre-flight Answers**

| Question | Answer |
|---|---|
| Best tier A pick in this family | E4B, 10 GB to train |
| Best tier B pick in this family | 31B with QLoRA at 22 GB |
| Is the license clean for products | Yes, Apache 2.0 |
| Does 12B train on a free T4 | Reported at 9.5 GB per card on two T4 cards, check with your data |

## **6. Sources**

| Source | Used for | Date |
|---|---|---|
| https://ai.google.dev/gemma/docs/core | Inference memory table | read 28 Nov 2026 |
| https://huggingface.co/blog/gemma4 | Sizes, parameters, context | read 28 Nov 2026 |
| https://unsloth.ai/docs/models/gemma-4/train | Training memory, MoE advice | read 28 Nov 2026 |
| https://huggingface.co/unsloth/gemma-4-12B-it | License, global layer design | read 28 Nov 2026 |
| https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/ | 12B launch | read 28 Nov 2026 |
| https://huggingface.co/exploitintel/cve-cwe-gemma4-12b | 17 GB peak run | read 28 Nov 2026 |
| https://huggingface.co/keypa/oracle-gemma4-12b-lora | 9.5 GB per T4 run | read 28 Nov 2026 |