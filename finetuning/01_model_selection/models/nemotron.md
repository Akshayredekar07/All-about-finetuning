# **Nemotron 3.5**

**Date checked:** 28 Nov 2026

Nemotron 3.5 is NVIDIA's open model line from Aug to Oct 2026. Two members matter for fine tuning: Bolt E4B, a small tier A model built on Gemma 4 E4B, and Lightning 30B A3B, a hybrid mixture of experts model for tier B and above.

## **1. Spec Table**

| Item | Bolt E4B | Lightning 30B A3B | Status |
|---|---|---|---|
| Released | 23 Oct 2026 | 11 Aug 2026 | verified |
| Total parameters | 8B | 30B | verified |
| Active or effective | 4B effective | 3B active | verified |
| Design | Gemma 4 E4B, post trained by NVIDIA with reinforcement learning | Mamba 2 layers, mixture of experts and a few attention layers | verified |
| Context | 64K in and out | 1M | verified |
| Input | text | text | verified |
| Reference weights | BF16 checkpoint for customization | BF16 checkpoint for customization | verified |
| Best for | Local voice and device assistants, personas | General reasoning, agent tasks | verified |

Other members, not planned for training here: Bolt 74B A9B, a 2 bit copy that fits about 24 GB, and Super VL at 120B total with 12B active.

## **2. License**

**OpenMDW License 1.1** for Bolt E4B and Lightning. Bolt E4B also carries Gemma 4's Apache 2.0 terms underneath, since it is built on Gemma 4. Card: https://catalog.ngc.nvidia.com/orgs/nim/nvidia/models/nemotron-3.5-bolt-e4b-4gb/-

This corrects my earlier plan to treat the Nemotron license as unchecked. OpenMDW 1.1 is a short permissive license close to MIT. The Open Source Initiative has not approved it, and one review flags a termination clause that covers copyright suits as well as patent suits.

## **3. Memory and Fine Tune Path**

| Item | Value | Source | Date | Status |
|---|---|---|---|---|
| Bolt E4B BF16 repository | 14.92 GB compressed | NVIDIA NGC | 28 Nov 2026 | verified |
| Bolt E4B 4 bit file | the name says 4GB, one guide reports a 6.2 GB Q4_K_M file | NGC and a guide | 28 Nov 2026 | size conflict |
| Bolt E4B QLoRA, LoRA, full | 7.0 GB, 18.6 GB, 130 GB | a third party estimate | 28 Nov 2026 | estimate only |
| Gemma 4 E4B training, same shape | 10 GB, LoRA 17 GB | Unsloth guide | 28 Nov 2026 | verified |
| Bolt E4B recipes | full and LoRA recipes in NeMo AutoModel, 8 GPU FSDP2 for full | NVIDIA docs | 28 Nov 2026 | verified |
| Lightning training memory | not found | none | 28 Nov 2026 | not checked |

### **3.1 Worked example: QLoRA on Bolt E4B**

| Step | Calculation | Result |
|---|---|---|
| 1. Weights at 4 bit | 8 × 0.5 | 4.0 GB |
| 2. Adapters | 200 × 16 / 1000 | 3.2 GB |
| 3. Activations and buffers | starting guess | 4.0 GB |
| 4. Total guess | 4.0 + 3.2 + 4.0 | 11.2 GB |
| 5. Gemma 4 E4B, Unsloth | published | 10 GB |

$$
\boxed{\text{Guess 11.2 GB, Gemma 4 E4B published 10 GB, so tier A}}
$$

Bolt E4B is the same shape as Gemma 4 E4B, so I expect the Gemma path in Unsloth to work. I did not find a source that confirms it.

## **4. Traps**

| Trap | What to do |
|---|---|
| Two licenses on one model | Keep both in docs/licenses.md |
| The card says it is mainly for post training and adaptation | Test it before you use it directly |
| Benchmarks are from NVIDIA | Treat them as vendor reported |
| Lightning is a mixture of experts model | Expert training in 4 bit may be blocked, not checked |
| The 4GB in the name | It is a quantized build, not the BF16 size |

## **5. Pre-flight Answers**

| Question | Answer |
|---|---|
| Tier for Bolt E4B | Tier A |
| Tier for Lightning | Tier B or above, memory not checked |
| Clean license | OpenMDW 1.1, permissive, not Apache |
| Compare with Gemma 4 E4B | Gemma has the simpler Apache license, Bolt has extra tool use and instruction tuning |

## **6. Sources**

| Source | Used for | Date |
|---|---|---|
| https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-BF16 | Bolt E4B spec, recipes | read 28 Nov 2026 |
| https://catalog.ngc.nvidia.com/orgs/nim/nvidia/models/nemotron-3.5-bolt-e4b-4gb/- | License, checkpoint size | read 28 Nov 2026 |
| https://modelbeats.com/models/nemotron-3-5-bolt-e4b | Third party memory estimates | read 28 Nov 2026 |
| https://aitoolsreview.co.uk/insights/nvidia-nemotron-3-5-bolt | Release date, card caveats | read 28 Nov 2026 |
| https://www.buildmvpfast.com/blog/nvidia-nemotron-3-5-super-vl-bolt-open-models-2026 | OpenMDW notes, family sizes | read 28 Nov 2026 |
| https://friendli.ai/models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | Lightning spec, license, date | read 28 Nov 2026 |