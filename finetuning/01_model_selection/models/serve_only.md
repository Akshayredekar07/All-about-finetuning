# **Serve Only Models**

**Date checked:** 28 Nov 2026

These models have open or downloadable weights but are too large to fine tune on one machine, so they are tier D. This stage only compares them for serving and for reference.

## **1. Spec Table**

| Model | Total and active | Released | License | Weights size | Status |
|---|---|---|---|---|---|
| Inkling 2 Small | 594B, 17B active, 48 layers, 8 of 384 experts, 1M context | weights 12 Nov 2026 | Apache 2.0 | 4 bit about 310 GB | size from earlier research, rest verified |
| Kimi K3.1 Pro | 2.8T, 104B active, 896 experts with 16 picked, 1M context | weights 4 Nov 2026, one source says 5 Nov | Kimi K3.1 Pro License | about 1.56 TB of files, MXFP4 trained | verified |
| Muse Spark Open | 816B, 42B active, 1M context | 15 Oct 2026 | Muse Model License | about 530 GB at the smallest official size | verified |
| Qwen3.8 2.4T A95B | 2.4T, 95B active | 12 Aug 2026 | qwen3.8 max license | not checked | verified |
| Qwen3.8 Flash Next | 125B, 6B active | 26 Aug 2026 | Qwen Community License 1.0 | about 74 GiB at 4 bit in one runtime | verified, size conflict |
| DeepSeek V4 Flash | 284B in the preview, 304B on NVIDIA's card for the 0731 update, 13B active | 23 Apr 2026, update 30 Jul 2026 | MIT | not checked | size conflict |
| Nemotron 3.5 Bolt 74B A9B | 74B, 9B active, 2 bit copy about 24 GB | 15 Oct 2026 | OpenMDW 1.1 | about 24 GB at 2 bit | one source |
| Olala mini 7A1B | 7B, 1B active | not rechecked | Apache 2.0 | not checked | earlier draft |
| Ripple 65B Preview | 65B | not rechecked | Apache 2.0 | not checked | earlier draft |

The DeepSeek size is still open. The New Stack says the 0731 update kept the 284B architecture, while NVIDIA's model card lists 304B for it. Check the Hugging Face file list before you write either number in the decision record.

## **2. Licenses That Need a Read**

| Model | Condition |
|---|---|
| Kimi K3.1 Pro | Model API hosts above 20 million dollars in revenue over 12 months need a separate agreement |
| Muse Spark Open | Products must show "Built with Muse Spark Open", and model as a service providers above 20 million dollars in revenue need a written agreement |
| Qwen3.8 Flash Next | Qwen Community License 1.0, not Apache |
| Inkling 2 Small | Apache 2.0, with the same separate acceptable use policy idea as Mini |

## **3. Memory Math**

Weights set the lower bound for the number of cards. Divide the weights by the usable memory on one card.

$$
\text{cards} \ge \frac{\text{weights (GB)}}{\text{card (GB)} - 1}
$$

| Step | Inkling 2 Small | Kimi K3.1 Pro |
|---|---|---|
| 1. Weights | 310 GB | 1,560 GB |
| 2. Usable per 80 GB card | 79 GB | 79 GB |
| 3. Divide | 310 / 79 = 3.92 | 1,560 / 79 = 19.75 |
| 4. Round up | 4 cards | 20 cards |

$$
\boxed{\text{At least 4 cards of 80 GB for Inkling 2 Small and 20 for Kimi K3.1 Pro, weights only}}
$$

The 4 bit rule of thumb gives 594 × 0.5 = 297 GB for Inkling 2 Small, so the 310 GB figure fits the usual extra for layers kept in higher precision. Add the attention cache and batch memory on top before you plan real serving.

## **4. Traps**

| Trap | What to do |
|---|---|
| Treating open weights as open source | Read each license file |
| Planning from a headline size | Use the file sizes from the model page |
| Using active parameters for memory | Use total parameters |
| Forgetting the cache at 1M context | Add it to the weights |
| Old draft entries | Olala and Ripple were not rechecked, mark them as earlier draft |

## **5. Pre-flight Answers**

| Question | Answer |
|---|---|
| Can I fine tune these on one machine | No |
| Which is the easiest to serve | Inkling 2 Small at 4 cards, or Qwen3.8 Flash Next at about 74 GiB |
| Which has the cleanest license | DeepSeek V4 Flash with MIT, then Inkling 2 Small with Apache 2.0 |
| Is Qwen4 here | No, Qwen4 Max is API only |

## **6. Sources**

| Source | Used for | Date |
|---|---|---|
| https://casrai.org/news/thinking-machines-inkling-2-small-open-weights-2026 | Release date, Apache 2.0 | read 28 Nov 2026 |
| https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Small.md | 594B, 17B, checkpoints | read 28 Nov 2026 |
| https://huggingface.co/moonshotai/Kimi-K3.1-Pro | Kimi spec and license badge | read 28 Nov 2026 |
| https://fourweekmba.com/ai-kimi-k3-1-pro-weights-out-leads-opus-5-5-on-16-of-37-tests/ | Experts, files size, license terms | read 28 Nov 2026 |
| https://pytorch.org/blog/muse-spark-open/ | Spark Open size and license | read 28 Nov 2026 |
| https://www.digitalapplied.com/blog/meta-muse-spark-open-open-weights-licence-hardware | 530 GB, license clauses | read 28 Nov 2026 |
| https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-v4-flash-0731 | 304B on the card | read 28 Nov 2026 |
| https://thenewstack.io/deepseek-v4-flash-open-weights/ | 284B, MIT | read 28 Nov 2026 |
| https://github.com/kiarina/kiapi/releases/tag/v0.8.0 | Flash Next 74 GiB resident | read 28 Nov 2026 |