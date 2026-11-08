# **Stage 03 Data Engineering: Index**

**Outcome of the stage.** A dataset you can defend: sourced, licensed, deduplicated, decontaminated, format-checked, versioned and documented. **Exit criterion.** A public dataset on the Hub with a data card, a license table, a decontamination report, and a held-out split that no training run ever touched.

This file is the map for the stage. It lists the notes in reading order, where each note's outputs go, the running example, the notation used across notes, and the GPU ladder that every calculation table uses. Update it after each new note.

**Contents.**

1. Reading Order and Status
2. Folder Map
3. The Running Example
4. Notation Across Notes
5. The GPU Ladder
6. Quick Calculations for the First Day in an Organization
7. Credits and Where to Run
8. Conventions
9. Assumptions and Corrections Log

---

## 1. Reading Order and Status

| note | file | topic | status |
|---|---|---|---|
| 01 | [01-why-data-and-data-shapes.md](01-why-data-and-data-shapes.md) | why data is first, trainer shapes, row share versus token share | done |
| 02 | [02-formats-and-chat-templates.md](02-formats-and-chat-templates.md) | storage formats, chat templates, loss masks | done |
| 03 | 03-dataset-catalog-and-licenses.md | where data comes from, what you may use | next |
| 04 | 04-quality-and-deduplication.md | exact and near-duplicate removal, MinHash | planned |
| 05 | 05-decontamination.md | removing evaluation leaks | planned |
| 06 | 06-synthetic-data.md | rubric generation, model collapse | planned |
| 07 | 07-pii-and-eu-ai-act.md | scrubbing, provenance, documentation | planned |
| 08 | 08-domain-dataset-and-splits.md | build, split, version | planned |
| 09 | 09-preflight-and-acceptance-gate.md | the exit checklist | planned |

Labs from the source stage live in `labs/` and are not notes:

| lab | file |
|---|---|
| 1 | labs/01-inspect-public-dataset.md |
| 2 | labs/02-build-small-domain-dataset.md |
| 3 | labs/03-build-dpo-dataset.md |
| 4 | labs/04-decontamination-pipeline.md |
| 5 | labs/05-format-consistency-check.md |
| 6 | labs/06-compliance-pass.md |

The next stage is Stage 04 Training Core, which reads the dataset this stage produces. The roadmap order was not pasted, so the order of notes 03 to 09 is assumed from the stage structure.

---

## 2. Folder Map

```text
03_data_engineering/
├── notes/        00-index.md and notes 01 to 09 (markdown only)
├── labs/         the six lab write-ups
├── configs/      yaml: dedup threshold, split seed, n-gram settings, pinned versions
├── formats/      sample jsonl per format, marker count reports
├── quality/      dedup report, length distribution plot
├── synthetic/    rubric yaml, generated candidates
├── domain_dataset/  final train, validation, test, data card, provenance table
└── results/      filled results tables, decontamination report, acceptance gate
```

Which note fills which folder (planned, one file per note, same number prefix):

| note | files it produces |
|---|---|
| 01 | results/01-gpu-results.md |
| 02 | formats/02-conversion-samples.jsonl, formats/02-marker-count-report.json |
| 03 | domain_dataset/03-provenance-table.md, results/03-license-audit.md |
| 04 | quality/04-dedup-report.json, quality/04-length-distribution.png, configs/04-dedup.yaml |
| 05 | results/05-decontamination-report.json, configs/05-ngram.yaml |
| 06 | synthetic/06-rubric.yaml, synthetic/06-candidates.jsonl |
| 07 | domain_dataset/07-pii-audit.md |
| 08 | domain_dataset/ (splits and data card), configs/08-split.yaml |
| 09 | results/09-acceptance-gate.md |

Notes stay self-contained, with the notebook cells inside them. The other folders hold only what those cells write out.

---

## 3. The Running Example

Six rows about deep learning, each with one planned defect. Every note in the stage fixes one kind of defect.

| id | question | answer | defect | fixed in |
|---|---|---|---|---|
| r1 | What does softmax do? | Gives probabilities. | none | |
| r2 | What does the softmax do? | Softmax gives probabilities. | near-duplicate of r1 | note 04 |
| r3 | Why use warmup? | Avoids unstable early updates. | none | |
| r4 | What is attention? | Tokens weigh other tokens. | none | |
| r5 | What does dropout do? | Randomly zeroes activations. | copied from the evaluation set | note 05 |
| r6 | My email is asha@example.com, why is my loss flat? | Check the learning rate. | contains an email address | note 07 |

The path is 6 rows, then 5 after deduplication, then 4 after decontamination, with r6 scrubbed. A 4-turn conversation c1 extends r1:

| turn | role | words |
|---|---|---|
| 1 | user | What does softmax do? |
| 2 | assistant | Gives probabilities. |
| 3 | user | Why exp? |
| 4 | assistant | Keeps scores positive. |

**Toy tokenizer.** One word is one token and punctuation is ignored. Two invented templates add markers:

| template | each turn | once at the start |
|---|---|---|
| A, ChatML style | start marker, role, words, end marker | nothing |
| B, Llama style | header start, role, header end, words, turn end | one begin marker |

Under template A, r1 has 12 tokens and 3 trained tokens. Under template B it has 15 tokens and 3 trained tokens. The note 01 value of 3 of 7 was the share without a template.

---

## 4. Notation Across Notes

| symbol | meaning | first used |
|---|---|---|
| $x$, $y_t$ | prompt tokens, response token $t$ | note 01 |
| $m_t$ | loss mask, 1 on trained tokens | note 01 |
| $\ell_t$, $L$, $\bar L$ | token loss, loss, mean over rows | note 01 |
| $\pi_\theta$, $\theta$ | model distribution, parameters | note 01 |
| $z$, $p$, $\mathbf{1}_y$ | logits, probabilities, one-hot target | note 01 |
| $N$, $n_k$, $\rho_k$ | rows, rows of source $k$, row share | note 01 |
| $\tau_k$, $\tau$, $\sigma_k$ | tokens of source $k$, total tokens, token share | note 01 |
| $c_k$, $w_k$ | rows about topic $k$, vote share of topic $k$ | note 01 |
| $r$ | trained share of tokens | note 01, refined in note 02 |
| $T$, $B$, $V$, $b$ | sequence length, batch size, vocabulary size, bytes per logit | note 01 |
| $P$, $M$, $G$ | parameters, memory in bytes, number of GPUs | note 01 |
| $H_{\text{free}}$, $H_{\text{lost}}$, $f$ | free hours, lost hours, lost share | note 01 |
| $n$, $C$, $k$, $c$ | turns, content tokens, overhead per turn, overhead once | note 02 |
| $d$, $\Delta$, $q$, $W$ | default system tokens, template length gap, share rendered with B, padding waste | note 02 |

**Reading tips.** The letter $t$ always means a token position. Shares use Greek letters ($\rho$ rows, $\sigma$ tokens) so they never clash with $t$ or $T$. The letter $c$ means a constant overhead in note 02, while $c_k$ means a row count in note 01. The letter $q$ is the mixing share so it never clashes with $\pi_\theta$. The planned symbols $\beta$ and $\eta$ are not used in this stage.

---

## 5. The GPU Ladder

The formulas in the notes use memory only, so GPUs are grouped by VRAM tier. Each tier stands for every GPU with that memory. Calculation tables in every note use these seven tiers.

| tier | stands for | Modal $ per hour | where to run |
|---|---|---|---|
| 16 GB | T4 | 0.59 | Lightning free hours (75), Modal |
| 24 GB | L4, A10 | 0.80 (L4) | Lightning free hours (L4, 31), Modal |
| 48 GB | L40S | 1.95 | Lightning free hours (5), Modal |
| 80 GB | A100 80GB, H100 | 2.50 and 3.95 | Modal, Lightning paid |
| 141 GB | H200 | 4.54 | Modal, Lightning paid |
| 180 GB | B200 | 6.25 | Modal |
| 288 GB | B300 | 7.10 | Modal, needs CUDA 13.1 or newer |

**Notes on the table.**

- Prices are the Modal per second rates times 3600, rounded to cents, read on the Modal pricing page in November 2026. Prices change, so read the page again before you plan.
- The 288 GB for the B300 is derived from Modal's statement of 2,304 GB for 8 GPUs, and is not stated directly.
- The RTX PRO 6000 (96 GB) is available on both platforms, but 96 GB sits between the 80 and 141 GB tiers, so it is not a tier. Consumer RTX cards (3090 and 4090 at 24 GB, from general knowledge) map to the 24 GB tier.
- Modal lists up to 8 GPUs per container for the B300, B200, H200, H100, A100, L4, T4 and L40S.
- A memory ladder says what fits, not how fast it runs. The Modal docs say Hopper GPUs (H100, H200) have more mature library support than Blackwell GPUs (B200, B300).

**Memory limits per tier, from the formulas in notes 01 and 02.** Illustrative: 85 percent of VRAM is usable for $16P$, and 20 percent of VRAM may hold the logits, with $V = 151{,}936$.

| tier | largest full fine-tune | longest row from logits | batch at $T=1024$ |
|---|---|---|---|
| 16 GB | 0.85 B | 5,265 | 5 |
| 24 GB | 1.275 B | 7,898 | 7 |
| 48 GB | 2.55 B | 15,796 | 15 |
| 80 GB | 4.25 B | 26,326 | 25 |
| 141 GB | 7.49 B | 46,401 | 45 |
| 180 GB | 9.56 B | 59,235 | 57 |
| 288 GB | 15.30 B | 94,776 | 92 |

**GPUs needed for a full fine-tune, with perfect sharding.** $G = \lceil 16P / (0.85 \cdot \text{VRAM}) \rceil$.

| tier | 1B | 8B | 70B |
|---|---|---|---|
| 16 GB | 2 | 10 | 83 |
| 24 GB | 1 | 7 | 55 |
| 48 GB | 1 | 4 | 28 |
| 80 GB | 1 | 2 | 17 |
| 141 GB | 1 | 2 | 10 |
| 180 GB | 1 | 1 | 8 |
| 288 GB | 1 | 1 | 5 |

**Eight GPUs on one machine.** Largest model that full-fine-tunes with perfect sharding: 34.0 B on 8 x 80 GB, 59.9 B on 8 x 141 GB, 76.5 B on 8 x 180 GB and 122.4 B on 8 x 288 GB. Real runs need more room for activations and communication, so treat these as upper bounds.

---

## 6. Quick Calculations for the First Day in an Organization

When someone names a model and a GPU, these five steps take under a minute on paper.

1. **Find the tier.** Match the GPU to the closest VRAM tier in section 5.
2. **Full fine-tune memory.** Multiply the parameter count by 16 bytes. A 7 billion parameter model needs $16 \times 7 = 112$ GB. This is a rule of thumb from general knowledge.
3. **GPU count.** Divide by 85 percent of the VRAM and round up. On 80 GB GPUs, $112 / 68 = 1.65$, so 2 GPUs. On one H200, $112 / 119.85 = 0.93$, so 1 GPU on paper, with activations still to add.
4. **Logits budget.** Logits take $4 \cdot V \cdot T$ bytes per row. For $V = 151{,}936$ and $T = 4096$ that is 2.49 GB per row.
5. **Batch size.** $B_{\max} = \lfloor 0.2 \cdot \text{VRAM} / (4VT) \rfloor$ per GPU.

Then ask three questions that the formulas cannot answer: which CUDA and library versions the cluster has, whether the GPUs are Hopper or Blackwell, and how many GPUs per node can share memory.

A lost-time check uses the lost share $f = H_{\text{lost}} / H_{\text{free}}$. For 10 lost hours on a $30 monthly credit it equals the price per hour over 3, so a B200 at $6.25 loses 2.08 months of credit.

---

## 7. Credits and Where to Run

| platform | free credit | recurring | source |
|---|---|---|---|
| Modal Starter | $30 of compute | monthly | Modal pricing page |
| Lightning AI | 5 credits at signup, 25 more with a card, expiring after 12 months | one time | Lightning pricing page |

- Modal's $30 buys about 50.85 hours of a T4, 37.5 hours of an L4, 15.38 hours of an L40S, 12 hours of an A100 80GB, 7.59 hours of an H100, 6.61 hours of an H200 and 4.8 hours of a B200 at the current rates.
- A secondary source says Lightning gave 15 credits per month before a September repricing. This was not verified on Lightning's own pages, so assume the one-time grant.
- The notebook cells in each note are plain Jupyter. Run them on a Lightning Studio, on a Modal Notebook (check that your plan includes Notebooks), or on your organization's cluster.
- Use the small tiers for the experiment cells, and use the ladder for paper calculations about everything above them.

---

## 8. Conventions

- Note files are named `NN-kebab-topic.md`, two digits, reading order. Artifacts reuse the note's number.
- Markdown only: no blockquotes, no em dashes or spaced dashes in prose, no emoji.
- Code cells use the banner header with indented comment lines, are 25 lines or fewer, and print dividers with the line character, never an equals sign.
- Code: no numpy, no underscore-prefixed names, no wrapper functions around a single call, one short comment or docstring per function.
- Numbers are labelled hand-calculated, measured, expected or illustrative. Measured means produced by the verification script on tiny CPU values. GPU numbers are never invented, and outputs of GPU cells are shown as formats until you fill in your own.
- Each new note appends its symbols to section 4 and its status to section 1.

---

## 9. Assumptions and Corrections Log

**Assumptions.**

- The roadmap order was not pasted, so notes 03 to 09 follow the stage structure.
- Notes are written from the source stage file (Mode B research with the source as input), and every correction to it is listed below.
- The experiment model is Qwen2.5-0.5B-Instruct and the dataset slice is `trl-lib/Capybara`, both small enough for every tier. Read the dataset card for the license before reuse.

**Corrections to the source stage file so far.**

| note | correction |
|---|---|
| 01 | The unpaired preference column is `label`, not `desirable`. |
| 01 | In a preference row with an explicit prompt, `chosen` and `rejected` hold only the assistant turn. |
| 01 | GRPO data is prompt-only with reward code passed to the trainer, and a verifier stored as text for `eval` is unsafe. |
| 01 | Alpaca, ShareGPT and messages are storage formats, not trainer shapes. |
| 01 | The claim "data beats algorithm, every time" is not supported as a general law by the papers read. |
| 02 | A bimodal length histogram is not a reliable sign of mixed templates. Count markers per row instead. |
| 02 | LLaMA-Factory joins instruction and input with one newline, not a blank line. |
| 02 | The Llama 3.1 template adds a date block by default. |
| 02 | The response share of 3 of 7 in note 01 was without a template, and the real value is lower. |

**Still to verify in later notes.** The dataset sizes and licenses in the catalog (note 03), the EU AI Act dates and the Digital Omnibus claim (note 07), the distilabel release date and API (note 06), and the compute threshold of about $3 \times 10^{21}$ FLOP (note 07).