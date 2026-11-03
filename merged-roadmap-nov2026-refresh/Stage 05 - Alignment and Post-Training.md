# **Stage 05 — Alignment and Post-Training**

**Libraries:** `trl` 1.17.1 (DPO, KTO, GRPO, RLOO, Reward, Distillation) · `peft` 0.21.2 · `verl` 0.9.1 · `OpenRLHF` 0.11.2 · `SkyRL` 0.4.0
**Outcome:** take an SFT checkpoint and either align it to preferences or train it against a verifiable reward — with evidence that the alignment helped and did not break the base model.

## **Preference optimization**

**What SFT teaches, and what it does not.** SFT teaches **format** (chat template, tool use, JSON output), **style** (voice, tone, length, structure), and **behavior** (instruction-following, a domain task). It does not teach **preferences** — that is DPO. It does not teach **reasoning** — that is GRPO.

**Why preference optimization.** SFT teaches what good output looks like. Preference optimization teaches the *ordering* between good and bad outputs. The RLHF objective (reward model + PPO) can be re-derived as a supervised binary classification loss over preference pairs: the Bradley-Terry model says `P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))`, the optimal reward is `r*(x, y) = β log(π(y|x) / π_ref(y|x)) + β log Z(x)`, and substituting back cancels `Z(x)`. That cancellation is the entire DPO paper — no reward model, no RL loop, no critic [32].

```python
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """policy = the model being trained, ref = frozen reference (initial model)."""
    chosen_logratio = policy_chosen_logp - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp
    loss = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()
    return loss
```

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model="Qwen/Qwen3-1.7B",
    args=DPOConfig(output_dir="./dpo-model", beta=0.1),
    train_dataset=dataset,
)
trainer.train()
```

The reference model is created for you, or derived from the adapter when using PEFT. Truncation is controlled by `max_length` only — `max_prompt_length` and `max_completion_length` were removed, so filter overlong examples beforehand.

**The DPO family, selected by data shape**

| Variant | Mechanism | Data shape it needs | When it wins |
|---|---|---|---|
| **DPO** | Direct preference loss against a reference model | `(prompt, chosen, rejected)` | Clean pairs — the default |
| **IPO** | Replaces the sigmoid with MSE, bounding the margin | Noisy pairs where DPO overfits | Noisy preference data |
| **KTO** | Kahneman-Tversky; no pairs required | `(prompt, completion, desirable: bool)` | Real-world thumbs-up logs without A/B pairs |
| **SimPO** | Removes the reference model; length-normalized log-prob with a margin | Pairs, but memory-bound | Half the memory (no reference model) |
| **ORPO** | SFT and preference losses combined in one stage | Pairs | A single-stage pipeline |
| **Iterative DPO** | Generate fresh pairs with the current policy, label, repeat | A reward model to label with | Static data has saturated |

`KTO` and `Distillation` are stable trainers in TRL 1.17.1; `IPO`, `ORPO`, `CPO`, `GKD` remain in the experimental tier and can disappear in a minor release — pin and verify before a long run. The stable class names are `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `RewardTrainer`, `RLOOTrainer`, `GRPOTrainer`, and the Distillation trainer; the experimental ones are imported from their module, for example `from trl.experimental.orpo import ORPOTrainer`.

**Reward modeling** (`RewardTrainer` / `RewardConfig`) takes the same `chosen`/`rejected` format and produces a reward model for RL or for labeling iterative pairs.

## **RL fine-tuning: PPO, GRPO, RLVR**

**The case for RL.** Reasoning models (o1, o3, DeepSeek R1 and R1-Zero) are produced by RL fine-tuning on tasks with verifiable rewards — math answers, passing code tests, format checks. DPO cannot do this: it learns from static preference data, not from the model's own attempts.

**PPO** (Schulman et al., 2017) [34] — policy + reference + reward + **critic**, four models in memory (about 28 GB minimum for a 7B), a clipped surrogate objective, and a tuning surface of KL coefficient, clipping range, and value-loss coefficient. In 2026 it is rarely the right starting point for a new project. Note that TRL removed `PPOTrainer` in 1.13.0, so a PPO run today means a separate library (verl, OpenRLHF) or your own loop [6].

**GRPO** (Shao et al., 2024, DeepSeekMath) [33] — policy + reference + reward, **no critic**. For each prompt, sample G rollouts; the group mean is the baseline and advantages are normalized within the group. About 30% less memory than PPO and less hyperparameter-sensitive.

```python
def grpo_loss(policy_logps, ref_logps, rewards, group_size, beta=0.04, clip_range=0.2):
    """
    policy_logps: (B*G,) log probs of chosen tokens under policy
    ref_logps:    (B*G,) log probs of chosen tokens under ref
    rewards:      (B*G,) scalar reward per rollout
    """
    kl = policy_logps - ref_logps
    rewards_grouped = rewards.view(-1, group_size)
    advantages = (rewards_grouped - rewards_grouped.mean(-1, keepdim=True)) / (
        rewards_grouped.std(-1, keepdim=True) + 1e-8
    )
    advantages = advantages.view(-1)
    ratio = torch.exp(policy_logps - policy_logps.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    loss = -torch.min(surr1, surr2).mean() + beta * kl.mean()
    return loss
```

```python
from trl import GRPOConfig, GRPOTrainer

def math_reward(completions, answer, **kwargs):
    """Reward: 1.0 if completion contains the correct answer, 0.0 otherwise."""
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if match:
            try:
                rewards.append(1.0 if abs(float(match.group(1)) - float(correct_answer)) < 1e-3 else 0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=math_reward,
    args=GRPOConfig(output_dir="./grpo-model", num_generations=8, max_new_tokens=512, beta=0.04),
    train_dataset=dataset,
)
trainer.train()
```

**RLVR (RL from verifiable rewards).** When the reward is programmatically checkable, skip the learned reward model entirely. R1-Zero used rule-based accuracy rewards for math and code. The 2025–2026 refinements in the source material: **DAPO** (open RL system with rule-based outcome rewards, 50% AIME 2024 accuracy with Qwen2.5-32B in half the baseline steps), **Dr.GRPO** (response-length bias in reward normalization), **CISPO** (MiniMax-M1 Team, 2025) (clipping replacement for training stability), **GFPO** (group-filtered preference optimization, addresses length inflation), **VIGOR** (variance-guided online rollout allocation, 2.3× fewer rollouts on math).

**RL algorithm selection**

| Task | Algorithm |
|---|---|
| Reasoning with verifiable reward (math, code) | GRPO (or DAPO / Dr.GRPO refinements) |
| General chat alignment with preference data | DPO (or KTO / SimPO if the data shape differs) |
| Output quality from human feedback (legacy) | PPO — still works, but GRPO usually wins |
| Long CoT reasoning on an SFT-seeded model | GRPO with iterative SFT merges |
| Reward hacking (the model gaming the reward) | Switch to rule-based RLVR, or add a KL drift constraint |

**The consensus pipeline**
1. SFT (QLoRA or LoRA) on 10K–100K high-quality instructions.
2. GRPO (or DPO/KTO depending on data) on top of the SFT checkpoint.
3. Optional iterative DPO to refine preferences.
4. Merge LoRA → quantize → serve.

## **Distillation**

Train a small model to mimic a large one — teacher → student — usually with a KL term between token distributions: `L = α·KL(p_student ‖ p_teacher) + (1-α)·CE(y, p_student)`.

Use it when you need a smaller, faster, cheaper model of similar quality, or when you cannot serve the teacher (cost, latency, licensing). Two caveats from the source material that remain true: distilled models are typically **less calibrated** and have **different optimal quantization thresholds**, and supervising on the teacher's outputs alone (SFT) discards information the logit distribution carries. Always re-evaluate the distilled model on your actual workload.

**Tools:** `transformers` + a custom training loop, `accelerate` for multi-GPU, TextBrewer for task-specific distillation, `distilwhisper`-style recipes for ASR, and the DeepSeek-R1-Distill family as the 2025–2026 standard for "reasoning model at small size".

## **Alignment mental model**

1. PEFT (the LoRA family) is the default for SFT. Vanilla LoRA r=16, all-linear, α=32, lr 1e-4 to 5e-4. Switch variants only when measured.
2. QLoRA is the de facto standard for single-GPU: NF4 + double quantization + LoRA in BF16.
3. DPO is the default for preferences. Siblings exist for different data shapes.
4. GRPO has replaced PPO for reasoning-model alignment. RLVR with rule-based rewards is the frontier.
5. The standard pipeline is SFT → DPO/GRPO. No RL works without a seeded SFT base.
6. Distillation is for compression, not new behavior.

## **The post-training stack**

```
Base Pretrained Model
        ↓
[Stage 1] SFT — Supervised Fine-Tuning
        ↓
[Stage 2] Preference Optimization (DPO / SimPO / KTO / ORPO)
        ↓
[Stage 3] Reasoning RL (GRPO / DAPO / Dr.GRPO / GSPO / GRPO-λ / RLVR)
        ↓
[Stage 4] Agentic RL — multi-turn tool use (Stage 9)
        ↓
Quantize + Export + Serve
```

## **Preference optimization methods**

| Method | Year | Reference model | Reward model | Key formula |
|---|---|---|---|---|
| PPO (RLHF) | 2022 | Yes | Yes | Complex actor-critic |
| DPO | 2023 | Yes (frozen) | No | Direct from pairs |
| IPO | 2023 | Yes | No | Better on small data |
| KTO | 2024 | Yes | No | Binary feedback |
| SimPO | 2024 | **No** | No | Average log-probability reward |
| ORPO | 2024 | **No** | No | Single-stage SFT + alignment |

**DPO objective:**

```
L_DPO = -E [ log σ ( β × [
    (log π_θ(y_w|x) - log π_ref(y_w|x))
  - (log π_θ(y_l|x) - log π_ref(y_l|x))
]) ]
β = 0.1–0.2 (KL penalty)
```

Best practice from the merge: mix roughly 80% SFT data with 20% preference data during DPO to prevent forgetting.

## **Reasoning RL: the GRPO family**

**GRPO — Group Relative Policy Optimization:**

```
For each prompt q:
  1. Sample G responses {o_1,...,o_G} (G = 8–64)
  2. Score: r_i = verifier(o_i, ground_truth)
  3. Advantage: A_i = (r_i - mean(r)) / std(r)
  4. Policy gradient update + KL regularization
No critic → 50% memory savings vs PPO
```

The variants that matter, and what each actually changes:

| Method | Key innovation | Use it when |
|---|---|---|
| GRPO | Group-relative advantage, no critic | General RLVR |
| DAPO | Clip-higher + dynamic sampling + token-level loss + overlong reward shaping | Long chain-of-thought reasoning |
| Dr.GRPO | Removes std normalisation and KL | Sparse rewards |
| **GSPO** | Sequence-level importance ratios | **MoE training**, where token-level ratios destabilise expert routing [94][96] |
| GRPO-λ | Critic-free TD learning with eligibility-trace credit assignment | Math and code reasoning; the paper claims 30–40% improvement over GRPO on math benchmarks [95] |
| SAPO | Soft adaptive gate replacing hard clipping | Off-policy drift in long runs |
| LUSPO | Length-unbiased sequence PO, a GSPO modification | Sequence-level IS with length bias problems |
| VESPO | Variational sequence-level soft PO, asymmetric Gamma weighting | Sequence-level stability |

TRL 1.17.1 implements all of these through two arguments, and the distinction matters because getting them wrong silently produces the wrong objective:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="grpo-gspo",
    # GSPO as the TRL paper index specifies it
    importance_sampling_level="sequence",
    loss_type="grpo",
    beta=0.0,             # GSPO sets KL regularisation to zero
    epsilon=3e-4,
    epsilon_high=4e-4,
    gradient_accumulation_steps=1,
    steps_per_generation=4,   # partition the rollout batch into 4 mini-batches
)
```

`GRPOConfig.loss_type` accepts exactly `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo` and `vespo` [102]. There is **no `gspo` value** — GSPO is `importance_sampling_level="sequence"` with `loss_type="grpo"`. TRL also warns that combining `importance_sampling_level="sequence"` with `bnpo`, `dr_grpo`, `dapo` or `cispo` re-weights sequences by completion length instead of optimising the per-sequence objective, and that TRL's `GRPOTrainer` has an `environment_factory` mode for environment-driven rollouts [103][104][105].

## **Verifier design**

The verifier is the actual algorithm. A weak verifier trains a model to exploit the verifier.

```python
# Math verifier
def math_verifier(response: str, ground_truth: str) -> float:
    extracted = extract_answer(response)  # parse \boxed{} or final line
    return 1.0 if extracted == ground_truth else 0.0

# Code verifier
def code_verifier(response: str, test_cases: list) -> float:
    code   = extract_code_block(response)
    passed = run_tests(code, test_cases)
    return len(passed) / len(test_cases)  # partial credit

# Format verifier
def format_verifier(response: str) -> float:
    has_think  = "" in response and "" in response
    has_answer = "" in response
    return 0.2 * has_think + 0.8 * has_answer
```

Partial credit on code is worth the extra effort: a binary pass/fail signal on a 200-test suite produces a reward gradient so sparse that most groups have zero variance, and GRPO cannot learn from a group where every sample scores identically.

Reward-hacking mitigation, in order of how often it matters: hold out a verifier variant the model never trains against; check that partial credit correlates with the fraction of tests passed rather than with code length; log the reward distribution per group, because a group where all rewards are equal contributes no gradient and is usually a formatting bug; and audit a sample of the highest-reward trajectories by hand every few hundred steps.

## **Gradient-free fine-tuning**

```
1. Sample parameter perturbations: {ε_1, ..., ε_N}
2. Evaluate via inference + verifiable reward: {r_1, ..., r_N}
3. Estimate gradient: ∇ ≈ (1/Nσ²) Σ(ε_i × r_i)
4. Update: θ ← θ + α∇
No backpropagation. Works natively in quantized environments.
```

**QES (Quantized Evolution Strategies)** fine-tunes directly in NF4/INT4 and sidesteps gradient instability through the quantisation itself. It is reported as competitive with GRPO on mathematical reasoning, and it remains emerging. For a learner it is a useful weekend project: it requires no backward pass, runs against the same verifiers, and makes the estimator's variance visible in a way that backprop-based training never does.

## **Alignment pre-flight checklist**

- [ ] State the LoRA forward pass in one equation.
- [ ] What is the default LoRA config (r, α, target_modules, lr)?
- [ ] How much memory does QLoRA save versus full FT on a 70B model?
- [ ] What does DPO optimize? Why is it "direct"?
- [ ] Compare DPO, IPO, KTO, SimPO, ORPO. When does each win?
- [ ] What is GRPO? Why does it beat PPO for reasoning?
- [ ] What is RLVR? Name two examples of verifiable rewards.
- [ ] State the consensus pipeline.
- [ ] When do you use full fine-tuning instead of LoRA?
- [ ] What is DoRA? When does it beat LoRA?
- [ ] Which TRL trainers are stable in 1.17.1, and which were removed?

---

---

## **Worksheet additions**

## **Scope**
Stage 4 is the step most people conflate with SFT. SFT teaches what a good answer looks like. Preference optimization teaches the ordering between a good and a bad answer. RL teaches the model from its own attempts, scored by something checkable.

**Exit criterion:** one preference-optimized checkpoint and one RL checkpoint, each with a held-out win rate, a capability regression table, and a written answer to "what went wrong".

---

## **Decision table**
| Situation | Method | Why |
|---|---|---|
| Format, style, instruction-following | SFT (Stage 3) | Imitation is enough |
| Clean preference pairs | DPO | The default alignment move |
| Noisy preference pairs | IPO | MSE loss bounds the margin |
| Binary feedback, no pairs | KTO | Works on thumbs-up style labels |
| Memory-bound, no reference model | SimPO | Half the memory |
| One stage for SFT + preference | ORPO | Combined loss |
| Static preference data saturated | Iterative DPO | Re-label with the current policy |
| Reasoning with a programmatic verifier | GRPO / RLVR | Learns from its own attempts |
| Human-feedback legacy pipeline | PPO | Still works, but TRL removed its trainer |
| Reward hacking detected | Switch to rule-based RLVR, or add a KL drift constraint | Learned rewards are gameable |
| Smaller model at the same quality | Distillation | Compression, not new behavior |

---

### **DPO**
- [ ] The Bradley-Terry assumption `P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))`
- [ ] The optimal reward `r*(x, y) = β log(π(y|x) / π_ref(y|x)) + β log Z(x)`
- [ ] Why `Z(x)` cancels and the loss becomes supervised
- [ ] `DPOTrainer`, `DPOConfig`, `beta`, loss types
- [ ] Dataset format: `prompt`, `chosen`, `rejected`
- [ ] Truncation with `max_length` only (`max_prompt_length` / `max_completion_length` were removed)
- [ ] Reference model created automatically, or derived from the adapter under PEFT
- [ ] The reference-free variants and when they help

### **The siblings**
- [ ] IPO — sigmoid replaced with MSE; bounds the margin
- [ ] KTO — Kahneman-Tversky; binary `desirable` labels; no pairs required
- [ ] SimPO — no reference model; length-normalized log-prob with a margin
- [ ] ORPO — SFT and preference losses combined in a single stage
- [ ] CPO, iterative DPO, and online variants — awareness only
- [ ] Which of these are stable in TRL 1.17.1 (DPO, KTO) and which live in `trl.experimental`

### **Reward modeling**
- [ ] `RewardTrainer` / `RewardConfig` on `chosen` / `rejected` data
- [ ] Using a reward model to label iterative preference pairs
- [ ] Reward-model calibration and its failure modes

---

### **PPO**
- [ ] Policy, reference, reward, and critic — four models resident at once
- [ ] Memory cost: about 28 GB minimum for a 7B policy plus the rest of the stack
- [ ] The clipped surrogate objective
- [ ] KL coefficient, clipping range, value-loss coefficient
- [ ] That TRL removed `PPOTrainer` in 1.13.0, so PPO today means verl or OpenRLHF

### **GRPO**
- [ ] Policy + reference + reward, no critic
- [ ] Group sampling: G rollouts per prompt
- [ ] Group-relative baseline and within-group advantage normalization
- [ ] Roughly 30% less memory than PPO and less hyperparameter sensitivity
- [ ] `GRPOTrainer`, `GRPOConfig`, `num_generations`, `beta`, `max_new_tokens`
- [ ] Reward functions: plain Python, reward models, or a list of both

### **RLVR and its refinements**
- [ ] RLVR: skip the learned reward when the reward is programmatically checkable
- [ ] DAPO — rule-based outcome rewards, 50% AIME 2024 accuracy with Qwen2.5-32B in half the baseline steps
- [ ] Dr.GRPO — addresses response-length bias in reward normalization
- [ ] CISPO — clipping replacement for training stability
- [ ] GFPO — group-filtered preference optimization; addresses length inflation
- [ ] VIGOR — variance-guided rollout allocation; 2.3× fewer rollouts on math
- [ ] Reward hacking: how to detect it and when to switch to rule-based rewards

### **Distillation**
- [ ] Teacher → student framing
- [ ] The KL objective `L = α·KL(p_student ‖ p_teacher) + (1-α)·CE(y, p_student)`
- [ ] When distillation beats SFT on the teacher's outputs (it usually does not)
- [ ] Calibration loss and different optimal quantization thresholds in distilled models
- [ ] `transformers` + custom loop, `accelerate`, TextBrewer, `distilwhisper` recipes
- [ ] The DeepSeek-R1-Distill family as the reference implementation

---

### **The DPO loss, in full**
```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """policy = the model being trained, ref = frozen reference (initial model)."""
    chosen_logratio   = policy_chosen_logp   - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp
    logits = beta * (chosen_logratio - rejected_logratio)
    return -F.logsigmoid(logits).mean()
```

### **Training with DPO**
```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model="Qwen/Qwen3-4B-Instruct-2507",
    args=DPOConfig(
        output_dir="./dpo-model",
        beta=0.1,
        max_length=2048,          # max_prompt_length / max_completion_length are gone
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        bf16=True,
    ),
    train_dataset=dataset,
)
trainer.train()
```

### **Verifying the merged LoRA and the reference model**
```python
# Under PEFT, the reference model is the frozen base with the adapter disabled.
# Under a full fine-tune, pass an explicit reference model.
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

trainer = DPOTrainer(
    model=peft_model,
    ref_model=None,              # adapter-disabled base
    args=DPOConfig(output_dir="./dpo-model", beta=0.1),
    train_dataset=dataset,
)
```

### **A verifiable reward for GRPO**
```python
import re

def math_reward(completions, answer, **kwargs):
    """1.0 when the completion contains the correct numeric answer."""
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if not match:
            rewards.append(0.0)
            continue
        try:
            rewards.append(1.0 if abs(float(match.group(1)) - float(correct_answer)) < 1e-3 else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards
```

```python
from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B-Instruct-2507",
    reward_funcs=math_reward,
    args=GRPOConfig(
        output_dir="./grpo-model",
        num_generations=8,        # G rollouts per prompt
        max_new_tokens=512,
        beta=0.04,                # KL coefficient against the reference
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        bf16=True,
    ),
    train_dataset=dataset,        # {prompt, answer} rows
)
trainer.train()
```

### **Combining a verifiable reward with a format reward**
```python
def combined_reward(completions, answer, **kwargs):
    """Correctness dominates; format compliance breaks ties.

    Two rewards rather than one makes the trade-off explicit and keeps a
    format-only shortcut from paying off.
    """
    correct = math_reward(completions, answer, **kwargs)
    formatted = [
        1.0 if "\\boxed{" in c else 0.0
        for c in completions
    ]
    return [c + 0.1 * f for c, f in zip(correct, formatted)]
```

### **Monitoring for reward hacking**
```python
# Track these every N steps and plot them together:
#   - mean reward per group
#   - within-group reward variance (falls to zero when the model collapses)
#   - mean completion length (rises when length inflation kicks in)
#   - KL divergence from the reference (rises when the policy drifts)
# A reward curve that rises while variance collapses and length explodes is
# the classic signature of reward hacking, not of learning.
```

### **Distillation loop**
```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=1.0):
    """alpha weights the KL term against the teacher's distribution."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce
```

---

## **GRPO configuration reference**
| Knob | Typical | Why |
|---|---|---|
| `num_generations` (G) | 8 | 4 is cheaper, 16 gives a smoother baseline |
| `beta` (KL) | 0.04 | Higher pins the policy to the reference |
| Learning rate | 1e-6 to 5e-6 | Far lower than SFT |
| `max_new_tokens` | 256–1024 | Drives rollout cost linearly |
| `max_steps` | 500–2000 | Watch the reward curve, not the step count |
| Reward | rule-based first | Learned rewards are gameable before they are useful |

---

## **Capability regression table**
Every alignment run reports this, whether or not it flatters the result:

| Benchmark | SFT | SFT + DPO | SFT + GRPO | Acceptable delta |
|---|---|---|---|---|
| MMLU | | | | < 1% (DPO), < 3% (GRPO) |
| IFEval | | | | no drop |
| GSM8K / HumanEval | | | | no drop |
| Domain held-out | | | | must rise |
| Format compliance | | | | ≥ 95% after GRPO |
| Mean completion length | | | | flat, unless you intend otherwise |

---

### **Lab 4-1 — DPO from scratch**
- Implement the DPO loss in PyTorch and verify it against TRL's implementation on one batch.
- Train DPO on UltraFeedback with a 4B model.
- Sweep `beta` ∈ {0.05, 0.1, 0.3}; report the held-out win rate and the reward margin for each.
- Compare a separate reference model against LoRA-as-reference.

### **Lab 4-2 — Pairwise data audit**
- Take 200 preference pairs from your own data.
- Label each as obvious win or close call.
- Report the hard-negative rate. If it is below 20%, the pairs are too easy to teach anything — regenerate them.

### **Lab 4-3 — GRPO on GSM8K**
- Train a verifiable reward for GSM8K.
- Run GRPO with G=8 and track mean reward, group variance, length, and KL.
- Plot the four curves and identify where reward and variance diverge.
- Evaluate on the held-out split and on MMLU for regression.

### **Lab 4-4 — Reward-hacking audit**
- Take a GRPO checkpoint and find 20 prompts where it produces the right answer for the wrong reason.
- Measure format compliance, not just accuracy.
- Fix by switching to a stricter verifier and re-running a short training.

### **Lab 4-5 — Distillation**
- Distill a larger model into a 4B student on your domain data.
- Compare student accuracy against the teacher's accuracy on the same held-out set.
- Quantize the student and compare the accuracy delta against the teacher's quantized delta — distilled models often have a different optimal quantization threshold.

---

## **Pitfalls**
- **DPO on easy pairs teaches nothing.** If chosen is obviously better, the gradient is tiny and the model learns nothing.
- **`beta` matters more than the learning rate.** It sets the strength of the KL constraint to the reference.
- **DPO overfits noisy pairs.** IPO's bounded margin exists for this.
- **GRPO without a KL term drifts.** Keep `beta` on and watch the KL curve.
- **Length inflation is the default GRPO failure.** If length rises monotonically, add Dr.GRPO or a length penalty.
- **Format compliance is the cheapest hacking signal.** Check it every eval.
- **PPO is not in TRL any more.** `from trl import PPOTrainer` fails; use verl 0.9.1 or OpenRLHF 0.11.2.
- **KTO is stable now; IPO, ORPO, and CPO are not.** A minor TRL bump can delete the trainer you are calling.
- **Distillation changes calibration.** A distilled model that matches the teacher on accuracy can still be worse on confidence calibration, which changes downstream thresholding.

---

## **Acceptance gate**
- [ ] A DPO run produced a held-out win rate on IFEval, AlpacaEval 2.0, or Arena-Hard-Auto v2.
- [ ] A GRPO run produced a held-out accuracy gain with a logged reward, variance, length, and KL curve.
- [ ] Capability regression table filled in for both runs.
- [ ] Format compliance measured (for GRPO, at or above 95%).
- [ ] The hard-negative rate of your preference data is above 20%.
- [ ] A written paragraph per run explaining what went wrong, not just what improved.

---

## **Resources**
- TRL v1.17.1 release — https://github.com/huggingface/trl/releases/tag/v1.17.1
- verl v0.9.1 release — https://github.com/verl-project/verl/releases/tag/v0.9.1
- OpenRLHF v0.11.2 release — https://github.com/OpenRLHF/OpenRLHF/releases/tag/v0.11.2
- SkyRL on PyPI — https://pypi.org/project/skyrl/
- DPO, Rafailov et al. — https://arxiv.org/abs/2305.18290
- GRPO, Shao et al. — https://arxiv.org/abs/2402.03300
- PPO, Schulman et al. — https://arxiv.org/abs/1707.06347

**Next:** Stage 06 — Frameworks at Scale.
