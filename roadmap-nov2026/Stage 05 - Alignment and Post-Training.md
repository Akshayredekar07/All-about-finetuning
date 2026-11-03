## **Stage 05: Alignment and Post-Training**

**Libraries:** `trl` (DPO, KTO, GRPO, RLOO, Reward, Distillation), `peft`, `verl`, `OpenRLHF`, `SkyRL`

**Outcome:** take an SFT checkpoint and either align it to preferences or train it against a verifiable reward, with evidence that the alignment helped and did not break the base model.

### **1. Scope and Exit Criterion**

### **1.1 Scope**

This stage is the step most people conflate with SFT. SFT teaches what a good answer looks like. Preference optimization teaches the ordering between a good and a bad answer. RL teaches the model from its own attempts, scored by something checkable.

### **1.2 Exit Criterion**

One preference-optimized checkpoint and one RL checkpoint, each with a held-out win rate, a capability regression table, and a written answer to "what went wrong".

### **2. Decision Table**

| **Situation** | **Method** | **Why** |
|---|---|---|
| Format, style, instruction-following | SFT (Stage 04) | Imitation is enough |
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

### **3. Preference Optimization**

### **3.1 What SFT Teaches, and What It Does Not**

SFT teaches **format** (chat template, tool use, JSON output), **style** (voice, tone, length, structure), and **behavior** (instruction-following, a domain task). It does not teach **preferences**, which is DPO's job. It does not teach **reasoning**, which is GRPO's job.

### **3.2 Why Preference Optimization**

SFT teaches what good output looks like. Preference optimization teaches the *ordering* between good and bad outputs. The RLHF objective (reward model plus PPO) can be re-derived as a supervised binary classification loss over preference pairs.

The Bradley-Terry model says:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

The optimal reward is:

$$r^*(x, y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting back cancels $Z(x)$. That cancellation is the entire DPO paper: no reward model, no RL loop, no critic.

### **3.3 DPO Objective**

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)\right]$$

$$\boxed{\beta = 0.1 \text{ to } 0.2 \text{ (KL penalty)}}$$

### **3.4 Worked Example: DPO Loss on One Pair**

Rohit has one preference pair with $\beta = 0.1$ and these log-probabilities:

- Policy chosen: $-10$, reference chosen: $-12$
- Policy rejected: $-15$, reference rejected: $-14$

Step 1: chosen log-ratio.

$$-10 - (-12) = 2$$

Step 2: rejected log-ratio.

$$-15 - (-14) = -1$$

Step 3: difference of log-ratios.

$$2 - (-1) = 3$$

Step 4: scale by $\beta$.

$$0.1 \cdot 3 = 0.3$$

Step 5: apply the loss, with $\sigma(0.3) \approx 0.5744$.

$$-\log(0.5744) \approx 0.5544$$

$$\boxed{\mathcal{L}_{\text{DPO}} \approx 0.554}$$

### **3.5 The DPO Loss in Code**

```python
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """policy = the model being trained, ref = frozen reference (initial model)."""
    chosen_logratio = policy_chosen_logp - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp
    loss = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()
    return loss
```

The full version with imports:

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

### **3.6 Training with DPO**

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

A fuller configuration:

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

The reference model is created for you, or derived from the adapter when using PEFT. Truncation is controlled by `max_length` only. The `max_prompt_length` and `max_completion_length` options were removed, so filter overlong examples beforehand.

### **3.7 Reference Model Under PEFT**

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

### **3.8 The DPO Family, Selected by Data Shape**

| **Variant** | **Mechanism** | **Data shape it needs** | **When it wins** |
|---|---|---|---|
| **DPO** | Direct preference loss against a reference model | `(prompt, chosen, rejected)` | Clean pairs. The default |
| **IPO** | Replaces the sigmoid with MSE, bounding the margin | Noisy pairs where DPO overfits | Noisy preference data |
| **KTO** | Kahneman-Tversky; no pairs required | `(prompt, completion, desirable: bool)` | Real-world thumbs-up logs without A/B pairs |
| **SimPO** | Removes the reference model; length-normalized log-prob with a margin | Pairs, but memory-bound | Half the memory (no reference model) |
| **ORPO** | SFT and preference losses combined in one stage | Pairs | A single-stage pipeline |
| **Iterative DPO** | Generate fresh pairs with the current policy, label, repeat | A reward model to label with | Static data has saturated |

`KTO` and `Distillation` are stable trainers in TRL. `IPO`, `ORPO`, `CPO`, and `GKD` remain in the experimental tier and can disappear in a minor release, so pin and verify before a long run. The stable class names are `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, `RewardTrainer`, `RLOOTrainer`, `GRPOTrainer`, and the Distillation trainer. The experimental ones are imported from their module, for example `from trl.experimental.orpo import ORPOTrainer`.

### **3.9 Preference Optimization Methods Compared**

| **Method** | **Year** | **Reference model** | **Reward model** | **Key idea** |
|---|---|---|---|---|
| PPO (RLHF) | 2022 | Yes | Yes | Complex actor-critic |
| DPO | 2023 | Yes (frozen) | No | Direct from pairs |
| IPO | 2023 | Yes | No | Better on small data |
| KTO | 2024 | Yes | No | Binary feedback |
| SimPO | 2024 | **No** | No | Average log-probability reward |
| ORPO | 2024 | **No** | No | Single-stage SFT + alignment |

Best practice: mix roughly 80% SFT data with 20% preference data during DPO to prevent forgetting.

### **3.10 Reward Modeling**

`RewardTrainer` and `RewardConfig` take the same `chosen`/`rejected` format and produce a reward model for RL or for labeling iterative pairs.

### **3.11 DPO Topic List**

- The Bradley-Terry assumption $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$
- The optimal reward $r^*(x, y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$
- Why $Z(x)$ cancels and the loss becomes supervised
- `DPOTrainer`, `DPOConfig`, `beta`, loss types
- Dataset format: `prompt`, `chosen`, `rejected`
- Truncation with `max_length` only (`max_prompt_length` and `max_completion_length` were removed)
- Reference model created automatically, or derived from the adapter under PEFT
- The reference-free variants and when they help

### **3.12 Siblings Topic List**

- IPO: sigmoid replaced with MSE; bounds the margin
- KTO: Kahneman-Tversky; binary `desirable` labels; no pairs required
- SimPO: no reference model; length-normalized log-prob with a margin
- ORPO: SFT and preference losses combined in a single stage
- CPO, iterative DPO, and online variants: awareness only
- Which of these are stable in TRL (DPO, KTO) and which live in `trl.experimental`

### **3.13 Reward Modeling Topic List**

- `RewardTrainer` / `RewardConfig` on `chosen` / `rejected` data
- Using a reward model to label iterative preference pairs
- Reward-model calibration and its failure modes

### **4. RL Fine-Tuning: PPO, GRPO, RLVR**

### **4.1 The Case for RL**

Reasoning models (o1, o3, DeepSeek R1 and R1-Zero) are produced by RL fine-tuning on tasks with verifiable rewards: math answers, passing code tests, format checks. DPO cannot do this, because it learns from static preference data, not from the model's own attempts.

### **4.2 PPO**

PPO (Schulman et al., 2017) uses a policy, a reference, a reward, and a **critic**: four models in memory (about 28 GB minimum for a 7B). It has a clipped surrogate objective and a tuning surface of KL coefficient, clipping range, and value-loss coefficient. Today it is rarely the right starting point for a new project. TRL removed `PPOTrainer`, so a PPO run today means a separate library (verl, OpenRLHF) or your own loop.

PPO topic list:

- Policy, reference, reward, and critic: four models resident at once
- Memory cost: about 28 GB minimum for a 7B policy plus the rest of the stack
- The clipped surrogate objective
- KL coefficient, clipping range, value-loss coefficient
- TRL removed `PPOTrainer`, so PPO today means verl or OpenRLHF

### **4.3 GRPO**

GRPO (Shao et al., 2024, DeepSeekMath) uses a policy, a reference, and a reward, with **no critic**. For each prompt, sample $G$ rollouts. The group mean is the baseline and advantages are normalized within the group. It uses about 30% less memory than PPO and is less hyperparameter-sensitive.

The procedure per prompt $q$:

1. Sample $G$ responses $\{o_1, \dots, o_G\}$ with $G = 8$ to $64$.
2. Score each one: $r_i = \text{verifier}(o_i, \text{ground truth})$.
3. Compute the advantage:

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

4. Apply a policy gradient update with KL regularization.

There is no critic, which gives about 50% memory savings versus PPO in the full-stack accounting.

### **4.4 Worked Example: GRPO Advantages**

Meera samples $G = 4$ rollouts for one prompt and the verifier returns rewards $r = [1, 0, 0, 1]$.

Step 1: group mean.

$$\text{mean}(r) = \frac{1 + 0 + 0 + 1}{4} = 0.5$$

Step 2: deviations from the mean.

- Rollout 1: $1 - 0.5 = 0.5$
- Rollout 2: $0 - 0.5 = -0.5$
- Rollout 3: $0 - 0.5 = -0.5$
- Rollout 4: $1 - 0.5 = 0.5$

Step 3: population standard deviation.

$$\text{std}(r) = \sqrt{\frac{0.25 + 0.25 + 0.25 + 0.25}{4}} = \sqrt{0.25} = 0.5$$

Step 4: normalized advantages.

- Rollout 1: $0.5 / 0.5 = 1$
- Rollout 2: $-0.5 / 0.5 = -1$
- Rollout 3: $-0.5 / 0.5 = -1$
- Rollout 4: $0.5 / 0.5 = 1$

$$\boxed{A = [1, -1, -1, 1]}$$

If all four rewards were equal, every deviation would be zero and the group would contribute no gradient.

### **4.5 GRPO Loss in Code**

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

### **4.6 Training with GRPO**

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

A verifiable reward on its own:

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

A fuller GRPO configuration:

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

Combining a verifiable reward with a format reward:

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

### **4.7 GRPO Topic List**

- Policy + reference + reward, no critic
- Group sampling: $G$ rollouts per prompt
- Group-relative baseline and within-group advantage normalization
- Roughly 30% less memory than PPO and less hyperparameter sensitivity
- `GRPOTrainer`, `GRPOConfig`, `num_generations`, `beta`, `max_new_tokens`
- Reward functions: plain Python, reward models, or a list of both

### **4.8 RLVR and Its Refinements**

**RLVR (RL from verifiable rewards).** When the reward is programmatically checkable, skip the learned reward model entirely. R1-Zero used rule-based accuracy rewards for math and code. The 2025 to 2026 refinements:

- **DAPO:** open RL system with rule-based outcome rewards, 50% AIME 2024 accuracy with Qwen2.5-32B in half the baseline steps.
- **Dr.GRPO:** addresses response-length bias in reward normalization.
- **CISPO** (MiniMax-M1 Team, 2025): clipping replacement for training stability.
- **GFPO:** group-filtered preference optimization, addresses length inflation.
- **VIGOR:** variance-guided online rollout allocation, 2.3 times fewer rollouts on math.

RLVR topic list:

- RLVR: skip the learned reward when the reward is programmatically checkable
- DAPO, Dr.GRPO, CISPO, GFPO, VIGOR (as described above)
- Reward hacking: how to detect it and when to switch to rule-based rewards

### **4.9 RL Algorithm Selection**

| **Task** | **Algorithm** |
|---|---|
| Reasoning with verifiable reward (math, code) | GRPO (or DAPO / Dr.GRPO refinements) |
| General chat alignment with preference data | DPO (or KTO / SimPO if the data shape differs) |
| Output quality from human feedback (legacy) | PPO: still works, but GRPO usually wins |
| Long CoT reasoning on an SFT-seeded model | GRPO with iterative SFT merges |
| Reward hacking (the model gaming the reward) | Switch to rule-based RLVR, or add a KL drift constraint |

### **4.10 The Consensus Pipeline**

1. SFT (QLoRA or LoRA) on 10K to 100K high-quality instructions.
2. GRPO (or DPO/KTO depending on data) on top of the SFT checkpoint.
3. Optional iterative DPO to refine preferences.
4. Merge LoRA, quantize, serve.

### **5. Reasoning RL: The GRPO Family**

### **5.1 Variants and What Each Changes**

| **Method** | **Key innovation** | **Use it when** |
|---|---|---|
| GRPO | Group-relative advantage, no critic | General RLVR |
| DAPO | Clip-higher + dynamic sampling + token-level loss + overlong reward shaping | Long chain-of-thought reasoning |
| Dr.GRPO | Removes std normalisation and KL | Sparse rewards |
| **GSPO** | Sequence-level importance ratios | **MoE training**, where token-level ratios destabilise expert routing |
| GRPO-$\lambda$ | Critic-free TD learning with eligibility-trace credit assignment | Math and code reasoning; the paper claims 30 to 40% improvement over GRPO on math benchmarks |
| SAPO | Soft adaptive gate replacing hard clipping | Off-policy drift in long runs |
| LUSPO | Length-unbiased sequence PO, a GSPO modification | Sequence-level IS with length bias problems |
| VESPO | Variational sequence-level soft PO, asymmetric Gamma weighting | Sequence-level stability |

### **5.2 Configuring These in TRL**

TRL implements all of these through two arguments, and the distinction matters because getting them wrong silently produces the wrong objective.

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

`GRPOConfig.loss_type` accepts exactly `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo` and `vespo`. There is **no `gspo` value**: GSPO is `importance_sampling_level="sequence"` with `loss_type="grpo"`. TRL also warns that combining `importance_sampling_level="sequence"` with `bnpo`, `dr_grpo`, `dapo` or `cispo` re-weights sequences by completion length instead of optimising the per-sequence objective, and that TRL's `GRPOTrainer` has an `environment_factory` mode for environment-driven rollouts.

### **5.3 GRPO Configuration Reference**

| **Knob** | **Typical** | **Why** |
|---|---|---|
| `num_generations` (G) | 8 | 4 is cheaper, 16 gives a smoother baseline |
| `beta` (KL) | 0.04 | Higher pins the policy to the reference |
| Learning rate | 1e-6 to 5e-6 | Far lower than SFT |
| `max_new_tokens` | 256 to 1024 | Drives rollout cost linearly |
| `max_steps` | 500 to 2000 | Watch the reward curve, not the step count |
| Reward | rule-based first | Learned rewards are gameable before they are useful |

### **6. Verifier Design**

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

Reward-hacking mitigation, in order of how often it matters:

- Hold out a verifier variant the model never trains against.
- Check that partial credit correlates with the fraction of tests passed rather than with code length.
- Log the reward distribution per group, because a group where all rewards are equal contributes no gradient and is usually a formatting bug.
- Audit a sample of the highest-reward trajectories by hand every few hundred steps.

### **6.1 Monitoring for Reward Hacking**

```python
# Track these every N steps and plot them together:
#   - mean reward per group
#   - within-group reward variance (falls to zero when the model collapses)
#   - mean completion length (rises when length inflation kicks in)
#   - KL divergence from the reference (rises when the policy drifts)
# A reward curve that rises while variance collapses and length explodes is
# the classic signature of reward hacking, not of learning.
```

### **7. Distillation**

### **7.1 Overview**

Distillation trains a small model to mimic a large one (teacher to student), usually with a KL term between token distributions:

$$L = \alpha \cdot \text{KL}(p_{\text{student}} \,\|\, p_{\text{teacher}}) + (1 - \alpha) \cdot \text{CE}(y, p_{\text{student}})$$

Use it when you need a smaller, faster, cheaper model of similar quality, or when you cannot serve the teacher (cost, latency, licensing). Two caveats that remain true: distilled models are typically **less calibrated** and have **different optimal quantization thresholds**, and supervising on the teacher's outputs alone (SFT) discards information the logit distribution carries. Always re-evaluate the distilled model on your actual workload.

**Tools:** `transformers` with a custom training loop, `accelerate` for multi-GPU, TextBrewer for task-specific distillation, `distilwhisper`-style recipes for ASR, and the DeepSeek-R1-Distill family as the 2025 to 2026 standard for "reasoning model at small size".

### **7.2 Distillation Loop**

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

### **7.3 Distillation Topic List**

- Teacher to student framing
- The KL objective $L = \alpha \cdot \text{KL}(p_{\text{student}} \,\|\, p_{\text{teacher}}) + (1 - \alpha) \cdot \text{CE}(y, p_{\text{student}})$
- When distillation beats SFT on the teacher's outputs (it usually does not)
- Calibration loss and different optimal quantization thresholds in distilled models
- `transformers` with a custom loop, `accelerate`, TextBrewer, `distilwhisper` recipes
- The DeepSeek-R1-Distill family as the reference implementation

### **8. Gradient-Free Fine-Tuning**

The procedure:

1. Sample parameter perturbations $\{\epsilon_1, \dots, \epsilon_N\}$.
2. Evaluate via inference and a verifiable reward: $\{r_1, \dots, r_N\}$.
3. Estimate the gradient:

$$\nabla \approx \frac{1}{N \sigma^2} \sum_{i=1}^{N} \epsilon_i \, r_i$$

4. Update:

$$\theta \leftarrow \theta + \alpha \nabla$$

There is no backpropagation, and the method works natively in quantized environments.

**QES (Quantized Evolution Strategies)** fine-tunes directly in NF4/INT4 and sidesteps gradient instability through the quantisation itself. It is reported as competitive with GRPO on mathematical reasoning, and it remains emerging. For a learner it is a useful weekend project: it requires no backward pass, runs against the same verifiers, and makes the estimator's variance visible in a way that backprop-based training never does.

### **9. Alignment Mental Model**

1. PEFT (the LoRA family) is the default for SFT. Vanilla LoRA r=16, all-linear, $\alpha = 32$, lr 1e-4 to 5e-4. Switch variants only when measured.
2. QLoRA is the de facto standard for single-GPU: NF4, double quantization, and LoRA in BF16.
3. DPO is the default for preferences. Siblings exist for different data shapes.
4. GRPO has replaced PPO for reasoning-model alignment. RLVR with rule-based rewards is the frontier.
5. The standard pipeline is SFT, then DPO/GRPO. No RL works without a seeded SFT base.
6. Distillation is for compression, not new behavior.

### **10. The Post-Training Stack**

```
Base Pretrained Model
        |
[Stage 1] SFT: Supervised Fine-Tuning
        |
[Stage 2] Preference Optimization (DPO / SimPO / KTO / ORPO)
        |
[Stage 3] Reasoning RL (GRPO / DAPO / Dr.GRPO / GSPO / GRPO-lambda / RLVR)
        |
[Stage 4] Agentic RL: multi-turn tool use (Stage 9)
        |
Quantize + Export + Serve
```

### **11. Capability Regression Table**

Every alignment run reports this, whether or not it flatters the result.

| **Benchmark** | **SFT** | **SFT + DPO** | **SFT + GRPO** | **Acceptable delta** |
|---|---|---|---|---|
| MMLU | | | | < 1% (DPO), < 3% (GRPO) |
| IFEval | | | | no drop |
| GSM8K / HumanEval | | | | no drop |
| Domain held-out | | | | must rise |
| Format compliance | | | | at least 95% after GRPO |
| Mean completion length | | | | flat, unless you intend otherwise |

### **12. Labs**

### **12.1 Lab 1: DPO from Scratch**

- Implement the DPO loss in PyTorch and verify it against TRL's implementation on one batch.
- Train DPO on UltraFeedback with a 4B model.
- Sweep `beta` in {0.05, 0.1, 0.3}; report the held-out win rate and the reward margin for each.
- Compare a separate reference model against LoRA-as-reference.

### **12.2 Lab 2: Pairwise Data Audit**

- Take 200 preference pairs from your own data.
- Label each as obvious win or close call.
- Report the hard-negative rate. If it is below 20%, the pairs are too easy to teach anything, so regenerate them.

### **12.3 Lab 3: GRPO on GSM8K**

- Train a verifiable reward for GSM8K.
- Run GRPO with G=8 and track mean reward, group variance, length, and KL.
- Plot the four curves and identify where reward and variance diverge.
- Evaluate on the held-out split and on MMLU for regression.

### **12.4 Lab 4: Reward-Hacking Audit**

- Take a GRPO checkpoint and find 20 prompts where it produces the right answer for the wrong reason.
- Measure format compliance, not just accuracy.
- Fix by switching to a stricter verifier and re-running a short training.

### **12.5 Lab 5: Distillation**

- Distill a larger model into a 4B student on your domain data.
- Compare student accuracy against the teacher's accuracy on the same held-out set.
- Quantize the student and compare the accuracy delta against the teacher's quantized delta, since distilled models often have a different optimal quantization threshold.

### **13. Pitfalls**

- **DPO on easy pairs teaches nothing.** If chosen is obviously better, the gradient is tiny and the model learns nothing.
- **`beta` matters more than the learning rate.** It sets the strength of the KL constraint to the reference.
- **DPO overfits noisy pairs.** IPO's bounded margin exists for this.
- **GRPO without a KL term drifts.** Keep `beta` on and watch the KL curve.
- **Length inflation is the default GRPO failure.** If length rises monotonically, add Dr.GRPO or a length penalty.
- **Format compliance is the cheapest hacking signal.** Check it every eval.
- **PPO is not in TRL any more.** `from trl import PPOTrainer` fails; use verl or OpenRLHF.
- **KTO is stable now; IPO, ORPO, and CPO are not.** A minor TRL bump can delete the trainer you are calling.
- **Distillation changes calibration.** A distilled model that matches the teacher on accuracy can still be worse on confidence calibration, which changes downstream thresholding.

### **14. Alignment Pre-Flight Checklist**

- State the LoRA forward pass in one equation.
- What is the default LoRA config (r, $\alpha$, target_modules, lr)?
- How much memory does QLoRA save versus full FT on a 70B model?
- What does DPO optimize? Why is it "direct"?
- Compare DPO, IPO, KTO, SimPO, ORPO. When does each win?
- What is GRPO? Why does it beat PPO for reasoning?
- What is RLVR? Name two examples of verifiable rewards.
- State the consensus pipeline.
- When do you use full fine-tuning instead of LoRA?
- What is DoRA? When does it beat LoRA?
- Which TRL trainers are stable, and which were removed?

### **15. Acceptance Gate**

- A DPO run produced a held-out win rate on IFEval, AlpacaEval 2.0, or Arena-Hard-Auto.
- A GRPO run produced a held-out accuracy gain with a logged reward, variance, length, and KL curve.
- Capability regression table filled in for both runs.
- Format compliance measured (for GRPO, at or above 95%).
- The hard-negative rate of your preference data is above 20%.
- A written paragraph per run explaining what went wrong, not just what improved.

### **16. Resources**

- TRL release notes: https://github.com/huggingface/trl/releases/tag/v1.17.1
- verl release notes: https://github.com/verl-project/verl/releases/tag/v0.9.1
- OpenRLHF release notes: https://github.com/OpenRLHF/OpenRLHF/releases/tag/v0.11.2
- SkyRL on PyPI: https://pypi.org/project/skyrl/
- DPO, Rafailov et al.: https://arxiv.org/abs/2305.18290
- GRPO, Shao et al.: https://arxiv.org/abs/2402.03300
- PPO, Schulman et al.: https://arxiv.org/abs/1707.06347

**Next:** Stage 06: Frameworks at Scale.