# **Stage 04 — Alignment and Post-Training**
**Outcome:** take an SFT checkpoint and either align it to preferences or train it against a verifiable reward — with evidence that the alignment helped and did not break the base model.**Libraries:** `trl` 1.17.1 (DPO, KTO, GRPO, RLOO, Reward, Distillation) · `peft` 0.21.2 · `verl` 0.9.1 · `OpenRLHF` 0.11.2 · `SkyRL` 0.4.0


---

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

## **Topic checklist — preference optimization**

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

## **Topic checklist — RL fine-tuning**

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

## **Code**

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
# **Under PEFT, the reference model is the frozen base with the adapter disabled.**
# **Under a full fine-tune, pass an explicit reference model.**
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
# **Track these every N steps and plot them together:**
#   **- mean reward per group**
#   **- within-group reward variance (falls to zero when the model collapses)**
#   **- mean completion length (rises when length inflation kicks in)**
#   **- KL divergence from the reference (rises when the policy drifts)**
# **A reward curve that rises while variance collapses and length explodes is**
# **the classic signature of reward hacking, not of learning.**
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

## **Labs**

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

**Next:** Stage 05 — Frameworks at Scale.
