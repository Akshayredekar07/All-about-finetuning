# Stage 8 — Distributed Training at Scale

**Outcome:** you can take a model that does not fit on one GPU and put it on a node or a rack — choosing between FSDP/ZeRO-3, tensor parallelism, pipeline parallelism and expert parallelism on evidence rather than habit, and you can read a training log well enough to tell whether the bottleneck is compute, communication or data.

> [!warning] This is a depth track, not a gate
> Stages 0–7 and 9–11 are all reachable on a single 24 GB card. Stage 8 matters when the base model is bigger than your GPU, when you are training an MoE, or when you are running RL rollouts that need throughput. If your base fits in 24 GB and your dataset is under 100K samples, do Stage 8 last and do it deliberately.

## The parallelism taxonomy

Five axes, each solving a different problem. Choosing well is mostly about knowing which resource is actually scarce.

| Axis | What it splits | Communication pattern | Use when | Cost |
|---|---|---|---|---|
| Data parallelism (DP / DDP) | The batch | Gradient all-reduce | The model fits and you have spare compute | Gradient sync cost grows with model size |
| Sharded data parallelism (FSDP / ZeRO-3) | Parameters, gradients, optimizer states | All-gather + reduce-scatter | The model does not fit on one GPU | Communication volume, but no extra hardware |
| Tensor parallelism (TP) | Individual layers, across GPUs | All-reduce per layer, twice | Ultra-high-speed interconnect inside a node | Needs NVLink; degrades badly over Ethernet |
| Pipeline parallelism (PP) | Layers, across stages | Point-to-point send/recv | More GPUs than a node has NVLink | Pipeline bubble, `bubble fraction = (p-1)/m` |
| Expert parallelism (EP) | MoE experts | All-to-all dispatch/combine | MoE models | All-to-all is the hardest pattern to optimise |

Context parallelism, sequence parallelism and disaggregated prefill extend the same idea across sequence dimensions; they appear in Stage 10 on the serving side.

## Sharded data parallelism in practice

FSDP is the modern default, and ZeRO-3 is the same idea with a different vocabulary. The equivalences are worth stating because they come up in every interview:

- **ZeRO-1** shards optimizer states.
- **ZeRO-2** adds gradient sharding.
- **ZeRO-3** adds parameter sharding — functionally equivalent to FSDP `FULL_SHARD`.
- **ZeRO-Offload** moves optimizer states to CPU; **ZeRO-Infinity** extends this to NVMe.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

mp_policy = MixedPrecision(
    param_dtype  = torch.bfloat16,
    reduce_dtype = torch.float32,
    buffer_dtype = torch.bfloat16,
)
model = FSDP(
    model,
    auto_wrap_policy   = transformer_auto_wrap_policy,
    mixed_precision    = mp_policy,
    sharding_strategy  = ShardingStrategy.FULL_SHARD,
    device_id          = torch.cuda.current_device(),
)
```

### DeepSpeed ZeRO-3 with offload, as an Axolotl config

```yaml
deepspeed:
  zero_stage: 3
  offload_optimizer_device: cpu
  offload_param_device: cpu
  overlap_comm: true
  contiguous_gradients: true
  reduce_bucket_size: 5e8
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6
  bf16:
    enabled: true
  gradient_clipping: 1.0
```

Offload trades PCIe bandwidth for HBM capacity. It is a last resort for a 7B–13B model on a consumer node, and roughly mandatory once the optimizer states no longer fit in aggregate GPU memory.

### Production multi-GPU config (Llama 4 Scout shape)

```yaml
base_model: meta-llama/Llama-4-Scout-17B-16E-Instruct
model_type: LlamaForCausalLM

datasets:
  - path: your_dataset
    type: sharegpt
    conversation: chatml

adapter: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

sequence_len: 8192        # use 8K–32K during SFT; not full 10M ctx
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 2e-4
warmup_ratio: 0.03

bf16: auto
tf32: true
flash_attention: true
gradient_checkpointing: true

fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

output_dir: ./outputs/llama4-scout-run-001
```

Two details in that config are load-bearing. `sequence_len: 8192` is a training choice, not a model limit — training at the model's advertised maximum context is a way to spend a GPU-month discovering OOM. And `fsdp_config.fsdp_transformer_layer_cls_to_wrap` must name the actual decoder class of the model you loaded, or FSDP will shard at the wrong granularity and lose most of its benefit.

## Expert parallelism and the MoE training problem

Expert parallelism is the hardest communication pattern in modern training: every token has to be dispatched to the GPUs holding its chosen experts, and then gathered back. DeepEP and Hybrid-EP are the two current implementations.

**DeepEP** is DeepSeek's own library: high-throughput and low-latency all-to-all kernels for MoE dispatch and combine, with FP8 support, low-precision and NVLink weight exchange for redundant experts. Version 2 refactored expert parallelism, moved to the NCCL Gin backend, and reports up to 1.3× peak performance with roughly 4× fewer streaming-multiprocessor resources than version 1, with scale-up to EP2048. It requires Hopper-class GPUs (SM90 or later with the SM90 PTX ISA), CUDA 12.3+, PyTorch 2.10+, NCCL 2.30.4+, NVLink inside a node and RDMA between nodes [117].

**Hybrid-EP** is NVIDIA's variant of the same idea, built on TMA instructions for minimal SM usage and a larger NVLink domain, with fine-grained communication–computation overlap for single-batch scenarios and PCIe fallback for non-NVLink environments [117].

### What Hybrid-EP is actually worth

The source document claimed "514% throughput improvement over baseline EP". NVIDIA's own technical blog from 2 February 2026 reports something much more modest and much more useful:

| Workload | Baseline | With Hybrid-EP | Delta |
|---|---|---|---|
| DeepSeek-V3 | 829 TFLOPs/GPU | 943 TFLOPs/GPU | **+14%** |
| Megatron-FSDP | — | — | **+8%** |
| Qwen3-235B, BF16 | — | — | **+5.5%** |
| Qwen3-235B, MXFP8 | — | — | **+9.9%** |

A 14% end-to-end win on a flagship model is a real result; a 514% claim is a category error. Quote the measured deltas, not the headline [116].

In Megatron Core, both backends are selected the same way:

```bash
# Cross-node dispatch (DeepEP)
--moe-token-dispatcher-type=flex --moe-flex-dispatcher-backend=deepep

# Intra-node / multi-node NVLink dispatch (HybridEP)
--moe-token-dispatcher-type=flex --moe-flex-dispatcher-backend=hybridep
```

The operational rule that follows from the topology: keep expert-parallel and tensor-parallel traffic inside the NVLink domain whenever possible, because EP×TP all-to-all is the most expensive collective in the stack [118].

### Megatron-Core path, for scale

Megatron-LM's October 2026 state includes MFSDP v2, MoE execution paths (DeepEP V2, MoonEP, Mixture-of-Kittens), dynamic context parallelism, hybrid attention, disaggregated inference, and a fix for HybridEP NCCL routing-map all-gather. Megatron Core 0.19 (August 2026) adds `HybridModel` with router balancing, packed-sequence dispatch and CUDA-graph-compatible activation offloading. NeMo Megatron Bridge 0.6.0 (19 August 2026) adds GLM-5.2 with fused cuDNN dynamic-sparse attention and 128K THD-packed context parallelism; Bridge 0.5.0 (22 June 2026) added Qwen3.5, DeepSeek V4, GLM-5, Gemma 4 and Falcon H1 support.

Published Megatron throughput numbers, for calibration: DeepSeek-V3 671B on 256 GB300 at 1,648 TFLOPs/s per GPU (6,338 tokens/s per GPU); Qwen3-235B-A22B at 8,832 tokens/s per GPU; Nemotron 3 Ultra in NVFP4 at 3,744 tokens/s per GPU. Treat them as an upper bound for a heavily optimised configuration, not as a promise for your cluster.

## Diagnosing a slow distributed run

The three bottlenecks produce three different log signatures, and they are worth memorising because the first instinct is almost always wrong.

| Symptom | Likely cause | Check | Fix |
|---|---|---|---|
| GPU utilisation near 100% but throughput flat | Compute-bound, or a bad kernel | `nvidia-smi dmon`, per-GPU power draw | Check sequence length, micro-batch size, whether FlashAttention is actually enabled |
| GPUs idle in bursts, collective time high | Communication-bound | `NCCL_DEBUG=INFO`, `nsys` trace, NVLink health | Move EP/TP traffic inside the NVLink domain; overlap comm and compute; enable `overlap_comm` |
| One GPU slower than the rest | Straggler or bad data | Per-rank step time distribution, dataset shard sizes | Fix the sampler; check for a shard with pathological sequence lengths |
| Steady loss but OOM on restart | Prefetch too aggressive | `stage3_prefetch_bucket_size`, `stage3_param_persistence_threshold` | Lower the prefetch bucket; raise the persistence threshold |
| All-reduce slow on Ethernet | Interconnect, not software | `nvidia-smi topo -p2p r`, NCCL topology output | Expect 3–10× penalty versus NVLink for TP; restructure to keep TP intra-node |

A fast sanity check before blaming the framework:

```bash
nvidia-smi topo -p2p r          # peer access between every pair
python -c "import torch; print(torch.cuda.get_device_capability())"
df -h /dev/shm                 # too-small shm breaks NCCL and dataloader workers
```

## Multi-node launch

```bash
# Single node
accelerate launch --num_processes 8 train_sft.py \
  --config_file configs/zero3.yaml

# Multi-node: node 0
accelerate launch --num_processes 8 --num_machines 2 --machine_rank 0 \
  --main_process_ip 10.0.0.1 --main_process_port 29500 \
  --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:29500 train_sft.py

# node 1
accelerate launch --num_processes 8 --num_machines 2 --machine_rank 1 \
  --main_process_ip 10.0.0.1 --main_process_port 29500 \
  --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:29500 train_sft.py
```

The rendezvous endpoint must be reachable from every node on the same port, and the shared filesystem must be mounted identically on all nodes if the run writes checkpoints. Both failures present as a hang with no error message, which is why they belong in a pre-flight checklist rather than in a debugging session.

## Distributed training pre-flight checklist

- [ ] The model actually does not fit on one GPU — the reason for going distributed is written down.
- [ ] The interconnect topology is known: which pairs have NVLink or PCIe P2P, and which cross nodes only over RDMA.
- [ ] TP groups are placed inside a single node. EP×TP is kept inside the NVLink domain.
- [ ] The sharding strategy matches the goal: `FULL_SHARD` for memory, `SHARD_GRAD_OP` when communication is the constraint. ZeRO-3 is FSDP Full Shard under a different name.
- [ ] Gradient checkpointing is enabled, and you have measured what it costs in step time.
- [ ] Effective batch size is computed across all ranks and matches the intent, not just the per-device value.
- [ ] `fsdp_transformer_layer_cls_to_wrap` names the real decoder class of the loaded model.
- [ ] MoE runs have an expert-utilisation histogram instrumented, with an alert below ~1% utilisation on any expert.
- [ ] Checkpointing writes to shared storage and has been tested by resuming from it, not just by saving to it.
- [ ] A single-rank dry run has completed on the same data before the multi-node launch.
