"""Estimate weight memory from parameter count and bytes per parameter."""
"""
Estimate how much GPU memory a model needs.

Run it with:  python estimate_vram.py

Parameters are in billions, so billions of parameters times bytes per
parameter gives gigabytes (GB) directly. GB here means 1 billion bytes.
"""

# Bytes needed to store one parameter at each precision
BYTES_PER_PARAM = {
    "bf16": 2.0,
    "int8": 1.0,
    "4bit": 0.5,
}

# Set this to your GPU memory in GB, for example 24. Leave None to skip the fit check.
MY_GPU_GB = None

# Memory kept free for the CUDA context and the display driver
RESERVE_GB = 1.0


def weights_gb(total_params_b, bytes_per_param):
    """Memory for the weights alone. Use TOTAL parameters, including every expert."""
    return total_params_b * bytes_per_param


def adapter_gb(adapter_params_m):
    """
    Memory for LoRA adapters while training.
    Per trainable parameter: 4 bytes weight, 4 bytes gradient, 8 bytes for Adam
    states. That is 16 bytes. Millions of parameters times 16 bytes gives MB.
    """
    return adapter_params_m * 16 / 1000


def lora_training_gb(total_params_b, bytes_per_param, adapter_params_m=200, extra_gb=4.0):
    """
    Rough training estimate: weights + adapters + extra.
    The extra part covers activations and buffers. It changes with batch size and
    sequence length, so treat the result as a starting guess and compare it with
    a published number.
    """
    return weights_gb(total_params_b, bytes_per_param) + adapter_gb(adapter_params_m) + extra_gb


def fits(vram_gb, need_gb):
    """True when the estimate fits after keeping the reserve free."""
    return need_gb <= vram_gb - RESERVE_GB


# name, total parameters (billions), published bf16 GB, published 4bit GB
EXAMPLES = [
    ("Inkling 2 Mini", 26.4, 53.0, 20.0),
    ("Muse Glimmer", 29.6, 56.0, 21.0),
    ("Gemma 4 31B", 30.7, 69.9, 17.5),
]


def main():
    print("Weights memory check")
    print("─" * 60)
    print(f"{'Model':<16}{'bf16 GB':>10}{'4bit GB':>10}{'published 4bit':>16}")
    print("─" * 60)
    for name, params_b, pub_bf16, pub_4bit in EXAMPLES:
        bf16 = weights_gb(params_b, BYTES_PER_PARAM["bf16"])
        q4 = weights_gb(params_b, BYTES_PER_PARAM["4bit"])
        print(f"{name:<16}{bf16:>10.1f}{q4:>10.1f}{pub_4bit:>16.1f}")

    print()
    print("Real bytes per parameter at 4 bit — published size divided by total parameters")
    print("─" * 60)
    for name, params_b, pub_bf16, pub_4bit in EXAMPLES:
        real = pub_4bit / params_b
        print(f"{name:<16}{real:>6.2f} bytes (formula says 0.50)")

    print()
    print("QLoRA training guess for a 26.4B model at 4 bit")
    print("─" * 60)
    base = weights_gb(26.4, BYTES_PER_PARAM["4bit"])
    adapters = adapter_gb(200)
    total = lora_training_gb(26.4, BYTES_PER_PARAM["4bit"])
    print(f"Weights    {base:>6.1f} GB")
    print(f"Adapters   {adapters:>6.1f} GB")
    print(f"Extra      {4.0:>6.1f} GB")
    print(f"Total      {total:>6.1f} GB — Unsloth lists 22 GB, so check this guess")

    print()
    print("Does it fit?")
    print("─" * 60)
    if MY_GPU_GB is None:
        print("Set MY_GPU_GB at the top of the file to run the fit check.")
    else:
        for name, params_b, pub_bf16, pub_4bit in EXAMPLES:
            need = lora_training_gb(params_b, BYTES_PER_PARAM["4bit"])
            answer = "yes" if fits(MY_GPU_GB, need) else "no"
            print(f"{name:<16}needs about {need:>5.1f} GB on a {MY_GPU_GB} GB card — fits: {answer}")


if __name__ == "__main__":
    main()