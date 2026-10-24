
import time
import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import (
    ContinuousBatchingConfig,
    GenerationConfig,
)


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda"

PROMPTS = [
    "Explain GPU inference in simple terms.",
    "What is a transformer KV cache?",
    "Explain why memory bandwidth matters for LLM inference.",
    "Compare static batching with continuous batching.",
]

MAX_NEW_TOKENS = 48


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Transformers:", transformers.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model with paged SDPA attention...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()

    # Each prompt becomes a list of token IDs.
    inputs = [tokenizer.encode(prompt) for prompt in PROMPTS]

    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )

    # Keep the KV cache budget conservative on the 6 GB GPU.
    batching_config = ContinuousBatchingConfig(
    max_memory_percent=0.20,
    page_size=16,
    scheduler_type="fifo",
    use_cuda_graph=False,
    )

    print("\nConfiguration:")
    print("  Attention: sdpa")
    print("  Requests:", len(inputs))
    print("  Max new tokens per request:", MAX_NEW_TOKENS)
    print("  KV-cache memory budget fraction: 0.20")
    print("  KV-cache page size:", batching_config.page_size)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    outputs = model.generate_batch(
        inputs=inputs,
        generation_config=generation_config,
        continuous_batching_config=batching_config,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_generated = 0

    print("\n=== Real model results ===")

    for request_id, output in outputs.items():
        generated_ids = output.generated_tokens
        total_generated += len(generated_ids)

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        print(f"\nRequest {request_id}")
        print("Generated tokens:", len(generated_ids))
        print("Output:", generated_text[:300])

    throughput = total_generated / elapsed if elapsed > 0 else 0
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\n=== Benchmark summary ===")
    print("Elapsed time:", f"{elapsed:.3f} seconds")
    print("Total generated tokens:", total_generated)
    print("Aggregate throughput:", f"{throughput:.2f} tokens/sec")
    print("Peak allocated VRAM:", f"{peak_mib:.2f} MiB")
    print("Status: completed")


if __name__ == "__main__":
    main()
