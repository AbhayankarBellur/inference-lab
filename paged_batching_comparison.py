
import gc
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import (
    ContinuousBatchingConfig,
    GenerationConfig,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 48
DTYPE = torch.float16

PROMPTS = [
    "Explain GPU inference in simple terms.",
    "What is a transformer KV cache?",
    "Explain why memory bandwidth matters for LLM inference.",
    "Compare static batching with continuous batching.",
]


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)
    model.eval()
    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_generation_config(tokenizer):
    return GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def reset_peak_memory():
    synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_memory_mib():
    synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_standard(tokenizer):
    print("\n=== Standard generation: one request at a time ===")

    model = load_model()
    config = make_generation_config(tokenizer)

    encoded = [
        tokenizer(prompt, return_tensors="pt")
        for prompt in PROMPTS
    ]

    try:
        # Warm up the ordinary generation path outside measurement.
        with torch.inference_mode():
            warmup = tokenizer(
                "Warmup request.",
                return_tensors="pt",
            )
            model.generate(
                input_ids=warmup["input_ids"].to(DEVICE),
                attention_mask=warmup["attention_mask"].to(DEVICE),
                generation_config=config,
            )

        synchronize()
        reset_peak_memory()

        outputs = []
        token_count = 0
        start = time.perf_counter()

        with torch.inference_mode():
            for item in encoded:
                input_ids = item["input_ids"].to(DEVICE)
                attention_mask = item["attention_mask"].to(DEVICE)

                sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=config,
                )

                new_tokens = sequences[0, input_ids.shape[1]:]
                token_ids = new_tokens.tolist()
                token_count += len(token_ids)

                outputs.append(
                    {
                        "token_count": len(token_ids),
                        "text": tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        ),
                    }
                )

        synchronize()
        elapsed = time.perf_counter() - start

        result = {
            "elapsed": elapsed,
            "tokens": token_count,
            "throughput": token_count / elapsed if elapsed else 0,
            "peak_mib": peak_memory_mib(),
            "outputs": outputs,
        }

        print(f"Elapsed: {elapsed:.3f} s")
        print(f"Generated tokens: {token_count}")
        print(f"Throughput: {result['throughput']:.2f} tokens/sec")
        print(f"Peak allocated VRAM: {result['peak_mib']:.2f} MiB")

        return result

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_continuous(tokenizer):
    print("\n=== Continuous batching with paged KV cache ===")

    model = load_model()
    config = make_generation_config(tokenizer)

    batching_config = ContinuousBatchingConfig(
        max_memory_percent=0.20,
        page_size=16,
        scheduler_type="fifo",
        use_cuda_graph=False,
    )

    encoded = [
        tokenizer.encode(prompt)
        for prompt in PROMPTS
    ]

    try:
        # IMPORTANT:
        # Do not wrap generate_batch() in an outer torch.inference_mode().
        # The continuous-batching engine manages its own execution context.
        print("Warming up continuous-batching engine...")

        warmup_result = model.generate_batch(
            inputs=[tokenizer.encode("Warmup request.")],
            generation_config=config,
            continuous_batching_config=batching_config,
        )

        if not warmup_result:
            raise RuntimeError("Continuous-batching warmup returned no results.")

        del warmup_result
        synchronize()
        reset_peak_memory()

        print("Running measured requests...")
        start = time.perf_counter()

        results = model.generate_batch(
            inputs=encoded,
            generation_config=config,
            continuous_batching_config=batching_config,
        )

        synchronize()
        elapsed = time.perf_counter() - start

        # Explicit request IDs let us check that every prompt completed.
        expected_ids = [f"req_{i}" for i in range(len(PROMPTS))]
        missing = [request_id for request_id in expected_ids
                   if request_id not in results]

        if missing:
            raise RuntimeError(
                f"Missing continuous-batching results: {missing}. "
                f"Returned IDs: {list(results.keys())}"
            )

        outputs = []
        token_count = 0

        for request_id in expected_ids:
            output = results[request_id]
            token_ids = list(output.generated_tokens)
            count = len(token_ids)
            token_count += count

            outputs.append(
                {
                    "token_count": count,
                    "text": tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                    ),
                }
            )

        if token_count == 0:
            raise RuntimeError(
                "Continuous batching returned zero generated tokens."
            )

        result = {
            "elapsed": elapsed,
            "tokens": token_count,
            "throughput": token_count / elapsed if elapsed else 0,
            "peak_mib": peak_memory_mib(),
            "outputs": outputs,
        }

        print(f"Elapsed: {elapsed:.3f} s")
        print(f"Generated tokens: {token_count}")
        print(f"Throughput: {result['throughput']:.2f} tokens/sec")
        print(f"Peak allocated VRAM: {result['peak_mib']:.2f} MiB")

        return result

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def print_comparison(standard, continuous):
    print("\n" + "=" * 64)
    print("FINAL COMPARISON")
    print("=" * 64)

    print(
        f"{'Metric':<28}"
        f"{'Sequential':>16}"
        f"{'Continuous':>20}"
    )
    print("-" * 64)

    print(
        f"{'Elapsed (seconds)':<28}"
        f"{standard['elapsed']:>16.3f}"
        f"{continuous['elapsed']:>20.3f}"
    )
    print(
        f"{'Generated tokens':<28}"
        f"{standard['tokens']:>16}"
        f"{continuous['tokens']:>20}"
    )
    print(
        f"{'Tokens/sec':<28}"
        f"{standard['throughput']:>16.2f}"
        f"{continuous['throughput']:>20.2f}"
    )
    print(
        f"{'Peak allocated VRAM (MiB)':<28}"
        f"{standard['peak_mib']:>16.2f}"
        f"{continuous['peak_mib']:>20.2f}"
    )

    if standard["tokens"] == 0 or continuous["tokens"] == 0:
        print("\nINVALID COMPARISON: one path generated zero tokens.")
        return

    ratio = continuous["throughput"] / standard["throughput"]
    elapsed_change = (
        (standard["elapsed"] - continuous["elapsed"])
        / standard["elapsed"]
    ) * 100

    print("\nRelative results")
    print(f"Continuous throughput ratio: {ratio:.2f}x")
    print(f"Elapsed-time reduction: {elapsed_change:.1f}%")

    if standard["tokens"] != continuous["tokens"]:
        print(
            "\nNOTE: Token counts differ. Compare each request's output "
            "and stopping behavior before drawing performance conclusions."
        )

    print("\n=== Decoded outputs ===")
    for i, prompt in enumerate(PROMPTS):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(
            f"Sequential ({standard['outputs'][i]['token_count']} tokens):"
        )
        print(standard["outputs"][i]["text"])
        print(
            f"Continuous ({continuous['outputs'][i]['token_count']} tokens):"
        )
        print(continuous["outputs"][i]["text"])


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate the project venv and verify "
            "your PyTorch CUDA installation."
        )

    print(f"Transformers: {transformers.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Attention: SDPA")
    print(f"Max new tokens per request: {MAX_NEW_TOKENS}")

    tokenizer = load_tokenizer()

    standard = run_standard(tokenizer)
    continuous = run_continuous(tokenizer)

    print_comparison(standard, continuous)


if __name__ == "__main__":
    main()
