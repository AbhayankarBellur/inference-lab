
import gc
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

PROMPT = (
    "Explain GPU inference, attention, KV caching, memory bandwidth, "
    "quantization, batching, and how these techniques affect the latency "
    "and throughput of transformer language models. " * 35
)

MAX_INPUT_TOKENS = 384
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 1
MEASURED_RUNS = 3

DEVICE = "cuda"
DTYPE = torch.float16


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(tokenizer, attention_implementation):
    print(f"\nLoading model with attention={attention_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        attn_implementation=attention_implementation,
    )
    model = model.to(DEVICE)
    model.eval()

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_ids = inputs["input_ids"].to(DEVICE)

    return model, input_ids


def generate_once(model, input_ids, backend):
    """
    Greedy generation with an explicit KV cache.

    backend:
      "auto"   -> PyTorch chooses an eligible SDPA backend
      "flash"  -> force Flash SDPA for every SDPA operation
      "eager"  -> model uses Hugging Face eager attention
    """
    generated = []
    past_key_values = None

    def forward_step(token_ids, cache):
        return model(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
        )

    if backend == "flash":
        # If Flash SDPA is unsupported, PyTorch raises an error.
        # We deliberately do not silently fall back.
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            start = time.perf_counter()

            outputs = forward_step(input_ids, None)
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            past_key_values = outputs.past_key_values
            generated.append(next_token)

            synchronize()
            ttft_ms = (time.perf_counter() - start) * 1000

            decode_start = time.perf_counter()

            for _ in range(MAX_NEW_TOKENS - 1):
                outputs = forward_step(next_token, past_key_values)
                next_token = outputs.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                past_key_values = outputs.past_key_values
                generated.append(next_token)

            synchronize()
            decode_seconds = time.perf_counter() - decode_start

    else:
        start = time.perf_counter()

        outputs = forward_step(input_ids, None)
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        past_key_values = outputs.past_key_values
        generated.append(next_token)

        synchronize()
        ttft_ms = (time.perf_counter() - start) * 1000

        decode_start = time.perf_counter()

        for _ in range(MAX_NEW_TOKENS - 1):
            outputs = forward_step(next_token, past_key_values)
            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            past_key_values = outputs.past_key_values
            generated.append(next_token)

        synchronize()
        decode_seconds = time.perf_counter() - decode_start

    output_ids = torch.cat(generated, dim=1)

    if not torch.isfinite(outputs.logits).all().item():
        raise RuntimeError("Non-finite logits detected")

    decode_tps = (
        (MAX_NEW_TOKENS - 1) / decode_seconds
        if decode_seconds > 0
        else float("inf")
    )

    return {
        "ttft_ms": ttft_ms,
        "decode_tps": decode_tps,
        "output_ids": output_ids.detach().cpu(),
    }


def benchmark_configuration(name, implementation, backend, tokenizer):
    model = None
    input_ids = None

    try:
        model, input_ids = load_model(tokenizer, implementation)

        # Warmup: do not include this run in reported measurements.
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                generate_once(model, input_ids, backend)

        synchronize()
        torch.cuda.reset_peak_memory_stats()

        ttft_values = []
        throughput_values = []
        reference_output = None

        with torch.inference_mode():
            for run_index in range(MEASURED_RUNS):
                result = generate_once(model, input_ids, backend)

                ttft_values.append(result["ttft_ms"])
                throughput_values.append(result["decode_tps"])

                if reference_output is None:
                    reference_output = result["output_ids"]
                elif not torch.equal(
                    reference_output, result["output_ids"]
                ):
                    print(
                        "WARNING: generated tokens differed between runs."
                    )

                print(
                    f"  Run {run_index + 1}: "
                    f"TTFT={result['ttft_ms']:.2f} ms, "
                    f"decode={result['decode_tps']:.2f} tok/s"
                )

        peak_memory_mib = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

        decoded = tokenizer.decode(
            reference_output[0],
            skip_special_tokens=True,
        )

        return {
            "name": name,
            "ttft_ms": statistics.median(ttft_values),
            "decode_tps": statistics.median(throughput_values),
            "peak_mib": peak_memory_mib,
            "output": decoded,
            "status": "PASS",
        }

    except Exception as exc:
        print(f"\n{name} FAILED: {type(exc).__name__}: {exc}")
        return {
            "name": name,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }

    finally:
        del input_ids
        if model is not None:
            del model

        gc.collect()
        torch.cuda.empty_cache()
        synchronize()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your PyTorch setup.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Input token limit:", MAX_INPUT_TOKENS)
    print("Output tokens:", MAX_NEW_TOKENS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    configurations = [
        ("Eager", "eager", "eager"),
        ("SDPA automatic", "sdpa", "auto"),
        ("Flash SDPA forced", "sdpa", "flash"),
    ]

    results = []

    for name, implementation, backend in configurations:
        result = benchmark_configuration(
            name, implementation, backend, tokenizer
        )
        results.append(result)

    print("\n" + "=" * 76)
    print("FINAL RESULTS — MEDIAN OF MEASURED RUNS")
    print("=" * 76)

    for result in results:
        print(f"\n{result['name']}: {result['status']}")

        if result["status"] != "PASS":
            continue

        print(f"  TTFT:          {result['ttft_ms']:.2f} ms")
        print(f"  Decode:        {result['decode_tps']:.2f} tokens/sec")
        print(f"  Peak allocated:{result['peak_mib']:.2f} MiB")
        print(f"  Sample output: {result['output'][:180]!r}")

    eager = next(
        (r for r in results if r["name"] == "Eager"
         and r["status"] == "PASS"),
        None,
    )

    if eager:
        for result in results:
            if result["name"] == "Eager" or result["status"] != "PASS":
                continue

            speedup = result["decode_tps"] / eager["decode_tps"]
            ttft_change = (
                (result["ttft_ms"] / eager["ttft_ms"]) - 1
            ) * 100

            print(f"\n{result['name']} vs Eager:")
            print(f"  Decode throughput: {speedup:.2f}x")
            print(f"  TTFT change:       {ttft_change:+.1f}%")

    print("\nNote: peak memory is PyTorch allocated memory, not total GPU usage.")
    print("Note: timings are preliminary; repeat the experiment for confidence.")


if __name__ == "__main__":
    main()

