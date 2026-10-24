
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 64
PROMPT = "Explain what GPU inference is in simple terms."

assert torch.cuda.is_available(), "CUDA is unavailable"

print("GPU:", torch.cuda.get_device_name(0))

# Load the pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to(DEVICE).eval()

messages = [{"role": "user", "content": PROMPT}]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
input_tokens = inputs["input_ids"].shape[1]

# Warm up GPU before measuring
with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# Measure prefill + first generated token
with torch.inference_mode():
    start = time.perf_counter()

    outputs = model(
        **inputs,
        use_cache=True,
    )

    next_token = outputs.logits[:, -1, :].argmax(
        dim=-1, keepdim=True
    )

    past = outputs.past_key_values
    torch.cuda.synchronize()
    ttft = time.perf_counter() - start

    generated = [next_token]

    # Decode remaining tokens
    start = time.perf_counter()

    for _ in range(MAX_NEW_TOKENS - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
        )

        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        generated.append(next_token)

    torch.cuda.synchronize()
    decode_time = time.perf_counter() - start

generated_ids = torch.cat(generated, dim=1)
text = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True,
)

decode_tps = (MAX_NEW_TOKENS - 1) / decode_time
total_time = ttft + decode_time
peak_vram = torch.cuda.max_memory_allocated() / 1024**2

print("\n--- BASELINE RESULTS ---")
print(f"Input tokens: {input_tokens}")
print(f"Generated tokens: {MAX_NEW_TOKENS}")
print(f"TTFT: {ttft * 1000:.2f} ms")
print(f"Decode throughput: {decode_tps:.2f} tokens/s")
print(f"Total latency: {total_time:.3f} s")
print(f"Peak allocated VRAM: {peak_vram:.2f} MiB")
print("\nGenerated text:\n", text)

# Save results for future comparisons
path = Path("results.csv")
exists = path.exists()

with path.open("a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment",
            "model",
            "input_tokens",
            "output_tokens",
            "ttft_ms",
            "decode_tps",
            "total_latency_s",
            "peak_vram_mib",
        ],
    )

    if not exists:
        writer.writeheader()

    writer.writerow({
        "experiment": "fp16_baseline",
        "model": MODEL_ID,
        "input_tokens": input_tokens,
        "output_tokens": MAX_NEW_TOKENS,
        "ttft_ms": round(ttft * 1000, 2),
        "decode_tps": round(decode_tps, 2),
        "total_latency_s": round(total_time, 3),
        "peak_vram_mib": round(peak_vram, 2),
    })

print("\nResults saved to results.csv")
