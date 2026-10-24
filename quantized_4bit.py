
import csv
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS = 64
PROMPT = "Explain what GPU inference is in simple terms."

assert torch.cuda.is_available(), "CUDA is unavailable"

print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
).eval()

messages = [{"role": "user", "content": PROMPT}]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
input_tokens = inputs["input_ids"].shape[1]

def generate(count):
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=count,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

# Warm-up
generate(8)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# Time to first token: separately measure prefill + first-token selection
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.perf_counter()

    out = model(**inputs, use_cache=True)
    first = out.logits[:, -1, :].argmax(
        dim=-1, keepdim=True
    )

    torch.cuda.synchronize()
    ttft = time.perf_counter() - start

# Measure the same fixed-size workload end to end.
torch.cuda.synchronize()
start = time.perf_counter()
output = generate(MAX_NEW_TOKENS)
torch.cuda.synchronize()
total_time = time.perf_counter() - start

output_tokens = output.shape[1] - input_tokens
decode_tps = output_tokens / total_time
peak_vram = torch.cuda.max_memory_allocated() / 1024**2

print("\n--- 4-BIT RESULTS ---")
print("Input tokens:", input_tokens)
print("Output tokens:", output_tokens)
print(f"First-token forward time: {ttft * 1000:.2f} ms")
print(f"End-to-end generation throughput: {decode_tps:.2f} tokens/s")
print(f"Generation latency: {total_time:.3f} s")
print(f"Peak allocated VRAM: {peak_vram:.2f} MiB")
print("\nGenerated text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))

path = Path("results.csv")
exists = path.exists()

with path.open("a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment", "model", "input_tokens",
            "output_tokens", "ttft_ms", "decode_tps",
            "total_latency_s", "peak_vram_mib",
        ],
    )
    if not exists:
        writer.writeheader()

    writer.writerow({
        "experiment": "4bit_nf4",
        "model": MODEL_ID,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": round(ttft * 1000, 2),
        "decode_tps": round(decode_tps, 2),
        "total_latency_s": round(total_time, 3),
        "peak_vram_mib": round(peak_vram, 2),
    })

print("\nResults appended to results.csv")
