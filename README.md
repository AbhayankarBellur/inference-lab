# Tiny Inference Lab --- LLM Inference Optimization Experiments

A hands-on, measurement-driven lab for understanding local LLM
inference: precision, attention implementations, KV caching, speculative
decoding, batching, and eventually production inference servers such as
vLLM and SGLang.

> **Principle:** use real model execution and measured results.
> Simulators may be useful for explaining concepts, but they are not
> substitutes for the experiments in this lab.

## Table of contents

-   [Goals](#goals)
-   [Hardware and software](#hardware-and-software)
-   [Project setup](#project-setup)
-   [Model download and baseline](#model-download-and-baseline)
-   [Experiments and results](#experiments-and-results)
-   [What the results mean](#what-the-results-mean)
-   [Known caveats](#known-caveats)
-   [Scripts created during the lab](#scripts-created-during-the-lab)
-   [Current WSL status and next
    steps](#current-wsl-status-and-next-steps)
-   [Reproducibility checklist](#reproducibility-checklist)

------------------------------------------------------------------------

## Goals

The lab investigates the practical trade-offs behind:

1.  Model precision and quantization.
2.  Attention implementation choices.
3.  KV-cache use and long-context scaling.
4.  Speculative decoding and draft-model acceptance.
5.  Sequential inference versus continuous batching.
6.  Concurrency and aggregate throughput.
7.  Eventually, real vLLM and SGLang inference servers on GPU-enabled
    Linux/WSL 2.

## Hardware and software

### Hardware

  Item                                 Observed configuration
  ------------------------------------ -----------------------------------------------
  GPU                                  NVIDIA GeForce RTX 3050 Laptop GPU
  GPU VRAM                             6 GB (6144 MiB reported by `nvidia-smi`)
  CPU                                  Intel Core i5-13450HX
  System RAM                           16 GB DDR5 (15.69 GB visible to Windows)
  Operating system                     Windows 11 Home Single Language, version 25H2
  Windows build                        26200.9711
  NVIDIA driver                        581.95
  Driver-reported CUDA compatibility   CUDA 13.0
  Free C: drive space at WSL check     775.61 GB

RAM availability is dynamic. One Windows snapshot showed about 5.02 GB
free; that is not the machine's total RAM.

### Python environment

The experiments were run in a Windows PowerShell virtual environment at:

``` text
C:\Users\abhay\tiny-inference-lab\.venv
```

Observed versions:

  Component            Version
  -------------------- --------------
  Python               3.11.9
  PyTorch              2.14.1+cu126
  Transformers         5.20.2
  Triton for Windows   3.8.0.post29

CUDA-enabled PyTorch successfully reported
`torch.cuda.is_available() == True`. The installed PyTorch build is
`cu126`; the driver's reported CUDA 13.0 compatibility does not mean the
PyTorch package itself was built for CUDA 13.0.

## Project setup

Open PowerShell and create the working directory and virtual
environment:

``` powershell
mkdir "$HOME\tiny-inference-lab"
cd "$HOME\tiny-inference-lab"

py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python --version
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use a process-scoped policy for the
current terminal only:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the CUDA-enabled PyTorch build and core tooling. The commands
below document the environment used in this lab; package availability
and compatibility can change, so preserve a working environment before
upgrading packages.

``` powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers accelerate bitsandbytes
python -m pip install triton-windows==3.8.0.post29
```

The observed versions after setup were PyTorch `2.14.1+cu126`,
Transformers `5.20.2`, and `triton-windows 3.8.0.post29`. If reproducing
later, check compatible package releases rather than assuming these
exact versions will remain available.

Verify GPU access:

``` powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
nvidia-smi
```

### Downloading and loading the model

The main target model was:

-   `HuggingFaceTB/SmolLM2-360M-Instruct`

The speculative draft model was:

-   `HuggingFaceTB/SmolLM2-135M-Instruct`

Transformers downloads model files from Hugging Face the first time
`from_pretrained()` is called, then uses the local Hugging Face cache on
later runs. For example:

``` python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
).to("cuda").eval()

print("GPU:", torch.cuda.get_device_name(0))
```

A typical inference call uses tokenized inputs moved to CUDA and
`model.generate(...)`. For benchmark correctness, keep the prompt,
generation settings, output length, warm-up, repetitions, and model
placement consistent across variants.

------------------------------------------------------------------------

## Experiments and results

All measurements below are the actual outputs reported during this lab.
Results are hardware-, software-, model-, prompt-, and
workload-specific; they are not universal performance guarantees.

### 1. FP16 baseline

A baseline run used FP16 weights, a 39-token prompt, and 64 generated
tokens.

  Metric                                  Result
  ----------------------------- ----------------
  Time to first token (TTFT)           150.06 ms
  Decode throughput               13.91 tokens/s
  Total latency                          4.678 s
  Peak PyTorch allocated VRAM            718 MiB

This is an early baseline, not a perfectly controlled reference for
every later experiment. Some subsequent tests used different prompts and
output lengths.

### 2. FP16 versus 4-bit NF4 quantization

A normalized comparison was run between FP16 and 4-bit NF4 quantization.

  Metric                              FP16     4-bit NF4
  --------------------------- ------------ -------------
  Median TTFT                     66.51 ms      72.23 ms
  Decode throughput             14.8 tok/s   23.71 tok/s
  Approximate total latency        4.324 s        2.73 s
  Peak allocated VRAM           718.04 MiB       261 MiB

Observed differences:

-   NF4 decode throughput was approximately **60.2% higher**.
-   Peak allocated VRAM was approximately **63.65% lower**.
-   Median TTFT was slower for NF4 in this run, showing that
    quantization does not necessarily improve every metric.

The comparison is tied to the tested setup and measurement method.
Quantization overhead, kernels, and GPU architecture affect results.

### 3. Attention implementation: eager versus SDPA

This experiment used a 384-token prompt and 32 generated tokens, with
three repetitions.

  Metric                        Eager     SDPA auto
  --------------------- ------------- -------------
  Median TTFT                89.48 ms      35.63 ms
  Decode throughput       11.04 tok/s   38.78 tok/s
  Peak allocated VRAM      764.73 MiB    763.99 MiB

SDPA auto was faster in this sample, but the measurements showed notable
variability and only three repetitions were used. Treat the size of the
improvement as provisional pending a more controlled rerun.

**Important:** SDPA auto does not mean FlashAttention was used. Forcing
Flash SDPA failed with:

``` text
RuntimeError: No available kernel. Aborting execution.
Torch was not compiled with flash attention.
```

Therefore, the experiment supports a comparison of eager attention
against the available SDPA path, not a claim that FlashAttention ran
successfully.

### 4. Triton fused scale-plus-bias microbenchmark

A custom Triton kernel combining scale and bias was compared with a
PyTorch baseline consisting of two operations.

-   Correctness checks passed.
-   The best reported speedup was **1.78× at 4 million elements**.

Caveat: the baseline used two separate operations and an intermediate
allocation. This makes it an imperfect comparison. The result is useful
as a kernel-fusion learning exercise, not as a general claim that Triton
is 1.78× faster than PyTorch. A fair comparison should ensure equivalent
operations, warm-up, synchronization, and repeated timing.

### 5. Paged KV-cache allocator and continuous-batching concept demos

Educational simulations were created to understand paged allocation and
scheduling. They passed their checks, but they are **not** evidence of
vLLM's actual PagedAttention implementation or production scheduling
behavior.

The lab later moved to a real Transformers continuous-batching API for
the primary batching measurements.

### 6. Real Transformers continuous batching versus sequential generation

The `model.generate_batch()` API from Transformers 5.20.2 was used with
`ContinuousBatchingConfig`, `page_size=16`, `scheduler_type="fifo"`,
`use_cuda_graph=False`, `max_memory_percent=0.20`, and SDPA attention.

An initial attempt wrapping `generate_batch()` in an outer
`torch.inference_mode()` raised:

``` text
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

Removing the outer inference-mode wrapper fixed the issue.

Measured comparison:

  Metric                   Sequential, 4 requests   Continuous batching, 4 requests
  ---------------------- ------------------------ ---------------------------------
  Elapsed time                            2.939 s                           1.889 s
  Total output tokens                         119                               119
  Aggregate throughput                40.49 tok/s                       62.98 tok/s
  Peak allocated VRAM                  714.92 MiB                       1476.11 MiB

Observed:

-   Continuous batching achieved about **1.56× aggregate throughput**.
-   Elapsed time fell by about **35.7%**.
-   Peak allocated VRAM was about **2.06×** the sequential result.

This is a real Transformers batching experiment, not a simulator. It is
not the same as vLLM, and the API/configuration is
implementation-specific.

### 7. Concurrency scaling with real Transformers continuous batching

`concurrency_scaling.py` tested batches of 1, 2, 4, and 8 requests, with
a maximum of 48 new tokens per request and three repetitions. The prompt
list was reused cyclically.

  --------------------------------------------------------------------------
        Requests Median elapsed   Total output      Aggregate Peak allocated
                           time         tokens     throughput           VRAM
  -------------- -------------- -------------- -------------- --------------
               1        1.019 s             22    21.59 tok/s     1473.2 MiB

               2        0.987 s             23    23.30 tok/s     1492.0 MiB

               4        1.820 s            119    65.40 tok/s     1507.9 MiB

               8        1.828 s            238   130.20 tok/s     1527.2 MiB
  --------------------------------------------------------------------------

Aggregate throughput rose about **6.03× from 1 to 8 requests** in these
measurements, while peak allocated VRAM increased by roughly 54 MiB.

**Fairness caveat:** token counts differ across concurrency levels
because short prompts sometimes reached EOS before the maximum output
length. This is therefore not a controlled speedup ratio. It is evidence
that aggregate throughput can rise under a larger batch, not proof that
each request gets faster. This experiment also represents completed
batches, not staggered live request arrivals in a production serving
system.

### 8. KV-cache enabled versus disabled

The `kv_cache_comparison.py` test used a 176-token prompt and 48
generated tokens.

  Metric                  KV cache ON   KV cache OFF
  --------------------- ------------- --------------
  Median elapsed time         1.180 s        1.098 s
  Decode throughput       40.69 tok/s    43.71 tok/s
  Peak allocated VRAM         723 MiB        718 MiB

Outputs matched. For this short context, caching did not improve
performance and added a small amount of memory overhead. This is not
surprising: caching is most useful when it avoids recomputing a growing
prefix across decode steps, but cache management and implementation
overhead can matter at small scales. The longer-context experiment below
makes the difference clearer.

### 9. Long-context KV-cache scaling

The `long_context_kv_scaling.py` experiment tested actual input lengths
of 176, 512, 1024, 1536, and 2048 tokens, with 48 generated tokens and
three repetitions. Exact-length prompts were formed by repeating token
IDs; they were not natural prose. Outputs matched between cache-on and
cache-off runs at each context length.

  ----------------------------------------------------------------------------
       Input  Cache ON:  Cache ON:  Cache ON: Cache OFF: Cache OFF: Cache OFF:
      tokens   median s      tok/s   peak MiB   median s      tok/s   peak MiB
  ---------- ---------- ---------- ---------- ---------- ---------- ----------
         176      1.184      40.53      723.0      1.180      40.68      718.0

         512      1.171      40.99      743.6      2.324      20.65      725.9

        1024      1.242      38.66      776.0      4.223      11.37      737.7

        1536      1.235      38.88      806.4      6.081       7.89      748.5

        2048      1.246      38.51      838.2      8.169       5.88      759.6
  ----------------------------------------------------------------------------

At 2048 input tokens:

-   Cache ON was approximately **6.56× faster by elapsed time** than
    cache OFF.
-   Cache-on peak allocated memory grew by **115.2 MiB** from 176 to
    2048 input tokens.
-   At 2048 tokens, cache ON used about **78.6 MiB more allocated
    memory** than cache OFF.
-   With cache ON, elapsed time remained near 1.2 seconds for these
    cases, whereas cache OFF became progressively slower as the prefix
    grew.

These values are PyTorch's allocated-memory measurements, not total GPU
memory usage. The earlier script printed a misleading `Pending`
output-match status in one summary because of a reporting-order bug; the
actual outputs matched at all tested lengths.

### 10. Speculative decoding: target-only versus assistant model

Models:

-   Target: `HuggingFaceTB/SmolLM2-360M-Instruct`
-   Draft: `HuggingFaceTB/SmolLM2-135M-Instruct`

The earlier experiment used a 177-token prompt, 96 output tokens, and
three repetitions.

  Metric                  Target only   Speculative, draft length 5
  --------------------- ------------- -----------------------------
  Median elapsed time         2.355 s                       2.484 s
  Throughput              40.76 tok/s                   38.64 tok/s
  Peak allocated VRAM       724.3 MiB                    1007.5 MiB

The generated token IDs matched between target-only and assisted
decoding.

Observed:

-   Speculative decoding was about **5.5% slower by elapsed time** in
    this test.
-   Throughput was about **5.2% lower**.
-   Peak allocated VRAM increased by about **283.2 MiB**.

This is a meaningful negative result: the smaller target model is
already relatively cheap to run, so the additional draft-model work can
outweigh the saved target computation.

### 11. Speculative draft-length sweep

The next sweep compared draft lengths 3, 5, and 8, using a 137-token
prompt, 96 generated tokens, and three measured trials per draft length.
The reported equality check compared assisted outputs with the first
assisted run; it was not a target-only equivalence check.

  -----------------------------------------------------------------------
    Draft length Median elapsed     Throughput Peak allocated Output
                           time                          VRAM match
  -------------- -------------- -------------- -------------- -----------
               3        2.499 s    38.41 tok/s     1000.7 MiB True

               5        2.458 s    39.05 tok/s     1000.7 MiB True

               8        2.535 s    37.87 tok/s     1000.7 MiB True
  -----------------------------------------------------------------------

Draft length 5 was fastest, but only by about 1.7% versus length 3---too
small to establish a reliable win from three trials. Length 8 was
slower.

### 12. Direct draft-acceptance instrumentation

The installed Transformers 5.20.2 source was inspected. In
`GenerationMixin._assisted_decoding()`, the implementation calculates
`n_matches` and passes it to:

``` python
candidate_generator.update_candidate_strategy(
    input_ids, new_logits, n_matches
)
```

The candidate generator's `update_candidate_strategy()` source documents
`num_matches` as the number of matches between candidate sequences and
target-model predictions.

A runtime wrapper instrumented that method and summed matches and
candidate counts across three measured trials after a separate warm-up.

  -----------------------------------------------------------------------------------
       Draft    Matches     Proposed     Reported     Median   Throughput        Peak
      length              candidates   acceptance    elapsed                allocated
                             counted        ratio       time                     VRAM
  ---------- ---------- ------------ ------------ ---------- ------------ -----------
           3        198          225       88.00%    2.760 s  34.78 tok/s   981.5 MiB

           5        222          252       88.10%    2.758 s  34.80 tok/s   983.4 MiB

           8        234          282       82.98%    2.902 s  33.08 tok/s   985.4 MiB
  -----------------------------------------------------------------------------------

The ratio was computed as
`sum(matches) / sum(proposed candidate tokens)`. It is a candidate-token
match ratio from the instrumented candidate-generator path, not a
speedup percentage or the fraction of all final output tokens emitted
directly from the draft. Treat the denominator as
implementation-specific and validate it against candidate tensor
semantics before comparing across different Transformers versions or
decoding algorithms.

The measured acceptance ratio is high, but it does not guarantee a
speedup. The actual throughput remained below the earlier target-only
baseline, and the target-only comparison used a different prompt length;
a definitive A/B run should use the same prompt, exact output length,
warm-up, and repetition count.

The script emitted warnings about passing `generation_config` alongside
generation arguments and duplicating a
`MinNewTokensLengthLogitsProcessor`. They did not prevent completion,
but should be cleaned up in the next benchmark revision.

------------------------------------------------------------------------

## What the results mean

### Practical takeaways so far

1.  **Quantization can materially reduce memory.** NF4 lowered measured
    peak allocated VRAM by roughly 64% and improved decode throughput in
    the tested workload, although TTFT was slightly worse.
2.  **Attention implementation matters.** SDPA auto was substantially
    faster than eager attention in one small comparison, but the
    experiment was noisy and does not prove FlashAttention was used.
3.  **KV caching becomes much more valuable as the prefix grows.** At
    2048 tokens, cache-on generation took about 1.25 seconds versus 8.17
    seconds with caching disabled.
4.  **Batching can increase aggregate throughput at a memory cost.** The
    Transformers continuous-batching test improved throughput from 40.49
    to 62.98 tok/s for four requests while using roughly twice the peak
    allocated VRAM.
5.  **Throughput is not per-request latency.** More concurrent work can
    increase aggregate tokens/s without making each request finish
    sooner.
6.  **Speculative decoding is not universally faster.** On this small
    target model, the draft model overhead outweighed the gains despite
    high measured draft-token acceptance.
7.  **Benchmark methodology matters.** Warm-up, synchronization, fixed
    prompt and output lengths, consistent EOS settings, multiple
    repetitions, and honest memory metrics are necessary for useful
    comparisons.

### Metrics used

-   **TTFT:** time from the beginning of a request until its first
    generated token is available.
-   **Decode throughput:** generated tokens divided by the measured
    decode or generation time, depending on the script.
-   **Aggregate throughput:** total generated tokens across requests
    divided by batch elapsed time.
-   **Peak allocated VRAM:** PyTorch's
    `torch.cuda.max_memory_allocated()` high-water mark. It is not the
    same as total GPU memory in use, which can include allocator
    reservations, CUDA context, driver use, and other processes.
-   **Acceptance ratio:** counted matching draft candidates divided by
    counted proposed candidates in the instrumentation. It is not
    equivalent to throughput gain.

Do not compare numbers from different scripts as if they were a single
controlled benchmark unless the prompt, token counts, generation
parameters, warm-up and timing method are aligned.

------------------------------------------------------------------------

## Scripts created during the lab

The following script names were used or referenced during the
experiments:

  ------------------------------------------------------------------------
  Script                               Purpose
  ------------------------------------ -----------------------------------
  `paged_batching_comparison.py`       Sequential versus Transformers
                                       continuous batching

  `concurrency_scaling.py`             Throughput/memory at 1, 2, 4 and 8
                                       concurrent requests

  `kv_cache_comparison.py`             Cache enabled versus disabled

  `long_context_kv_scaling.py`         KV-cache scaling across
                                       176--2048-token prompts

  `speculative_decoding.py`            Target-only versus draft-assisted
                                       decoding

  `speculative_draft_sweep.py`         Draft lengths 3, 5 and 8

  `inspect_speculative_internals.py`   Inspect installed Transformers
                                       assisted-decoding internals

  `speculative_acceptance_probe.py`    Instrument candidate matches and
                                       candidate counts

  Triton scale+bias benchmark script   Fused Triton kernel versus two
                                       PyTorch operations
  ------------------------------------------------------------------------

This README documents the names and purpose known from the experiment
history. It does not recreate source files or guarantee every script is
currently present in the working directory. Keep the actual scripts
beside this README and consider recording exact dependency versions in
`requirements-lock.txt`.

------------------------------------------------------------------------

## Current WSL status and next steps

The next goal is to run **actual vLLM and SGLang inference servers**,
not simulate their behavior.

### Windows/WSL diagnostics collected

  ------------------------------------------------------------------------------------------
  Check                                 Result
  ------------------------------------- ----------------------------------------------------
  Windows                               Windows 11 Home Single Language, 25H2, build
                                        26200.9711

  WSL package                           `MicrosoftCorporationII.WindowsSubsystemForLinux`,
                                        version 3.0.3.0, status `Ok`

  WSL status                            Reported `Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG`

  `wsl --list --verbose`                `The file cannot be accessed by the system.`

  `wsl --version`                       Reported WSL was not installed and requested
                                        elevation

  `Microsoft-Windows-Subsystem-Linux`   Disabled

  `VirtualMachinePlatform`              Disabled

  NVIDIA Windows driver                 581.95

  Windows `nvidia-smi`                  RTX 3050 Laptop GPU visible; 6144 MiB VRAM

  Free C: drive space                   775.61 GB

  System RAM                            15.69 GB visible to Windows
  ------------------------------------------------------------------------------------------

### Pending repair

The agreed next step is to run PowerShell as Administrator and enable
the two disabled Windows features:

``` powershell
Enable-WindowsOptionalFeature -Online `
  -FeatureName Microsoft-Windows-Subsystem-Linux `
  -All -NoRestart

Enable-WindowsOptionalFeature -Online `
  -FeatureName VirtualMachinePlatform `
  -All -NoRestart
```

If both commands succeed, restart Windows:

``` powershell
Restart-Computer
```

Then check:

``` powershell
wsl --status
wsl --version
wsl --list --verbose
```

If no Linux distribution is installed, install one only after WSL is
functioning. Then verify GPU passthrough from inside Linux with
`nvidia-smi` before installing an inference server. Do not install a
separate Linux NVIDIA display driver inside WSL; follow the supported
NVIDIA WSL setup for the Windows driver and Linux compute stack.

### Planned real experiments

1.  Verify WSL 2 and NVIDIA GPU passthrough.
2.  Install a compatible vLLM release in an isolated Linux environment.
3.  Serve a small model through vLLM's actual OpenAI-compatible API.
4.  Benchmark real requests: TTFT, end-to-end latency, throughput,
    concurrency, and GPU memory.
5.  Investigate vLLM's PagedAttention and continuous batching through
    actual server behavior and supported metrics.
6.  Test SGLang separately if its dependencies and memory footprint fit
    the laptop.
7.  Compare vLLM and SGLang only with the same model, prompt workload,
    output constraints, warm-up and measurement procedure.

The RTX 3050's 6 GB VRAM and the system's 16 GB RAM make model choice,
quantization, and memory settings important. The large free disk space
is helpful for packages and model caches, but it does not remove RAM or
VRAM limits.

------------------------------------------------------------------------

## Reproducibility checklist

Before trusting a benchmark comparison, record:

-   [ ] Exact Python, PyTorch, Transformers, CUDA-build and driver
    versions.
-   [ ] Exact model IDs, model revisions and precision/quantization.
-   [ ] Prompt text and input token count.
-   [ ] Exact generation parameters (`max_new_tokens`, EOS behavior,
    sampling, cache settings).
-   [ ] Warm-up count and measured repetition count.
-   [ ] Whether GPU synchronization was used around timing.
-   [ ] Whether latency means TTFT, decode-only time or end-to-end
    generation time.
-   [ ] Total generated tokens and whether EOS caused early termination.
-   [ ] Peak allocated VRAM and, when relevant, total GPU memory from
    `nvidia-smi`.
-   [ ] Correctness check between the tested outputs.
-   [ ] Whether results are actual model execution or an educational
    simulation.

### Final note

This lab is a record of measured results, including negative results and
limitations. The goal is not to make every optimization appear
successful; it is to understand when an optimization helps, when it does
not, and why.
