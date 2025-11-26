# HIP Kernel Port: LayerNorm & Softmax

This project explores how CUDA compute kernels can be brought up, validated, and
optimized on AMD GPUs using HIP. It focuses on two core operations heavily used
in modern LLM inference pipelines: **LayerNorm** and **Softmax**.

The goal is to understand cross-vendor behavior, ensure numerical correctness,
and compare performance characteristics across CUDA and ROCm environments. The
structure of this repository mirrors a real-world workflow where kernels must
run efficiently on different accelerator platforms and integrate cleanly into a
larger model-serving stack.

---

## Why This Project Exists

Large-scale AI inference relies on fast, reproducible, and highly optimized GPU
kernels. When models are deployed across diverse hardware fleets (e.g., NVIDIA
and AMD accelerators), the underlying kernel implementations must behave
consistently and deliver strong performance on each backend.

This project serves as a practical exploration of:

- **Establishing a CUDA reference implementation**
- **Porting kernels to HIP for AMD GPUs**
- **Validating correctness between CUDA and HIP**
- **Profiling and identifying performance differences**
- **Building a foundation for multi-GPU model execution**

These skills directly map to real-world work in scaling and optimizing
multi-platform inference systems.

---

## Project Goals

### ✔ Bring-Up & Correctness
- Implement baseline CUDA kernels for LayerNorm and Softmax  
- Port each kernel to HIP  
- Validate numerical output parity across both platforms  

### ✔ Kernel Performance & Profiling
- Compare runtime behavior on NVIDIA CUDA vs AMD ROCm  
- Use Nsight Systems / Nsight Compute for CUDA profiling  
- Use rocprof for HIP profiling  
- Investigate memory access, block sizing, and kernel occupancy  

### ✔ Integration-Oriented Structure
While the repo focuses on individual kernels, the layout anticipates integration
into a broader LLM inference stack (e.g., PyTorch, Triton, or custom runtimes):

- modular launchers under `include/`
- standalone kernels under `cuda/` and `hip/`
- reproducible build scripts under `scripts/`
- benchmarking harness under `benchmarks/`

---

## Repository Layout

```

cuda/         – CUDA implementations (LayerNorm, Softmax)
hip/          – HIP ports for AMD ROCm
include/      – Kernel launch headers
scripts/      – Build and profiling scripts (CUDA + HIP)
benchmarks/   – Python benchmarking harness + results
docs/         – Findings, notes, profiling summaries

```

---

## Benchmarking

All benchmarking scripts live under `benchmarks/`.  
This includes:

- timing tests for CUDA and HIP versions  
- hooks to parse profiler outputs  
- initial performance comparisons across hardware  

Results and analysis will be tracked in:

```

benchmarks/benchmark_results.md
docs/findings.md

```

---

## Current Status

- CUDA and HIP kernels scaffolded  
- Build scripts configured  
- Benchmark harness stubbed  
- Profiling hooks in place  
- Documentation structure ready  

This project will evolve as additional correctness checks, optimizations, and
profiling insights are added.

---

## Future Work

- Add mixed-precision variants (FP16/BF16)  
- Introduce kernel fusion experiments  
- Compare algorithmic variants (tree-reduce vs warp-reduce)  
- Explore multi-GPU scalability on AMD hardware  
- Integrate kernels into a minimal LLM inference graph  

---

## License

MIT License. Feel free to explore, benchmark, and adapt the kernels for your own
experiments with cross-platform GPU inference.
```

