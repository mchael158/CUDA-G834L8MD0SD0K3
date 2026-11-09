# CoreWarp — product identity

**Category:** GPU Compute Toolkit  
**Language:** Rust 2021  
**Backends 0.1:** Host + CUDA (`cudarc` 0.19)  
**Domains:** AI + Game on the same runtime  
**Target license:** MIT OR Apache-2.0  

## Thesis

A library of **reusable GPU primitives** with a memory runtime, asynchronous execution, and telemetry. It is not an AI framework, a game engine, or a CUDA replacement.

## Layers

```text
corewarp-ai / corewarp-game     ← domain packs
corewarp-kernels                ← shared ops
corewarp-runtime                ← Device, submit, streams, events
corewarp-memory                 ← pools, lifetime, dirty flags
corewarp-core + telemetry       ← types, errors, metrics
```

## Differentiators (0.1)

1. Reusable primitives (not just kernel launch)
2. Memory-aware execution (resident buffers + dirty flags)
3. Composition without mandatory download between ops
4. Benchmarking / per-stage metrics

## Out of scope (0.1)

GUI, multi-GPU, Vulkan, custom CUDA Graphs, replacing cuBLAS/cuDNN/TensorRT.
