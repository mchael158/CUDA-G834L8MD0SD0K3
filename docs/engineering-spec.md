# CoreWarp — engineering specification

## Audit of the current repository (pre-pivot)

| Area | State | Decision |
|------|-------|----------|
| Priority queue + `tick`/`drain` | Useful | Preserve logic; rename to scheduler |
| Multi-stream slots | Useful | Preserve |
| Host backend | Useful | Keep as contract validator (no fake GPU timing) |
| Feature `cuda` + cudarc/NVRTC | Useful | Isolate in `backend::cuda` |
| SAXPY as the only job | Weak | Expand to 4 ops |
| `Job` with embedded `Vec` | Replace | `BufferId` + memory manager |
| CUDA retire by ms floor + `synchronize` | Replace | `CudaEvent::try_is_complete` |
| `Pipeline` / `Job::saxpy` API | Break | `CoreWarp` + `Operation` API |
| CPU / Byteflow analogies | Already removed | Keep out |

## Thesis

Lightweight Rust runtime for dynamic CUDA workloads: reusable buffers, event-based async completion, measurable dispatch. Does not compete with Graphs/TensorRT; reduces boilerplate for many independent jobs.

## Modules (single crate; published crates later)

```
core        API, Operation, handles, states
scheduler   priority, aging, admission, Fixed|Adaptive
memory      pool by size-class, lifetime until completion event
telemetry   per-stage and aggregate metrics
backend     trait + host + cuda (cudarc only here)
```

## Job state machine

`Queued → Submitted → Running → Completed | Failed`

Transitions:
- `submit` → Queued
- dispatch on a free stream → Submitted → Running (after CUDA enqueue)
- `event.try_is_complete() == true` → Completed (or Failed if async query returns error)

## Buffer ownership

- `upload` creates `BufferId` with host data (+ lazy device on CUDA).
- Ops increment `in_flight` on referenced buffers.
- Completion decrements; only then can the pool block be reused.
- `download` requires the producing job completed / buffer with no in-flight writers.

## CUDA completion

```
H2D → kernel → D2H (staging) → record CudaEvent
poll: event.try_is_complete() → writeback host → release pool → Completed
```

Forbidden: millisecond floor as retire criterion.

## MVP (order)

1. Events + lifecycle + new API (Week 1)
2. Memory pool + 4 ops (Week 2)
3. Per-stage telemetry + experimental Adaptive + benches (Week 3)

## Out of scope

Multi-GPU, multi-process, work-stealing, fused kernels, custom Graphs, kernel language, AI runtime, OpenCL/Vulkan, GUI.
