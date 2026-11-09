# Architecture

```text
submit(Operation) → scheduler → stream slot
        ↓
 ensure resident (H2D pinned only if host_dirty)
        ↓
 kernel (permanent device buffers)
        ↓
 CudaEvent done  →  poll try_is_complete
        ↓
 mark device_dirty; download performs on-demand D2H
```

## Components

| Module | Role |
|--------|------|
| `core` | `CoreWarp`, `Operation`, handles |
| `scheduler` | queue, aging, Fixed/Adaptive |
| `memory` | host pool + `host_dirty`/`device_dirty` |
| `telemetry` | metrics; CUDA stages via events |
| `backend::host` | contracts (does not emulate GPU) |
| `backend::cuda` | residents, pinned, events, NVRTC |

## Rules

- CUDA completion only via `try_is_complete`.
- Stage timing: `CudaEvent` with `CU_EVENT_DEFAULT`.
- H2D uses pinned staging; D2H only in `download` (or partial reduction paths).
- Buffer with `in_flight > 0` blocks `download`/`free`.
