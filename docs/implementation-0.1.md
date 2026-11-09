# Implementation spec — CoreWarp 0.1.0

## What to preserve from the current repo

| Component | Destination |
|-----------|-------------|
| `CoreWarp` + scheduler + aging | `corewarp-runtime` |
| `MemoryManager` + dirty flags | `corewarp-memory` |
| Host backend + contracts | `corewarp-runtime::backend::host` |
| CUDA residents / pinned / events | `corewarp-runtime::backend::cuda` |
| Saxpy/VAdd/VScale/Reduction ops | `corewarp-kernels` (+ runtime registry) |
| Host contract tests | `corewarp-runtime` tests |
| Benches | `examples/` + future `benches/` |

## 0.1 workspace (physical crates)

```text
crates/corewarp-core        types, errors (no cudarc)
crates/corewarp-telemetry   JobMetrics / RuntimeMetrics
crates/corewarp-memory      MemoryManager
crates/corewarp-runtime     Device, Buffer, Submission, CoreWarp, backends
crates/corewarp-kernels     shared ops (high-level API)
crates/corewarp-ai          normalize (+ stubs)
crates/corewarp-game        particle_update (+ stubs)
crates/corewarp             facade + prelude
```

> Separate `corewarp-cuda` / `corewarp-host` crates are planned for 0.2; in 0.1 they are isolated modules inside `runtime` (the only place that may use `cudarc`).

## Public contracts

### Device
- `Device::host()` / `Device::cuda(index)`
- `upload`, `allocate`, `submit`, `poll`, `drain`, `shutdown`
- Does not hide streams/priority: `DeviceConfig { streams, max_in_flight, … }`

### Buffer
- Typed handle `Buffer` (wraps `BufferId` + weak `Device` / id)
- `download` only when not in-flight; on-demand D2H if `device_dirty`

### Submission
- `poll() -> Pending | Completed | Failed`
- `wait()`
- `metrics()`

### Operation
- Remains an enum in core; kernels/ai/game build `Operation` or typed submits

## Features

```toml
default = ["host"]
host = []
cuda = ["corewarp-runtime/cuda"]
ai = ["dep:corewarp-ai"]
game = ["dep:corewarp-game"]
full = ["cuda", "ai", "game"]
```

## 0.1 ops (proposal target = 6)

| Op | Crate | 0.1 target status |
|----|-------|-------------------|
| Reduction | kernels | already exists |
| VectorScale | kernels | already exists |
| Normalize | ai | new (host+cuda) |
| ParticleUpdate | game | new (simple host+cuda) |
| PrefixSum | kernels | done (host + CUDA block scan, n≤65536) |
| Histogram | kernels | done (host + CUDA atomics) |

Delivery order aligned with weeks: foundation → CUDA polish → ops → benches.

## 0.1 acceptance criteria

- Workspace builds with `--features host` and `--features cuda`
- Host contract tests 100% pass
- Device API stable enough for examples
- 3 examples: image/tensor/particle (minimal)
- Host + CUDA benches (when GPU is available)
