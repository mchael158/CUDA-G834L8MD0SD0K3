# CoreWarp

**GPU Compute Toolkit** in Rust — reusable primitives, a memory/execution runtime, and **AI** + **Game** packs on the same core.

## Workspace

```text
crates/corewarp-core        types and errors
crates/corewarp-telemetry   metrics
crates/corewarp-memory      pools + dirty flags
crates/corewarp-runtime     Device / Buffer / Submission + CUDA/host
crates/corewarp-kernels     shared ops
crates/corewarp-ai          normalize, …
crates/corewarp-game        particle update, …
crates/corewarp             facade + prelude
```

## Build

```bash
cargo test -p corewarp-runtime --release
cargo run -p corewarp --release --example pipeline_sim
cargo run -p corewarp --release --example image_processing
cargo run -p corewarp --release --example tensor_operations --features ai
cargo run -p corewarp --release --example particle_simulation --features game
cargo run -p corewarp --release --example bench_host
```

CUDA:

```bash
cargo run -p corewarp --release --example saxpy_cuda --features cuda
cargo run -p corewarp --release --example bench_manual_vs_runtime --features cuda
```

## Usage

```rust
use corewarp::prelude::*;

let mut device = Device::host()?;
let input = device.upload(&data)?;
let output = device.allocate(data.len())?;
let sub = device.submit(Operation::VectorScale {
    alpha: 2.0,
    input: input.id(),
    output: output.id(),
})?;
device.wait(&sub)?;
let result = device.download(&output)?;
```

## Docs

- [product.md](docs/product.md) — product identity
- [implementation-0.1.md](docs/implementation-0.1.md) — 0.1 contracts
- [architecture.md](docs/architecture.md) — runtime architecture
