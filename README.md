# CoreWarp

**GPU Compute Toolkit** em Rust — primitivas reutilizáveis, runtime de memória/execução e packs **AI** + **Game** sobre o mesmo núcleo.

## Workspace

```text
crates/corewarp-core        tipos e erros
crates/corewarp-telemetry   métricas
crates/corewarp-memory      pools + dirty flags
crates/corewarp-runtime     Device / Buffer / Submission + CUDA/host
crates/corewarp-kernels     ops compartilhadas
crates/corewarp-ai          normalize, …
crates/corewarp-game        particle update, …
crates/corewarp             facade + prelude
```

## Build

```bat
cargo test -p corewarp-runtime --release
cargo run -p corewarp --release --example pipeline_sim
cargo run -p corewarp --release --example image_processing
cargo run -p corewarp --release --example tensor_operations --features ai
cargo run -p corewarp --release --example particle_simulation --features game
cargo run -p corewarp --release --example bench_host
```

CUDA:

```bat
cargo run -p corewarp --release --example saxpy_cuda --features cuda
cargo run -p corewarp --release --example bench_manual_vs_runtime --features cuda
```

## Uso

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

- [product.md](docs/product.md) — identidade
- [implementation-0.1.md](docs/implementation-0.1.md) — contratos 0.1
- [architecture.md](docs/architecture.md) — runtime
