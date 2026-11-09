# Spec de implementação — CoreWarp 0.1.0

## O que preservar do repo atual

| Componente | Destino |
|------------|---------|
| `CoreWarp` + scheduler + aging | `corewarp-runtime` |
| `MemoryManager` + dirty flags | `corewarp-memory` |
| Host backend + contratos | `corewarp-runtime::backend::host` |
| CUDA residentes / pinned / events | `corewarp-runtime::backend::cuda` |
| Ops Saxpy/VAdd/VScale/Reduction | `corewarp-kernels` (+ registry runtime) |
| Testes de contrato host | `corewarp-runtime` tests |
| Benches | `examples/` + futuro `benches/` |

## Workspace 0.1 (crates físicas)

```text
crates/corewarp-core        tipos, erros (sem cudarc)
crates/corewarp-telemetry   JobMetrics / RuntimeMetrics
crates/corewarp-memory      MemoryManager
crates/corewarp-runtime     Device, Buffer, Submission, CoreWarp, backends
crates/corewarp-kernels     ops compartilhadas (API alta)
crates/corewarp-ai          normalize (+ stubs)
crates/corewarp-game        particle_update (+ stubs)
crates/corewarp             facade + prelude
```

> `corewarp-cuda` / `corewarp-host` como crates separados ficam para 0.2; em 0.1 são módulos isolados dentro de `runtime` (único ponto que pode usar `cudarc`).

## Contratos públicos

### Device
- `Device::host()` / `Device::cuda(index)`
- `upload`, `allocate`, `submit`, `poll`, `drain`, `shutdown`
- Não esconde streams/prioridade: `DeviceConfig { streams, max_in_flight, … }`

### Buffer
- Handle tipado `Buffer` (wrap `BufferId` + `Device` fraco / id)
- `download` só quando não in-flight; D2H sob demanda se `device_dirty`

### Submission
- `poll() -> Pending | Completed | Failed`
- `wait()`
- `metrics()`

### Operation
- Continua enum no core; kernels/ai/game constroem `Operation` ou submits tipados

## Features

```toml
default = ["host"]
host = []
cuda = ["corewarp-runtime/cuda"]
ai = ["dep:corewarp-ai"]
game = ["dep:corewarp-game"]
full = ["cuda", "ai", "game"]
```

## Ops 0.1 (meta da proposta = 6)

| Op | Crate | Status alvo 0.1 |
|----|-------|-----------------|
| Reduction | kernels | já existe |
| VectorScale | kernels | já existe |
| Normalize | ai | novo (host+cuda) |
| ParticleUpdate | game | novo (host+cuda simples) |
| PrefixSum | kernels | feito (host + CUDA scan em blocos, n≤65536) |
| Histogram | kernels | feito (host + CUDA atomics) |

Ordem de entrega alinhada às semanas: foundation → CUDA polish → ops → benches.

## Critérios de aceite 0.1

- Workspace compila com `--features host` e `--features cuda`
- Testes host de contrato 100%
- Device API estável o bastante para examples
- 3 examples: image/tensor/particle (mínimos)
- Benches host + cuda (quando GPU disponível)
