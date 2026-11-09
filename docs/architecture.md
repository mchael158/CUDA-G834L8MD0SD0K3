# Arquitetura

```text
submit(Operation) → scheduler → stream slot
        ↓
 ensure resident (H2D pinned só se host_dirty)
        ↓
 kernel (device buffers permanentes)
        ↓
 CudaEvent done  →  poll try_is_complete
        ↓
 mark device_dirty; download faz D2H sob demanda
```

## Componentes

| Módulo | Papel |
|--------|-------|
| `core` | `CoreWarp`, `Operation`, handles |
| `scheduler` | fila, aging, Fixed/Adaptive |
| `memory` | pool host + `host_dirty`/`device_dirty` |
| `telemetry` | métricas; estágios CUDA via events |
| `backend::host` | contratos (não emula GPU) |
| `backend::cuda` | residentes, pinned, events, NVRTC |

## Regras

- Conclusão CUDA só via `try_is_complete`.
- Timing de estágio: `CudaEvent` com `CU_EVENT_DEFAULT`.
- H2D usa pinned staging; D2H só em `download` (ou parciais de reduction).
- Buffer `in_flight > 0` bloqueia `download`/`free`.
