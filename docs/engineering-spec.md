# CoreWarp — especificação de engenharia

## Auditoria do repositório atual (pré-pivot)

| Área | Estado | Decisão |
|------|--------|---------|
| Fila por prioridade + `tick`/`drain` | Útil | Preservar lógica; renomear para scheduler |
| Multi-stream slots | Útil | Preservar |
| Backend host | Útil | Manter como validador de contrato (sem fingir tempos GPU) |
| Feature `cuda` + cudarc/NVRTC | Útil | Isolar em `backend::cuda` |
| SAXPY como único job | Fraco | Expandir para 4 ops |
| `Job` com `Vec` embutido | Substituir | `BufferId` + memory manager |
| Retire CUDA por piso de ms + `synchronize` | Substituir | `CudaEvent::try_is_complete` |
| API `Pipeline` / `Job::saxpy` | Quebrar | API `CoreWarp` + `Operation` |
| Analogias CPU / Byteflow | Já removidas | Manter fora |

## Tese

Runtime Rust leve para workloads CUDA dinâmicos: buffers reutilizáveis, conclusão assíncrona por evento, despacho mensurável. Não compete com Graphs/TensorRT; reduz o boilerplate de muitos jobs independentes.

## Módulos (crate único; crates publicados depois)

```
core        API, Operation, handles, estados
scheduler   prioridade, aging, admission, Fixed|Adaptive
memory      pool por size-class, lifetime até completion event
telemetry   métricas por estágio e agregadas
backend     trait + host + cuda (cudarc só aqui)
```

## Máquina de estados do job

`Queued → Submitted → Running → Completed | Failed`

Transições:
- `submit` → Queued
- emissão em stream livre → Submitted → Running (após enqueue CUDA)
- `event.try_is_complete() == true` → Completed (ou Failed se query retornar erro async)

## Ownership de buffers

- `upload` cria `BufferId` com dados host (+ device lazy no CUDA).
- Ops incrementam `in_flight` nos buffers referenciados.
- Completion decrementa; só então o bloco do pool pode ser reutilizado.
- `download` exige job produtor concluído / buffer sem writers em voo.

## CUDA completion

```
H2D → kernel → D2H (staging) → record CudaEvent
poll: event.try_is_complete() → writeback host → liberar pool → Completed
```

Proibido: piso de milissegundos como critério de retire.

## MVP (ordem)

1. Events + lifecycle + API nova (Semana 1)
2. Memory pool + 4 ops (Semana 2)
3. Telemetria por estágio + Adaptive experimental + benches (Semana 3)

## Fora de escopo

Multi-GPU, multi-processo, work-stealing, fused kernels, Graphs próprios, linguagem de kernels, runtime de IA, OpenCL/Vulkan, GUI.
