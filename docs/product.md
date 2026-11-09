# CoreWarp — identidade de produto

**Categoria:** GPU Compute Toolkit  
**Linguagem:** Rust 2021  
**Backends 0.1:** Host + CUDA (`cudarc` 0.19)  
**Domínios:** AI + Game sobre o mesmo runtime  
**Licença alvo:** MIT OR Apache-2.0  

## Tese

Biblioteca de **primitivas GPU reutilizáveis** com runtime de memória, execução assíncrona e telemetria. Não é framework de IA, engine de jogos nem substituto do CUDA.

## Camadas

```text
corewarp-ai / corewarp-game     ← packs de domínio
corewarp-kernels                ← ops compartilhadas
corewarp-runtime                ← Device, submit, streams, events
corewarp-memory                 ← pools, lifetime, dirty flags
corewarp-core + telemetry       ← tipos, erros, métricas
```

## Diferenciais (0.1)

1. Primitivas reutilizáveis (não só launch de kernel)
2. Execução memory-aware (buffers residentes + dirty flags)
3. Composição sem download obrigatório entre ops
4. Benchmarking / métricas por estágio

## Fora de escopo (0.1)

GUI, multi-GPU, Vulkan, CUDA Graphs próprios, substituir cuBLAS/cuDNN/TensorRT.
