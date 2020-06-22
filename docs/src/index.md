# SpikingNN.jl

SpikingNN is spiking neural network (SNN) simulator written in Julia that targets multiple platforms. The span of appropriate hardware/software combinations for an arbitrary SNNs is quite wide. Emulating an SNN involves simulating a large, interconnected dynamical system — this allows for sparsity in space and time. If a network is dense in both dimensions, a GPU (or multi-GPU) target is most appropriate. If a network uses temporal coding, its activity is sparse over time, so event-driven simulation is well-suited. Targeting these various platforms should not require code changes or complete knowledge of low-level programming requirements. On the other hand, using glue-languages, like Python, to generate low-level code can leave performance on the table and become difficult to extend and debug.

SpikingNN attempts to address these issues by leveraging the rich Julia ecosystem ([StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl), [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl), to name a few packages). These packages allow us to remap the underlying representation of the network to different platforms. This packages provides a framework for building spiking neural networks and simple functions like [`gpu`](@ref) to map the network to supported targets. It is also designed to be extensible, so that implementing a simple interface allows your model to enjoy the same remapping functionality.

## Suported Platforms

Available platforms:
- Dense CPU
- Dense single-node GPU

Planned platforms:
- Sparse CPU
- Event-driven (sparse in time) CPU
- Block-sparse in time GPU
- Distributed CPU/GPU