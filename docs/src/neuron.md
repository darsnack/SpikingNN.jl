# Neurons

A neuron in SpikingNN is compromised of a soma (cell body) and vector of synapses (dendrites), much like a biological neuron. The soma itself is broken into the cell and threshold function. Beyond making SpikingNN flexible and extensible, this breakdown of a neuron into parts allows for efficient evaluation.

A [`Neuron`](@ref) can be simulated with `simulate!`.

```@docs
simulate!(::Neuron, ::Integer; ::Real, ::Any, ::Any)
```

## Soma

A [`Soma`](@ref) is made up of a cell body (found in [Neuron Models](@ref)) and theshold function (found in [Threshold Models](@ref)).

```@docs
Soma
excite!(::Union{Soma, AbstractArray{<:Soma}}, ::Any)
evaluate!(::Soma, t::Integer; dt::Real = 1.0)
reset!(::Union{Soma, AbstractArray{<:Soma}})
```

## Neuron

Neurons are the smallest simulated block in SpikingNN.

```@docs
Neuron
connect!(::Neuron, ::AbstractSynapse)
getvoltage(::Neuron)
excite!(::Neuron, ::Integer)
evaluate!(::Neuron, ::Integer; ::Real)
reset!(::Union{Neuron, AbstractArray{<:Neuron}})
```