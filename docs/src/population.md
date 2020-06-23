# Populations

A population of neurons is a collection of a single type of neuron connected by a single type of synapse. A learning mechanism is associated with the synapses of a population.

```@docs
Population
size(::Population)
neurons
synapses
evaluate!(::Population, ::Integer; ::Real, ::Any, ::Any)
update!(::Population, ::Integer; ::Real)
reset!(::Population)
simulate!(::Population, ::Integer; ::Real, ::Any, ::Any, ::Any)
```