# Synapse Models

The following synapse models are supported:
- Dirac-Delta
- Alpha
- Excitatory post-synaptic potential (EPSP)

The following wrapper types are also supported:
- [`QueuedSynapse`](@ref)
- [`DelayedSynapse`](@ref)

## Default function implementations

Some function implementations apply to all synapses (unless they override them) that inherit from [`AbstractSynapse`](@ref).

```@docs
excite!(::AbstractSynapse, ::Vector{<:Integer})
spike!(::AbstractSynapse, ::Integer; ::Real)
```

## Dirac-Delta

```@docs
Synapse.Delta
excite!(::Synapse.Delta, ::Integer)
evaluate!(::Synapse.Delta, ::Integer; ::Real)
reset!(::Synapse.Delta)
```

## Alpha

```@docs
Synapse.Alpha
excite!(::Synapse.Alpha, ::Integer)
evaluate!(::Synapse.Alpha, ::Integer; ::Real)
reset!(::Synapse.Alpha)
```

## Exitatory Post-Synaptic Potential

```@docs
Synapse.EPSP
excite!(::Synapse.EPSP, ::Integer)
spike!(::Synapse.EPSP, ::Integer; ::Real)
evaluate!(::Synapse.EPSP, ::Integer; ::Real)
reset!(::Synapse.EPSP)
```

## Queued Synapse

```@docs
QueuedSynapse
excite!(::QueuedSynapse, ::Integer)
evaluate!(::QueuedSynapse, ::Integer; ::Real)
reset!(::QueuedSynapse)
```

## Delayed Synapse

```@docs
DelayedSynapse
excite!(::DelayedSynapse, ::Integer)
evaluate!(::DelayedSynapse, ::Integer; ::Real)
reset!(::DelayedSynapse)
```