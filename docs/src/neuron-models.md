# Neuron Models

The following neuron (cell body) models are supported:
- Leaky integrate-and-fire (LIF)
- Simplified spike response model (SRM0)

## Leaky Integrate-and-Fire

```@docs
LIF
excite!(::LIF, ::Any)
spike!(::LIF, ::Integer; ::Real)
evaluate!(::LIF, ::Integer; ::Real)
reset!(::LIF)
```

## Simplified Spike Response Model

```@docs
SRM0
excite!(::SRM0, ::Any)
spike!(::SRM0, ::Integer; ::Real)
evaluate!(::SRM0, ::Integer; ::Real)
reset!(::SRM0)
```