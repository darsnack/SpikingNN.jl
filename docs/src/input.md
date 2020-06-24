# Inputs

SpikingNN provides several synthetic inputs:
- constant rate
- step current
- inhomogenous Poisson process

Inputs can also form a [`InputPopulation`](@ref) that behaves similar to a [`Population`](@ref).

## Constant Rate

A constant rate input fires at a fixed frequency.

```@docs
ConstantRate
evaluate!(::ConstantRate, ::Integer; ::Real)
```

## Step Current

A step current input is low until a fixed time step, then it is high.

```@docs
StepCurrent
evaluate!(::StepCurrent, ::Integer; ::Real)
```

## Inhomogenous Poisson Input Process

An input that behaves like an inhomogenous Poisson process given by a provided instantaneous rate function.

```@docs
PoissonInput
evaluate!(::PoissonInput, ::Integer; ::Real)
```

## Input Population

```@docs
InputPopulation
evaluate!(::InputPopulation, ::Integer; ::Real)
```