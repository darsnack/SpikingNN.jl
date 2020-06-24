# Threshold Models

The following threshold models are supported:
- Ideal
- Inhomogenous Poisson process

## Ideal

```@docs
Threshold.Ideal
evaluate!(::Threshold.Ideal, ::Real, ::Real; ::Real)
```

## Inhomogenous Poisson Process

```@docs
Threshold.Poisson
evaluate!(::Threshold.Poisson, ::Integer, ::Real; ::Real)
```