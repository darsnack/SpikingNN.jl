module Synapse

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.

Expected Fields:
- `T_window::Real`: the length of time that the response function is non-zero
    (i.e. the response will be sampled over the interval `[0, T_window]`)
"""
abstract type AbstractSynapse end

"""
    sample_response(response, dt::Real = 1.0)

Sample the provided response function.
"""
function sample_response(response::AbstractSynapse, dt::Real = 1.0)
    N = Int(response.T_window / dt) # number of samples to acquire
    t = collect(1:N)
    return response.(dt .* t .- dt), N
end

"""
    Delta{T<:Real}

Synapse that returns `q` whenever `t = 0` and zero otherwise.
"""
struct Delta{T<:Real} <: AbstractSynapse
    T_window::T
    q::T
end

"""
    Delta(q::Real = 1; dt::Real = 1.0)

Create a Dirac delta synapse.
Optionally, specify `dt` to compute the appropriate `T_window`.
"""
Delta{T}(q::Real = 1; dt::Real = 1.0) where {T<:Real} = Delta{T}(2 * dt, q)
Delta(q::Real = 1; dt::Real = 1.0) = Delta{Float64}(q; dt = dt)

"""
    (::Delta)(Δ::Real)

Evaluate Dirac delta synapse.
"""
(synapse::Delta)(Δ::Real) = (Δ == 0) ? synapse.q : zero(synapse.q)

"""
    Alpha{T<:Real}

Synapse that returns `(q / τ) * exp(-Δ / τ) Θ(Δ)`
(where `Θ` is the Heaviside function).
"""
struct Alpha{T<:Real} <: AbstractSynapse
    T_window::T
    q::T
    τ::T
end

"""
    Alpha()

Create an alpha synapse.
Optionally, specify `dt` to compute the appropriate `T_window`.
"""
Alpha{T}(q::Real = 1, τ::Real = 1) where {T<:Real} = Alpha{T}(10 * τ, q, τ)
Alpha(q::Real = 1, τ::Real = 1) = Alpha{Float64}(q, τ)

"""
    (::Alpha)(Δ::Real)

Evaluate an alpha synapse.
"""
function (synapse::Alpha)(Δ::Real)
    v = Δ * (synapse.q / synapse.τ) * exp(-(Δ - synapse.τ) / synapse.τ)
    return (Δ >= 0) ? v : zero(v)
end

end