abstract type AbstractLearner end

"""
    STDP

A simple spike-timing-dependent plasticity mechanism.

Fields:
- `A₊::Real`: positive weight change amplitude
- `A₋::Real`: negative weight change amplitude
- `τ₊::Real`: positive decay parameter
- `τ₋::Real`: negative decay parameter
- `lastpre::Array{Float64, 2}`: the last occurence of a pre-synaptic spike
    (matrix row = src, col = dest)
- `lastpost::Array{Float64, 2}`: the last occurence of a post-synaptic spike
    (matrix row = src, col = dest)
- `Δw::Array{Float64, 2}`: the current synaptic weight change
    (matrix row = src, col = dest)
"""
struct STDP{T<:Real, S<:AbstractArray{<:Real}} <: AbstractLearner
    Ap::T
    An::T
    τp::T
    τn::T
    lastpre::S
    lastpost::S
end

"""
    STDP(A₀::Real, τ::Real, n::Integer)

Create an STDP learner for `n` neuron population with weight change amplitude `A₀` and decay `τ`.
"""
STDP{T}(;A::Real, τ::Real, n::Integer) where T =
    STDP{T, Matrix{T}}(A, -A, τ, τ, zeros(T, n, n), zeros(T, n, n))
STDP(;kwargs...) = STDP{Float32}(;kwargs...)

function stdp(Ap, An, τp, τn, tpre, tpost)
    Δt = tpre - tpost
    
    return (Δt >= 0) ? Ap * exp(-Δt / τp) : An * exp(-Δt / τn)
end

function update!(learner::STDP, w, t, prespikes, postspikes; dt = 1.0)
    learner.lastpre .= w .* prespikes
    learner.lastpost .= transpose(w) .* postspikes
    w .+= (prespikes .+ transpose(postspikes) .> 0) .* 
          stdp.(learner.Ap, learner.An, learner.τp, learner.τn, learner.lastpre, learner.lastpost)

    return w
end