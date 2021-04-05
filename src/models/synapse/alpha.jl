"""
    alpha(t, q, tau)
    alpha(t::AbstractArray, q::AbstractArray, tau::AbstractArray)
    alpha!(I::AbstractArray, t::AbstractArray, q::AbstractArray, tau::AbstractArray)

Evaluate an alpha synaptic response at time `t`.
Modeled as `t * (q / τ) * exp(-(t - τ) / τ) Θ(t)`
(where `Θ` is the Heaviside function).
Store the output current in `I` when specified.

# Fields
- `I`: destination array for output current
- `t`: current time
- `q`: amplitude
- `tau`: time constant
"""
alpha(t, q, tau) = (t >= 0 && t < Inf) ? t * (q / tau) * exp(-(t - tau) / tau) : zero(q)
alpha(t::AbstractArray, q::AbstractArray, tau::AbstractArray) = alpha!(similar(q), t, q, tau)
function alpha!(I::AbstractArray, t::AbstractArray, q::AbstractArray, tau::AbstractArray)
    @avx @. I = t * (q / tau) * exp(-(t - tau) / tau)
    map!((δ, i) -> (δ >= 0) && (δ < Inf) ? δ * i : zero(i), I, Δ, I)

    return I
end
function alpha!(I::CuArray, t::CuArray, q::CuArray, tau::CuArray)
    @. I = (t >= 0) * (t < Inf) * t * (q / tau) * exp(-(t - tau) / tau)

    return I
end

"""
    Alpha{IT<:Integer, VT<:Real}
    Alpha{IT, VT}(;q::Real = 1, τ::Real = 1)
    Alpha(;q::Real = 1, τ::Real = 1)

Synapse that returns
`(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
mutable struct Alpha{IT<:Integer, VT<:Real} <: AbstractSynapse
    lastspike::VT
    q::VT
    τ::VT
end
Alpha{IT, VT}(;q::Real = 1, τ::Real = 1) where {IT<:Integer, VT<:Real} = Alpha{IT, VT}(-Inf, q, τ)
Alpha(;q::Real = 1, τ::Real = 1) = Alpha{Int, Float32}(q = q, τ = τ)

"""
    excite!(synapse::Alpha, spike::Integer)
    excite!(synapses::AbstractArray{<:Alpha}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
function excite!(synapse::Alpha, spike::Integer)
    if spike > 0
        synapse.lastspike = spike
    end

    return synapse
end
function excite!(synapses::AbstractArray{<:Alpha}, spike::Integer)
    if spike > 0
        synapses.lastspike .= spike
    end

    return synapses
end

# isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = dt * (t - synapse.lastspike) <= 10 * synapse.τ
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
#     any(dt .* (t .- synapses.lastspike) .<= 10 .* synapses.τ)

"""
    evaluate!(synapse::Alpha, t::Integer; dt::Real = 1.0)
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:Alpha}, t::Integer; dt::Real = 1.0)
    evaluate!(current, synapses::AbstractArray{<:Alpha}, t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
evaluate!(synapse::Alpha, t::Integer; dt::Real = 1.0) =
    alpha((t - synapse.lastspike) * dt, synapse.q, synapse.τ)
(synapse::Alpha)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
    alpha((t .- synapses.lastspike) * dt, synapses.q, synapses.τ)
evaluate!(current, synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:Alpha} =
    alpha!(current, (t .- synapses.lastspike) * dt, synapses.q, synapses.τ)

"""
    reset!(synapse::Alpha)
    reset!(synapses::AbstractArray{<:Alpha})

Reset `synapse`.
"""
function reset!(synapse::Alpha)
    synapse.lastspike = -Inf
    
    return synapse
end
function reset!(synapses::AbstractArray{<:Alpha})
    synapses.lastspike .= -Inf

    return synapses
end