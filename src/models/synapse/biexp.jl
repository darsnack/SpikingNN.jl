"""
    biexp(t::Real, lastspike, q, taum, taus)
    biexp(t::AbstractArray, q::AbstractArray, taum::AbstractArray, taus::AbstractArray)
    biexp!(I::AbstractArray, t::AbstractArray, q::AbstractArray, taum::AbstractArray, taus::AbstractArray)

Evaluate an bi-exponential synaptic response at time `t`.
Modeled as `(q / τm - τs) * (exp(-t / τm) - exp(-t / τs)) Θ(t)`
(where `Θ` is the Heaviside function).
Store the output current in `I` when specified.

!!! note
    `taum` must not equal `taus` or an error will be thrown

# Fields
- `I`: destination array for the output current
- `t`: current time
- `q`: amplitude
- `taum`: rise time constant
- `taus`: fall time constant
"""
biexp(t::Real, q, taum, taus) =
    (t >= 0 && t < Inf) ? q / (1 - taus / taum) * (exp(-t / taum) - exp(-t / taus)) : zero(q)
biexp(t::AbstractArray, q::AbstractArray, taum::AbstractArray, taus::AbstractArray) =
    biexp!(similar(q), t, q, taum, taus)
function biexp!(I::AbstractArray,
                t::AbstractArray, q::AbstractArray, taum::AbstractArray, taus::AbstractArray)
    # @tullio I[i] = q[i] / (1 - taus[i] / taum[i]) * (exp(-t[i] / taum[i]) - exp(-t[i] / taus[i]))
    # @tullio I[i] = (t[i] >= 0 && t[i] < Inf) ? I[i] : zero(I[i])

    @avx @. I = q / (1 - taus / taum) * (exp(-t / taum) - exp(-t / taus))
    map!((δ, i) -> (δ >= 0) && (δ < Inf) ? i : zero(i), I, t, I)

    return I
end
function biexp!(I::CuArray, t::CuArray, q::CuArray, taum::CuArray, taus::CuArray)
    @. I = (t >= 0) * (t < Inf) * q / (1 - taus / taum) * (exp(-t / taum) - exp(-t / taus))

    return I
end

"""
    BiExponential(ϵ₀, τm, τs, buffsize = 100, dims = (1,))
    BiExponential(; ϵ₀ = 1f0, τm = 10f0, τs = 4f0, buffsize = 100, dims = (1,))

Synapse that returns current equal to

``\\sum_{t^f \\in \\mathcal{T}^f}
    \\frac{\\epsilon_0}{\\tau_m - \\tau_s} \\left[e^{-(t - t^f) / \\tau_m}} - e^{-(t - t^f) / \\tau_s}\\right]
    \\Theta(t - t^f)``

(where ``\\Theta`` is the Heaviside function and ``\\mathcal{T}^f`` is the set of pre-synaptic spikes).
Set `buffsize` to control how many pre-synaptic spikes are remembered.
"""
struct BiExponential{T<:AbstractArray{<:Real}, S<:ArrayOfImpulseBuffers} <: AbstractSynapse
    spikes::S
    ϵ₀::T
    τm::T
    τs::T
end

BiExponential(ϵ₀, τm, τs, buffsize = 100, dims = (1,)) =
    BiExponential(ArrayOfImpulseBuffers{eltype(ϵ₀)}(buffsize),
                  _fillmemaybe(ϵ₀, dims),
                  _fillmemaybe(τm, dims),
                  _fillmemaybe(τs, dims))
BiExponential(; ϵ₀ = 1f0, τm = 10f0, τs = 4f0, buffsize = 100, dims = (1,)) =
    BiExponential(ϵ₀, τm, τs, buffsize, dims)

"""
    excite!(synapse::BiExponential, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
function excite!(synapse::BiExponential, spikes)
    spiked = spike .> 0
    push!(view(synapse.spikes, spiked), view(spikes, spike))

    return synapse
end

# isactive(synapse::BiExponential, t::Integer; dt::Real) = dt * (t - first(synapse.spikes)) <= synapse.τs + 8 * synapse.τm
# isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:BiExponential} =
#     any(dt .* (t .- first.(synapses.spikes)) .<= synapses.τs .+ 8 .* synapses.τm)

"""
    evaluate!(synapse::BiExponential, t; dt = 1)
    evaluate!(current, synapses::BiExponential, t; dt = 1)

Evaluate an BiExponential synapse. See [`Synapse.BiExponential`](@ref).
"""
evaluate!(synapse::BiExponential, t; dt = 1) =
    conv_impulses(t -> biexp(t * dt, synapse.ϵ₀, synapse.τm, synapse.τs), t, synapse.spikes)
function evaluate!(current, synapses::BiExponential, t; dt = 1)
    current .= conv_impulses!((I, t) -> biexp!(I, t * dt, synapses.ϵ₀, synapses.τm, synapses.τs),
                              current, t, synapses.spikes)

    return current
end

"""
    reset!(synapse::BiExponential)

Reset `synapse` by emptying the queue of pre-synaptic spikes
"""
reset!(synapse::BiExponential, mask = trues(size(synapse))) = empty!(view(synapse.spikes, mask))
