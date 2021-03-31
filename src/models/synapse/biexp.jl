"""
    biexp(t::Real, lastspike, q, taum, taus)
    biexp(t::Real, lastspike::AbstractArray{<:Real}, q::AbstractArray{<:Real}, taum::AbstractArray{<:Real}, taus::AbstractArray{<:Real})
    biexp(t::Real, lastspike::CuArray, q::CuVecOrMat{<:Real}, taum::CuVecOrMat{<:Real}, taus::CuVecOrMat{<:Real})

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
    BiExponential{T<:Real}
    BiExponential{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100)
    BiExponential(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100)

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function and `Δ = t - lastspike`).
Set `N` to control how many pre-synaptic spikes are remembered.
"""
struct BiExponential{T<:Real} <: AbstractSynapse
    spikes::CircularArray{T}
    ϵ₀::T
    τm::T
    τs::T
end
BiExponential{T}(;ϵ₀::Real = 1, τm::Real = 10, τs::Real = 4, N = 100) where T<:Real =
    BiExponential{T}(CircularArray{T}(N), ϵ₀, τm, τs)
BiExponential(;kwargs...) = BiExponential{Float32}(;kwargs...)

"""
    excite!(synapse::BiExponential, spike::Integer)
    excite!(synapses::AbstractArray{<:BiExponential}, spike::Integer)

Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
excite!(synapse::BiExponential, spike::Integer) = (spike > 0) && push!(synapse.spikes, spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:BiExponential} =
    (spike > 0) && push!(synapses.spikes, spike)

"""
    spike!(synapse::BiExponential, spike::Integer; dt::Real = 1.0)
    spike!(synapses::AbstractArray{<:BiExponential}, spikes; dt::Real = 1.0)

Reset `synapse` when the post-synaptic neuron spikes.
"""
spike!(synapse::BiExponential, spike::Integer; dt::Real = 1.0) = reset!(synapse)
spike!(synapses::T, spikes; dt::Real = 1.0) where T<:AbstractArray{<:BiExponential} = reset!(synapses)

isactive(synapse::BiExponential, t::Integer; dt::Real) = dt * (t - first(synapse.spikes)) <= synapse.τs + 8 * synapse.τm
isactive(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:BiExponential} =
    any(dt .* (t .- first.(synapses.spikes)) .<= synapses.τs .+ 8 .* synapses.τm)

"""
    evaluate!(synapse::BiExponential, t::Integer; dt::Real = 1.0)
    (synapse::BiExponential)(t::Integer; dt::Real = 1.0)
    evaluate!(synapses::AbstractArray{<:BiExponential}, t::Integer; dt::Real = 1.0)
    evaluate!(current, synapses::AbstractArray{<:BiExponential}, t::Integer; dt::Real = 1.0)

Evaluate an BiExponential synapse. See [`Synapse.BiExponential`](@ref).
"""
evaluate!(synapse::BiExponential, t::Integer; dt::Real = 1.0) =
    conv_impulses(t -> biexp(t * dt, synapse.ϵ₀, synapse.τm, synapse.τs), t, synapse.spikes)
(synapse::BiExponential)(t::Integer; dt::Real = 1.0) = evaluate!(synapse, t; dt = dt)
evaluate!(synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:BiExponential} =
    conv_impulses(t -> biexp(t * dt, synapses.ϵ₀, synapses.τm, synapses.τs), t, synapses.spikes)
function evaluate!(current, synapses::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:BiExponential}
    current .= conv_impulses!((I, t) -> biexp!(I, t * dt, synapses.ϵ₀, synapses.τm, synapses.τs),
                              current, t, synapses.spikes)

    return current
end

"""
    reset!(synapse::BiExponential)
    reset!(synapses::AbstractArray{<:BiExponential})

Reset `synapse`.
"""
reset!(synapse::BiExponential) = empty!(synapse.spikes)
reset!(synapses::T) where T<:AbstractArray{<:BiExponential}= empty!(synapses.spikes)