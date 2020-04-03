module Synapse

export excite!

using SNNlib.Synapse: delta, alpha, epsp
using DataStructures
using Base: @kwdef

_ispending(synapse, t) = !isempty(synapse.spikes) && first(synapse.spikes) <= t
function _shiftspike!(synapse, t; dt)
    while _ispending(synapse, t)
        synapse.lastspike .= dequeue!(synapse.spikes) * dt
    end

    return synapse
end

"""
    AbstractSynapse

Inherit from this type to create a concrete synapse.
"""
abstract type AbstractSynapse end

"""
    push!(synapse::AbstractSynapse, spike::Integer)
    push!(synapse::AbstractSynapse, spikes::Vector{<:Integer})

Push a spike(s) into a synapse. The synapse decides how to process this event.
"""
excite!(synapse::Function, spike::Integer) = nothing
excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer}) = map(x -> excite!(synapse, x), spikes)

"""
    Delta{IT<:Integer, VT<:Real}

A synapse representing a Dirac-delta at `lastspike`.
"""
@kwdef struct Delta{I<:Integer, VT} <: AbstractSynapse
    lastspike::VT = [-1.0]
    spikes::Queue{I} = Queue{Int}()
    q::VT = [1.0]
end

excite!(synapse::Delta, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::Delta, t::Integer; dt::Real = 1.0) = (t == synapse.lastspike[]) || _ispending(synapse, t)

paramtype(::Type{T}) where {VT, T<:Delta{<:Any, VT}} = Tuple{VT, VT}
function packparams!(::Type{<:Delta}, synapses)
    VT = typeof(synapses[1].q)
    lastspikes = similar(VT, axes(synapses))
    qs = similar(VT, axes(synapses))
    for (i, synapse) in enumerate(synapses)
        lastspikes[i] = synapse.lastspike[]
        qs[i] = synapse.q[]
        synapses[i] = Delta(view(lastspikes, i), synapse.spikes, view(qs, i))
    end

    return (lastspikes, qs)
end
packparams!(synapses::VecOrMat{T}) where T<:Delta = packparams!(Delta, synapses)
function packparams(synapses::VecOrMat{<:Delta})
    newsynapses = similar(synapses, Any)
    copyto!(newsynapses, synapses)
    ps = packparams!(Delta, newsynapses)

    return (convert.(typeof(newsynapses[1]), newsynapses), ps)
end

"""
    (synapse::Delta)(t::Integer; dt::Real = 1.0)

Return `synapse.q` if `t == synapse.lastspike` otherwise return zero.
"""
function (synapse::Delta)(t::Integer; dt::Real = 1.0)
    _shiftspike!(synapse, t; dt = dt)

    return delta(t * dt, synapse.lastspike, synapse.q)
end
function evalsynapses(synapses::VecOrMat{T}, t::Integer, ps...; dt::Real = 1.0) where T<:Delta
    map(s -> _shiftspike!(s, t; dt = dt), synapses)

    return delta(t * dt, ps...)
end

"""
    Alpha{IT<:Integer, VT<:Real}

Synapse that returns `(t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike)`
(where `Θ` is the Heaviside function).
"""
@kwdef struct Alpha{I<:Integer, VT} <: AbstractSynapse
    lastspike::VT = [-1.0]
    spikes::Queue{I} = Queue{Int}()
    q::VT = [1.0]
    τ::VT = [1.0]
end

excite!(synapse::Alpha, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::Alpha, t::Real; dt::Real = 1.0) = _ispending(synapse, t) ||
                                                    (synapse.lastspike[] > 0 && dt * (t - synapse.lastspike[]) <= 10 * synapse.τ)

paramtype(::Type{T}) where {VT, T<:Alpha{<:Any, VT}} = Tuple{VT, VT, VT}
function packparams!(::Type{<:Alpha}, synapses)
    VT = typeof(synapses[1].q)
    lastspikes = similar(VT, axes(synapses))
    qs = similar(VT, axes(synapses))
    τs = similar(VT, axes(synapses))
    for (i, synapse) in enumerate(synapses)
        lastspikes[i] = synapse.lastspike[]
        qs[i] = synapse.q[]
        τs[i] = synapse.τ[]
        synapses[i] = Alpha(view(lastspikes, i), synapse.spikes, view(qs, i), view(τs, i))
    end

    return (lastspikes, qs, τs)
end
packparams!(synapses::VecOrMat{T}) where T<:Alpha = packparams!(Alpha, synapses)
function packparams(synapses::VecOrMat{<:Alpha})
    newsynapses = similar(synapses, Any)
    copyto!(newsynapses, synapses)
    ps = packparams!(Alpha, newsynapses)

    return (convert.(typeof(newsynapses[1]), newsynapses), ps)
end

"""
    (synapse::Alpha)(t::Integer; dt::Real = 1.0)

Evaluate an alpha synapse. See [`Synapse.Alpha`](@ref).
"""
function (synapse::Alpha)(t::Integer; dt::Real = 1.0)
    _shiftspike!(synapse, t; dt = dt)

    return alpha(t * dt, synapse.lastspike, synapse.q, synapse.τ)
end
function evalsynapses(synapses::VecOrMat{T}, t::Integer, ps...; dt::Real = 1.0) where T<:Alpha
    map(s -> _shiftspike!(s, t; dt = dt), synapses)

    return alpha(t * dt, ps...)
end

"""
    EPSP{T<:Real}

Synapse that returns `(ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ)`
(where `Θ` is the Heaviside function and `Δ = t - lastspike`).

Specifically, this is the EPSP time course for the SRM0 model introduced by Gerstner.
Details: [Spiking Neuron Models: Single Neurons, Populations, Plasticity]
         (https://icwww.epfl.ch/~gerstner/SPNM/node27.html#SECTION02323400000000000000)
"""
@kwdef struct EPSP{I<:Integer, VT} <: AbstractSynapse
    lastspike::VT = [-1.0]
    spikes::Queue{I} = Queue{Int}()
    ϵ₀::VT = [1.0]
    τm::VT = [1.0]
    τs::VT = [1.0]
end

excite!(synapse::EPSP, spike::Integer) = enqueue!(synapse.spikes, spike)

isactive(synapse::EPSP, t::Integer; dt::Real) = _ispending(synapse, t) ||
                                                (synapse.lastspike[] > 0 && dt * (t - synapse.lastspike[]) <= synapse.τs + 8 * synapse.τm)

paramtype(::Type{T}) where {VT, T<:EPSP{<:Any, VT}} = Tuple{VT, VT, VT, VT}
function packparams!(::Type{<:EPSP}, synapses)
    VT = typeof(synapses[1].ϵ₀)
    lastspikes = similar(VT, axes(synapses))
    ϵ₀s = similar(VT, axes(synapses))
    τms = similar(VT, axes(synapses))
    τss = similar(VT, axes(synapses))
    for (i, synapse) in enumerate(synapses)
        lastspikes[i] = synapse.lastspike[]
        ϵ₀s[i] = synapse.ϵ₀[]
        τms[i] = synapse.τm[]
        τss[i] = synapse.τs[]
        synapses[i] = EPSP(view(lastspikes, i), synapse.spikes, view(ϵ₀s, i), view(τms, i), view(τss, i))
    end

    return (lastspikes, ϵ₀s, τms, τss)
end
packparams!(synapses::VecOrMat{T}) where T<:EPSP = packparams!(EPSP, synapses)
function packparams(synapses::VecOrMat{<:EPSP})
    newsynapses = similar(synapses, Any)
    copyto!(newsynapses, synapses)
    ps = packparams!(EPSP, newsynapses)

    return (convert.(typeof(newsynapses[1]), newsynapses), ps)
end

"""
    (synapse::EPSP)(t::Integer; dt::Real = 1.0)

Evaluate an EPSP synapse. See [`Synapse.EPSP`](@ref).
"""
function (synapse::EPSP)(t::Integer; dt::Real = 1.0)
    _shiftspike!(synapse, t; dt = dt)

    return epsp(t * dt, synapse.lastspike, synapse.ϵ₀, synapse.τm, synapse.τs)
end
function evalsynapses(synapses::VecOrMat{T}, t::Integer, ps...; dt::Real = 1.0) where T<:EPSP
    map(s -> _shiftspike!(s, t; dt = dt), synapses)

    return epsp(t * dt, ps...)
end

end