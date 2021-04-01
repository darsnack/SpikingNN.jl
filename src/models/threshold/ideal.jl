"""
    Ideal(vth::Real)

An ideal threshold spikes when `v > vth`.
"""
struct Ideal{T<:Real} <: AbstractThreshold
    vth::T
end

isactive(threshold::Ideal, t::Integer; dt::Real = 1.0) = false

"""
    evaluate!(threshold::Ideal, t::Real, v::Real; dt::Real = 1.0)
    (threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0)
    evaluate!(thresholds::AbstractArray{<:Ideal}, t::Integer, v; dt::Real = 1.0)
    evaluate!(spikes, thresholds::AbstractArray{<:Ideal}, t::Integer, v; dt::Real = 1.0)

Return `t` when `v > threshold.vth` (optionally writing the result to `spikes`).
"""
evaluate!(threshold::Ideal, t::Real, v::Real; dt::Real = 1.0) = (v >= threshold.vth) ? t : zero(t)
(threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0) = evaluate!(threshold, t, v; dt = dt)
evaluate!(thresholds::T, t::I, v; dt::Real = 1.0) where {T<:AbstractArray{<:Ideal}, I<:Integer} =
    I.((v .>= thresholds.vth) .* adapt(typeof(v), fill(t, size(v))))
function evaluate!(spikes, thresholds::AbstractArray{<:Ideal}, t::Integer, v; dt::Real = 1.0)
    spikes .= I.((v .>= thresholds.vth) .* t)

    return spikes
end
