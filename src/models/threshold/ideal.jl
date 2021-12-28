"""
    Ideal(vth, dims = (1,))
    Ideal(; vth, dims = (1,))

An ideal threshold spikes when `v >= vth`.
"""
struct Ideal{T<:AbstractArray{<:Real}} <: AbstractThreshold
    vth::T
end

Ideal(vth, dims = (1,)) = Ideal(_fillmemaybe(vth, dims))
Ideal(; vth, dims = (1,)) = Ideal(vth, dims)

Base.size(threshold::Ideal) = size(threshold.vth)

"""
    evaluate!(spikes, thresholds::Ideal, t, voltages; dt = 1)

Return `t` when `v > threshold.vth` (optionally writing the result to `spikes`).
"""
function evaluate!(spikes, threshold::Ideal, t, voltage; dt = 1)
    @. spikes = ifelse((voltage >= threshold.vth), t, zero(t))

    return spikes
end
