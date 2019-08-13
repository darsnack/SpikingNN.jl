"""
    LIF

A leaky-integrate-fire neuron.

# Fields:
- `voltage::Real`: membrane potential
- `class::Symbol`: the class of the neuron (:input, :output, or :none)
- `spikes_in::Queue{Integer}`: a FIFO of input spike times
- `τ_m::Real`: membrane time constant
- `v_reset::Real`: reset voltage potential
- `v_th::Real`: threshold voltage potential
- `R::Real`: resistive constant (typically = 1)
"""
struct LIF <: AbstractNeuron
    # required fields
    voltage::Real
    class::Symbol
    spikes_in::Queue{Integer}

    # model specific fields
    τ_m::Real
    v_reset::Real
    v_th::Real
    R::Real
end

"""
    LIF(class, τ_m, v_reset, v_th, R = 1f0)

Create a LIF neuron with zero initial voltage and empty spike queue.
"""
LIF(class::Symbol, τ_m::T, v_reset::T, v_th::T, R::T = 1f0) where T <: Real
    = LIF(0f0, class, Queue{Int}(), R, τ_m, v_reset, v_th)