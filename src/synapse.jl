"""
    delta(Δ::Real; q::Real = 1)

Return 1.0 whenever Δ == 0 and 0.0 otherwise. (i.e. Dirac delta function)
"""
delta(Δ::Real; q::Real = 1.0) = (Δ == 0) ? q : zero(q)

"""
    α(Δ::Real; q::Real = 1, τ::Real = 1)

Return (q / τ) * exp(-Δ / τ) Θ(Δ) (where Θ is the Heaviside function).
"""
function α(Δ::Real; q::Real = 1.0, τ::Real = 1.0)
    v = Δ * (q / τ) * exp(-(Δ - τ) / τ)
    return (Δ >= 0) ? v : zero(v)
end

# """
#     Synapse{NT<:AbstractNeuron, F} <: AbstractEdge{NT}

# Create a synapse between two neurons.

# Fields:
# - `src::NT`: pre-synaptic neuron
# - `dst::NT`: post-synaptic neuron
# - `weight::Float64`: the synaptic weight
# - `response::F`: the (pre-)synaptic response function (e.g. `delta`)
# """
# struct Synapse{NT<:AbstractNeuron, F} <: AbstractEdge{NT}
#     src::NT
#     dst::NT
#     weight::Float64
#     response::F
# end

# Synapse(src::NT, dst::NT; weight = 1.0, response = delta) where {NT<:AbstractNeuron} = Synapse{NT, typeof(response)}(src, dst, weight, response)
# Synapse(t::Tuple{NT, NT}; weight = 1.0, response = delta) where {NT<:AbstractNeuron} = Synapse{NT, typeof(response)}(t[1], t[2], 1.0, response)
# Synapse(p::Pair; weight = 1.0, response = delta) = Synapse(p[1], p[2], weight, response)

# eltype(e::Synapse{NT, Any}) where {NT<:AbstractNeuron} = NT

# src(e::Synapse) = e.src
# dst(e::Synapse) = e.dst
# pre(e::Synapse) = src(e)
# post(e::Synapse) = dst(e)

# Base.show(io::IO, ::MIME"text/plain", e::Synapse) =
#     print(io, "Synapse with weight = $(e.weight) and response = $(e.response)")
# Base.show(io::IO, e::Synapse) = print(io, "Synapse(w = $(e.weight)")

# Pair(e::Synapse) = Pair(src(e), dst(e))
# Tuple(e::Synapse) = (src(e), dst(e))

# reverse(e::Synapse) = Synapse(dst(e), src(e), e.weight, e.response)