"""
    Population{T<:Soma,
               NT<:AbstractArray{T, 1},
               WT<:AbstractMatrix{<:Real},
               ST<:AbstractArray{<:AbstractSynapse, 2},
               LT<:AbstractLearner} <: AbstractArray{T, 1}

A population of neurons is an array of [`Soma`](@ref)s,
  a weighted matrix of [synapses](@ref Synapse Models), and a [learner](@ref Learning).

Fields:
- `somas::AbstractArray{<:Soma, 1}`: a vector of somas
- `weights::AbstractMatrix{<:Real}`: a weight matrix
- `synapses::AbstractArray{<:AbstractSynapse, 2}`: a matrix of synapses
- `learner::AbstractLearner`: a learning mechanism
"""
struct Population{T<:Soma, NT<:AbstractArray{T, 1}, WT<:AbstractMatrix{<:Real}, ST<:AbstractArray{<:AbstractSynapse, 2}, LT<:AbstractLearner} <: AbstractArray{T, 1}
    somas::NT
    weights::WT
    synapses::ST
    learner::LT
end

"""
    size(pop::Population)

Return the number of neurons in a population.
"""
Base.size(pop::Population) = length(pop.somas)

Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.getindex(pop::Population, i::Int) = Neuron(pop.synapses[:, i], pop.somas[i])
function Base.setindex!(pop::Population, neuron::Neuron, i::Int)
    pop.synapses[:, i] = neuron.synapses
    pop.somas[i] = neuron.soma
end

Base.show(io::IO, pop::Population{T, <:Any, <:Any, ST, LT}) where {T, ST, LT} =
    print(io, "Population{$(nameof(eltype(pop.somas.body))), $(nameof(eltype(ST))), $(nameof(LT))}($(size(pop)))")
Base.show(io::IO, ::MIME"text/plain", pop::Population) = show(io, pop)

_instantiate(x::AbstractArray, I...) = x[I...]
_instantiate(x, I...) = x()

"""
    Population(weights::AbstractMatrix{<:Real};
               cell = LIF, synapse = Synapse.Delta,
               threshold = Threshold.Ideal, learner = George())
Create a population by specifying the `weights`
  and optionally the cell type, synapse type, threshold type, and learner.

# Keyword Fields:
- `cell::AbstractCell`: a constructor or function that creates a cell body, or a vector of pre-constructed cells
- `synapse::AbstractSynapse`: a constructor or function that creates a synapse, or a matrix of pre-constructed synapses
- `threshold::AbstractThreshold`: a constructor or function that creates a threshold, or a vector of pre-constructed thresholds
- `learner::AbstractLearner`: a learner object
"""
function Population(weights::AbstractMatrix{<:Real}; cell = LIF, synapse = Synapse.Delta, threshold = Threshold.Ideal, learner = George())
    n = _checkweights(weights)
    synapses = StructArray(_instantiate(synapse, i, j) for i in 1:n, j in 1:n)
    somas = StructArray(Soma(_instantiate(cell, i), _instantiate(threshold, i)) for i in 1:n;
                        unwrap = t -> t <: AbstractCell || t <: AbstractThreshold)

    Population(somas, weights, synapses, learner)
end

"""
    neurons(pop::Population)

Return an array of neurons within the population.
"""
neurons(pop::Population) = pop[1:end]

"""
    synapses(pop::Population)

Return an array of edges representing the synapses within the population.
"""
synapses(pop::Population) = pop.synapses

function _processspikes!(pop::Population, spikes; dt::Real = 1.0)
    # record spikes with learner
    record!(pop.learner, pop.weights, spikes; dt = dt)

    # excite post-synaptic neurons
    map((row, s) -> (s > 0) && excite!(row, s + 1), eachrow(pop.synapses), spikes)

    # apply refactory period to synapses
    map((col, s) -> (s > 0) && spike!(col, s), eachcol(pop.synapses), spikes)
end

"""
    evaluate!(pop::Population, t::Integer; dt::Real = 1.0, dense = false, inputs = nothing)
    (::Population)(t::Integer; dt::Real = 1.0, dense = false)

Evaluate a population of neurons at time step `t`.
Return a vector of time stamps (`t` if the neuron spiked and zero otherwise).
"""
function evaluate!(pop::Population, t::Integer; dt::Real = 1.0, dense = false, inputs = nothing)
    # evalute inputs
    !isnothing(inputs) && excite!(pop.somas, [input(t; dt = dt) for input in inputs])

    # evaluate synapses and excite neuron bodies w/ current
    current = vec(reduce(+, pop.weights .* evaluate!(pop.synapses, t; dt = dt); dims = 1))
    excite!(pop.somas, current)

    # evaluate somas
    spikes = evaluate!(pop.somas, t; dt = dt)

    # process spike events
    _processspikes!(pop, spikes; dt = dt)

    return spikes
end
(pop::Population)(t::Integer; kwargs...) = evaluate!(pop, t; kwargs...)

"""
    update!(pop::Population, t::Integer; dt::Real = 1.0)

Update synaptic weights within population according to `pop.learner`.
"""
update!(pop::Population, t::Integer; dt::Real = 1.0) = update!(pop.learner, pop.weights, t; dt = dt)

"""
    reset!(pop::Population)

Reset `pop.synapses` and `pop.somas`.
"""
function reset!(pop::Population)
    reset!(pop.synapses)
    reset!(pop.somas)
end

function _recordspikes!(dict::Dict{Int, Array{Int, 1}}, spikes)
    for (id, spiketime) in enumerate(spikes)
        if spiketime > 0
            record = get!(dict, id, Int[])
            push!(record, spiketime)
        end
    end
end

"""
    simulate!(pop::Population, dt::Real = 1.0)

Simulate a population of neurons. Optionally specify a learner. The `prespike` and
`postspike` functions will be called immediately after either event occurs.

Fields:
- `pop::Population`: the population to simulate
- `T::Integer`: number of time steps to simulate
- `dt::Real`: the simulation time step
- `cb::Function`: a callback function that is called after event evaluation (expects `(neuron_id, t)` as input)
- `dense::Bool`: set to `true` to evaluate every time step even in the absence of events
"""
function simulate!(pop::Population, T::Integer; dt::Real = 1.0, cb = () -> (), dense = false, inputs = nothing)
    spiketimes = Dict([(i, Int[]) for i in 1:size(pop)])

    for t = 1:T
        # evaluate callback
        cb()

        # evaluate population once
        spikes = evaluate!(pop, t; dt = dt, dense = dense, inputs = inputs)

        # record spike time
        _recordspikes!(spiketimes, spikes)

        # update weights
        update!(pop, t; dt = dt)
    end

    return spiketimes
end