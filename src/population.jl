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
struct Population{T<:Neuron, S<:AbstractSynapse, W<:AbstractMatrix{R}} <: AbstractVector{T}
    neurons::NT
    weights::WT
    synapses::ST
end

"""
    size(pop::Population)

Return the number of neurons in a population.
"""
Base.size(pop::Population) = length(pop.neurons)

Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.getindex(pop::Population, i::Int) = pop.neurons[i]
function Base.setindex!(pop::Population, neuron::Neuron, i::Int)
    pop.neurons[i] = neuron
end

Base.show(io::IO, pop::Population{T, <:Any, <:Any, ST, LT}) where {T, ST, LT} =
    print(io, "Population{$(nameof(eltype(pop.neurons.body))), $(nameof(eltype(ST)))}($(size(pop)))")
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
function Population(weights::AbstractMatrix{<:Real};
                    cell = LIF, synapse = Delta, threshold = Ideal)
    n = _checksquare(weights)
    synapses = [_instantiate(synapse, i, j) for i in 1:n, j in 1:n]
    neurons = [Neuron(_instantiate(cell, i), _instantiate(threshold, i)) for i in 1:n]

    Population(neurons, weights, synapses)
end

"""
    neurons(pop::Population)

Return an array of neurons within the population.
"""
neurons(pop::Population) = pop.neurons

"""
    synapses(pop::Population)

Return an array of edges representing the synapses within the population.
"""
synapses(pop::Population) = pop.synapses

init(pop::Population) = init(pop.neurons)

function excite!(pop::Population, current)
    pop.neuron_currents .+= current

    return pop
end

"""
    evaluate!(pop::Population, t::Integer; dt::Real = 1.0, dense = false, inputs = nothing)
    (::Population)(t::Integer; dt::Real = 1.0, dense = false)

Evaluate a population of neurons at time step `t`.
Return a vector of time stamps (`t` if the neuron spiked and zero otherwise).
"""
function evaluate!(spikes, dstate, state, pop::Population, t; dt = 1)
    # evaluate synapses
    evaluate!(pop.synapse_currents, pop.synapses, t; dt = dt)
    excite!(pop, vec(sum(pop.weights .* pop.synapse_currents; dims = 1)))

    # evaluate neurons
    evaluate!(spikes, dstate, state, pop.neurons, t, pop.neuron_currents; dt = dt)

    cpu_spikes = adapt(Array, spikes)
    # excite post-synaptic neurons
    foreach((row, s) -> (s > 0) && excite!(row, s + 1), eachrow(pop.synapses), cpu_spikes)

    # apply refactory period to synapses
    refactor!(state, pop.neurons, pop.synapses, spikes; dt = dt)

    # reset current cache
    pop.neuron_currents .= 0

    return spikes
end
evaluate!(dstate, state, pop::Population, t; dt = 1) =
    evaluate!(similar(pop.weights, Int, size(pop)), dstate, state, pop, t; dt = dt)

function step!(spikes, dstate, state, pop::Population, learner::AbstractLearner, t; dt = 1)
    evaluate!(spikes, dstate, state, pop, t; dt = dt)
    update!(learner, pop.weights, t, spikes, spikes; dt = dt)

    return spikes
end

"""
    reset!(pop::Population)

Reset `pop.synapses` and `pop.somas`.
"""
function reset!(pop::Population)
    reset!(pop.synapses)
    reset!(state, pop.neurons)
    pop.neuron_currents .= 0
    pop.synapse_currents .= 0
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
function simulate!(pop::Population, learner::AbstractLearner, T::Integer; dt = 1, cb = () -> ())
    spikes = similar(pop.weights, Int, size(pop), T)
    state = init(pop)
    dstate = fill!(similar(state), 0)

    for t = 1:T
        # advance population with learner
        step!(view(spikes, :, t), dstate, state, pop, learner, t; dt = dt)

        # evaluate callback
        cb()
    end

    return spikes
end
function simulate!(pop::Population, T::Integer; dt = 1, cb = () -> ())
    spikes = similar(pop.weights, Int, size(pop), T)
    state = init(pop)
    dstate = fill!(similar(state), 0)

    for t in 1:T
        # advance population
        evaluate!(view(spikes, :, t), dstate, state, pop, t; dt = dt)

        # evaluate callback
        cb()
    end

    return spikes
end
