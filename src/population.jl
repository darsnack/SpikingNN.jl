"""
    Population{NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner} <: AbstractArray{Int, 1}

A population of neurons is a directed graph with weights and response functions on each edge.

Fields:
- `graph::SimpleDiGraph`: the connectivity graph of the population
- `neurons::Array{NT}`: an array of the neurons in the population
- `learner::AbstractLearner`: a learning mechanism (see `AbstractLearner`)

Node Metadata:
- `:class`: the class of the neuron (`:input`, `:output`, or `:none`)

Edge Metadata:
- `:response`: the (pre-)synaptic response function
"""
struct Population{T<:Neuron, NT<:AbstractArray{T, 1}, WT<:AbstractMatrix{<:Real}, ST<:AbstractArray{<:AbstractSynapse, 2}, LT<:AbstractLearner} <: AbstractArray{T, 1}
    neurons::NT
    weights::WT
    synapses::ST
    learner::LT
end

"""
    size(pop::Population)

Return the number of neurons in a population.
"""
Base.size(pop::Population) = length(pop.neurons)

Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.getindex(pop::Population, i::Int) = pop.neurons[i]
function Base.setindex!(pop::Population, neuron::Neuron, i::Int)
    pop.synapses[:, i] = neuron.synapses
    pop.neurons[i] = Neuron(view(pop.synapses, :, i), neuron.body, neuron.threshold)
end

Base.show(io::IO, pop::Population{T, <:Any, <:Any, ST, LT}) where {T, ST, LT} =
    print(io, "Population{$(nameof(eltype(pop.neurons.body))), $(nameof(eltype(ST))), $(nameof(LT))}($(size(pop)))")
Base.show(io::IO, ::MIME"text/plain", pop::Population) = show(io, pop)

function Population(weights::AbstractMatrix{<:Real}; cell = LIF, synapse = Synapse.Delta, threshold = Threshold.Ideal, learner = George())
    n = _checkweights(weights)
    synapses = StructArray([synapse() for i in 1:n, j in 1:n])
    neurons = StructArray([Neuron(view(synapses, :, i), cell(), threshold()) for i in 1:n]; unwrap = t -> t <: AbstractCell || t <: AbstractThreshold)

    Population(neurons, weights, synapses, learner)
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

function _processspikes!(pop::Population, spikes::AbstractVector{<:Integer}; dt::Real = 1.0)
    # record spikes with learner
    record!(pop.learner, pop.weights, spikes; dt = dt)

    @inbounds for (i, spike) in enumerate(spikes)
        if spike > 0
            # record spikes with neurons
            spike!(view(pop.neurons.body, i), spike; dt = dt)

            # excite post-synaptic neurons
            for j in 1:size(pop)
                (pop.weights[i, j] != 0) && excite!(pop.synapses[i, j], spike)
            end
        end
    end
end

_filteractive(pop::Population, neuronids, t::Integer) =
    filter(id -> isactive(pop[id], t), neuronids)

"""
    (::Population)(t::Integer; dt::Real = 1.0, dense = false)

Evaluate a population of neurons at time step `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (pop::Population)(t::Integer; dt::Real = 1.0, dense = false, inputs = nothing)
    spikes = zeros(Int, size(pop))

    # evalute inputs
    !isnothing(inputs) && excite!(view(pop.neurons.body, :), [input(t; dt = dt) for input in inputs])

    # filter inactive neurons for sparsity
    ids = 1:size(pop)
    evalids = dense ? ids : _filteractive(pop, collect(ids), t)

    # evaluate synapses and excite neuron bodies w/ current
    current = vec(reduce(+, pop.weights[:, evalids] .* evalsynapses(pop.synapses[:, evalids], t; dt = dt); dims = 1))
    excite!(view(pop.neurons.body, evalids), current)

    # evaluate neuron bodies
    voltage = evalcells(view(pop.neurons.body, evalids), t; dt = dt)

    # evaluate thresholds
    spikes[evalids] .= evalthresholds(view(pop.neurons.threshold, evalids), t, voltage; dt = dt)

    # process spike events
    _processspikes!(pop, spikes; dt = dt)

    return spikes
end

"""
    update!(pop::Population)

Update synaptic weights within population according to learner.
"""
update!(pop::Population, t::Integer; dt::Real = 1.0) = update!(pop.learner, pop.weights, t; dt = dt)

function reset!(pop::Population)
    reset!(pop.synapses)
    reset!(pop.neurons.body)
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
function simulate!(pop::Population, T::Integer; dt::Real = 1.0, cb = (id::Int, t::Integer) -> (), dense = false, inputs = nothing)
    spiketimes = Dict([(i, Int[]) for i in 1:size(pop)])

    for t = 1:T
        # evaluate population once
        spikes = pop(t; dt = dt, dense = dense, inputs = inputs)

        # record spike time
        _recordspikes!(spiketimes, spikes)

        # update weights
        update!(pop, t; dt = dt)

        # evaluate callback
        cb.(1:size(pop), t)
    end

    return spiketimes
end