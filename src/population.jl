using .Synapse: AbstractSynapse

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
struct Population{NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner} <: AbstractArray{Int, 1}
    graph::SimpleDiGraph
    neurons::Array{NT, 1}
    weights::Array{Float64, 2}
    responses::Dict{Tuple{Int, Int}, AbstractSynapse}
    learner::LT
end

"""
    size(pop::Population)

Return the number of neurons in a population.
"""
Base.size(pop::Population) = length(pop.neurons)

Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.getindex(pop::Population, i::Int) = pop.neurons[i]
Base.setindex!(pop::Population{NT}, neuron::NT, i::Int) where {NT<:Union{AbstractInput, AbstractNeuron}} =
    (pop.neurons[i] = neuron)

Base.show(io::IO, pop::Population{NT, LT}) where {NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner} =
    print(io, "Population{$(nameof(NT)), $(nameof(LT))}($(size(pop)))")

"""
    add_synapse(pop::Population, source::Integer, destination::Integer, weight::Real)

Add a synapse between `source` and `destination` neurons with `weight`.
"""
function add_synapse(pop::Population, source::Integer, destination::Integer, weight::Real; ϵ::AbstractSynapse = Synapse.Delta())
    add_edge!(pop.graph, source, destination)
    pop.weights[source, destination] = weight
    pop.responses[(source, destination)] = ϵ
end

"""
    Population(graph::SimpleDiGraph, neurons::Array{NT<:AbstractNeuron})

Create a population based on the connectivity graph, `graph`.
Optionally, specify the default synaptic response function or learner.

**Note:** the default response function assumes a simulation time step of 1 second.
"""
function Population(graph::SimpleDiGraph, neurons::Vector{NT};
                    ϵ = Synapse.Delta, learner::LT = George()) where {NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner}
    weights = zeros(length(neurons), length(neurons))
    responses = Dict([((src(e), dst(e)), ϵ()) for e in edges(graph)])

    Population{NT, LT}(graph, neurons, weights, responses, learner)
end

"""
    Population(weights::Array{Real, 2}, neurons::Array{NT<:AbstractNeuron})

Create a population based on the connectivity matrix, `weights`.
Optionally, specify the default synaptic response or learner.

**Note:** the default response function assumes a simulation time step of 1 second.
"""
function Population(weights::Array{<:Real, 2}, neurons::Vector{NT};
                    ϵ = Synapse.Delta, learner::LT = George()) where {NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner}
    if size(weights, 1) != size(weights, 2)
        error("Connectivity matrix of population must be a square.")
    end

    graph = SimpleDiGraph(abs.(weights))
    responses = Dict([((src(e), dst(e)), ϵ()) for e in edges(graph)])

    Population{NT, LT}(graph, neurons, weights, responses, learner)
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
synapses(pop::Population) = collect(edges(pop.graph))

"""
    findinputs(pop::Population)

Return a vector of the input neuron indices in a population.
"""
findinputs(pop::Population) = filter!(x -> isempty(inneighbors(pop.graph, x)), collect(vertices(pop.graph)))

"""
    inputs(pop::Population)

Return the input neurons in a population.
"""
inputs(pop::Population) = pop.neurons[findinputs(pop)]

"""
    findoutputs(pop::Population)

Return a vector of the output neuron indices in a population.
"""
findoutputs(pop::Population) = filter!(x -> isempty(outneighbors(pop.graph, x)), collect(vertices(pop.graph)))

"""
    outputs(pop::Population)

Return the output neurons in a population.
"""
outputs(pop::Population) = pop.neurons[findoutputs(pop)]

"""
    isdone(pop::Population)

Returns true if all neurons in the population have no current events left to process.
"""
isdone(pop::Population) = all(isdone, neurons(pop))

function _processspike!(pop::Population, neuron_id::Integer, spike_time::Integer; dt::Real = 1.0)
    # call postsynaptic spike functions for upstream neurons
    for src_id in inneighbors(pop.graph, neuron_id)
        w = pop.weights[src_id, neuron_id]
        if w != 0
            postspike!(pop.learner, w, spike_time, src_id, neuron_id; dt = dt)
        else
            # remove edges that have zero weight
            rem_edge!(pop.graph, src_id, neuron_id)
        end
    end

    # process downstream neurons
    ws = pop.weights[neuron_id, :] # make matrix access not across columns
    for dest_id in outneighbors(pop.graph, neuron_id)
        w = ws[dest_id]

        if w != 0
            # call presynaptic spike function for downstream neuron
            prespike!(pop.learner, w, spike_time, neuron_id, dest_id; dt = dt)

            # process response function
            response = pop.responses[(neuron_id, dest_id)]
            excite!(pop[dest_id], spike_time; response = response, dt = dt, weight = w)
        else
            # remove edges that have zero weight
            rem_edge!(pop.graph, neuron_id, dest_id)
        end
    end
end

_filteractive(pop::Population, neuron_ids::Array{<:Integer}, t::Integer) =
    filter(id -> isactive(pop[id], t), neuron_ids)

_parents(pop::Population, neuron_ids::Vector{<:Integer}) =
    reduce(vcat, [outneighbors(pop.graph, id) for id in neuron_ids])

"""
    (::Population)(t::Integer; dt::Real = 1.0, dense = false)

Evaluate a population of neurons at time step `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (pop::Population)(t::Integer; dt::Real = 1.0, dense = false)
    spikes = zeros(Int, size(pop))

    # evaluate neurons layer by layer starting at input neurons
    ids = findinputs(pop)
    ids = isempty(ids) ? collect(1:size(pop)) : ids # in case graph is cyclic
    processed = Int[]
    while !isempty(ids)
        processed = vcat(processed, ids)

        # filter inactive neurons for sparsity
        evalids = dense ? ids : _filteractive(pop, ids, t)

        # evaluate layer
        for id in evalids
            spike_time = pop[id](t; dt = dt)
            if spike_time > 0
                _processspike!(pop, id, spike_time; dt = dt)
                spikes[id] = t
            end
        end

        # update ids to parents of layer
        ids = setdiff(_parents(pop, ids), processed)
    end

    return spikes
end

"""
    update!(pop::Population)

Update synaptic weights within population according to learner.
"""
update!(pop::Population, t::Integer; dt::Real = 1.0) = update!(pop.learner, t, pop.weights; dt = dt)
update!(pop::Population{NT, George}, t::Integer; dt::Real = 1.0) where {NT<:Union{AbstractInput, AbstractNeuron}} = return

function _recordspikes!(dict::Dict{Int, Array{Int, 1}}, spikes::Array{Int, 1})
    for (id, spike_time) in enumerate(spikes)
        if spike_time > 0
            record = get!(dict, id, Int[])
            push!(record, spike_time)
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
function simulate!(pop::Population, T::Integer; dt::Real = 1.0, cb = (id::Int, t::Integer) -> (), dense = false)
    spike_times = Dict([(i, Int[]) for i in 1:size(pop)])

    for t = 1:T
        # evaluate population once
        ids = _filteractive(pop, collect(1:size(pop)), t)
        spikes = pop(t; dt = dt, dense = dense)

        # record spike time
        _recordspikes!(spike_times, spikes)

        # update weights
        update!(pop, t; dt = dt)

        # evaluate callback
        cb.(ids, t)
    end

    return spike_times
end