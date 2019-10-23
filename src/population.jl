using .Synapse: AbstractSynapse

"""
    Population{NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner} <: AbstractArray{Int, 1}

A population of neurons is a directed graph with metadata.

Fields:
- `graph::MetaDiGraph`: the connectivity graph of the population
- `neurons::Array{NT}`: an array of the neurons in the population
- `learner::AbstractLearner`: a learning mechanism (see `AbstractLearner`)

Node Metadata:
- `:class`: the class of the neuron (`:input`, `:output`, or `:none`)

Edge Metadata:
- `:response`: the (pre-)synaptic response function
"""
struct Population{NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner} <: AbstractArray{Int, 1}
    graph::MetaDiGraph
    neurons::Array{NT, 1}
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
function add_synapse(pop::Population, source::Integer, destination::Integer, weight::Real)
    add_edge!(pop.graph, source, destination)
    set_prop!(pop.graph, source, destination, :weight, weight)
end

"""
    Population(graph::SimpleDiGraph, neurons::Array{NT<:AbstractNeuron})

Create a population based on the connectivity graph, `graph`.
Optionally, specify the default synaptic response function or learner.

**Note:** the default response function assumes a simulation time step of 1 second.
"""
function Population(graph::SimpleDiGraph, neurons::Vector{NT};
                    ϵ::AbstractSynapse = Synapse.Delta(), learner::LT = George()) where {NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner}
    mgraph = MetaDiGraph{Int, Float64}(graph, 0)
    for vertex in vertices(mgraph)
        if isa(neurons[vertex], AbstractInput)
            set_prop!(mgraph, vertex, :class, :input)
        else
            set_prop!(mgraph, vertex, :class, :none)
        end
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
    end

    Population{NT, LT}(mgraph, neurons, learner)
end

"""
    Population(weights::Array{Real, 2}, neurons::Array{NT<:AbstractNeuron})

Create a population based on the connectivity matrix, `weights`.
Optionally, specify the default synaptic response or learner.

**Note:** the default response function assumes a simulation time step of 1 second.
"""
function Population(weights::Array{<:Real, 2}, neurons::Vector{NT};
                    ϵ::AbstractSynapse = Synapse.Delta(), learner::LT = George()) where {NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner}
    if size(weights, 1) != size(weights, 2)
        error("Connectivity matrix of population must be a square.")
    end

    mgraph = MetaDiGraph{Int, Float64}(SimpleDiGraph(abs.(weights)), 0)
    for vertex in vertices(mgraph)
        if isa(neurons[vertex], AbstractInput)
            set_prop!(mgraph, vertex, :class, :input)
        else
            set_prop!(mgraph, vertex, :class, :none)
        end
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
        set_prop!(mgraph, edge, :weight, weights[src(edge), dst(edge)])
    end

    Population{NT, LT}(mgraph, neurons, learner)
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

Return an iterator over the input neurons in a population.
"""
findinputs(pop::Population) = collect(filter_vertices(pop.graph, :class, :input))

"""
    inputs(pop::Population)

Return the input neurons in a population.
"""
inputs(pop::Population) = pop.neurons[collect(findinputs(pop))]

"""
    findoutputs(pop::Population)

Return iterator over the output neurons in a population.
"""
findoutputs(pop::Population) = collect(filter_vertices(pop.graph, :class, :output))

"""
    outputs(pop::Population)

Return the output neurons in a population.
"""
outputs(pop::Population) = pop.neurons[collect(findoutputs(pop))]

"""
    setclass(pop::Population, i::Integer, class::Symbol)

Set neuron `i` as an input neuron in `pop`.
"""
function setclass(pop::Population, i::Integer, class::Symbol)
    if !(class ∈ [:input, :output, :none])
        error("Neuron can only be a class of :input, :output, or :none.")
    end
    set_prop!(pop.graph, i, :class, class)
end

# """
#     excite!(pop::Population, neuron_ids::Array{Integer}, spikes::Array{Integer})

# Excite the neurons in a population.

# Fields:
# - `pop::Population`: the population to excite
# - `neuron_ids::Array{Integer}`: an array of neuron indices that should be excited
# - `spikes::Array{Integer}`: an array of spike times
# - `response::Function`: a response function applied to each spike
# - `dt::Real`: the sample rate for the response function
# """
# function excite!(pop::Population, neuron_ids::Array{<:Integer}, spikes::Array{<:Integer}; response = delta, dt::Real = 1.0)
#     nrns = neurons(pop)
#     for id in neuron_ids
#         excite!(nrns[id], spikes; response = response, dt = dt)
#     end
# end

"""
    isdone(pop::Population)

Returns true if all neurons in the population have no current events left to process.
"""
isdone(pop::Population) = all(isdone, neurons(pop))

function _processspike!(pop::Population, neuron_id::Integer, spike_time::Integer; dt::Real = 1.0)
    # call postsynaptic spike functions for upstream neurons
    for src_id in inneighbors(pop.graph, neuron_id)
        w = get_prop(pop.graph, src_id, neuron_id, :weight)
        postspike!(pop.learner, w, spike_time, src_id, neuron_id; dt = dt)
    end

    # process downstream neurons
    for dest_id in outneighbors(pop.graph, neuron_id)
        # call presynaptic spike function for downstream neuron
        w = get_prop(pop.graph, neuron_id, dest_id, :weight)
        prespike!(pop.learner, w, spike_time, neuron_id, dest_id; dt = dt)

        # process response function
        response = get_prop(pop.graph, neuron_id, dest_id, :response)
        w = weights(pop.graph)[neuron_id, dest_id]
        excite!(pop[dest_id], spike_time; response = response, dt = dt, weight = w)
    end
end

_filteractive(pop::Population, neuron_ids::Array{<:Integer}, t::Integer) =
    filter(id -> isactive(pop[id], t), neuron_ids)

"""
    (::Population)(t::Integer; dt::Real = 1.0, dense = false)

Evaluate a population of neurons at time step `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (pop::Population)(t::Integer; dt::Real = 1.0, dense = false)
    spikes = zeros(Int, size(pop))

    # evaluate inputs first
    ids = dense ? findinputs(pop) : _filteractive(pop, findinputs(pop), t)
    for id in ids
        spike_time = pop[id](t; dt = dt)
        if spike_time > 0
            _processspike!(pop, id, spike_time; dt = dt)
            spikes[id] = t
        end
    end

    # evaluate remaining neurons
    ids = setdiff(1:size(pop), findinputs(pop))
    ids = dense ? ids : _filteractive(pop, ids, t)
    for id in ids
        spike_time = pop[id](t; dt = dt)
        if spike_time > 0
            _processspike!(pop, id, spike_time; dt = dt)
            spikes[id] = t
        end
    end

    return spikes
end

"""
    update!(pop::Population)

Update synaptic weights within population according to learner.
"""
function update!(pop::Population)
    for src_id in 1:size(pop), dest_id in 1:size(pop)
        Δw = update!(pop.learner, src_id, dest_id)
        if Δw != 0
            w = get_prop(pop.graph, src_id, dest_id, :weight)
            set_prop!(pop.graph, src_id, dest_id, :weight, w + Δw)
        end
    end
end
update!(pop::Population{NT, George}) where {NT<:Union{AbstractInput, AbstractNeuron}} = return

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
        update!(pop)

        # evaluate callback
        cb.(ids, t)
    end

    return spike_times
end