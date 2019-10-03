"""
    Population{NT<:AbstractNeuron} <: AbstractArray{Int, 1}

A population of neurons is a directed graph with metadata.

Parameterized Types:
- `NT<:AbstractNeuron`: the type of the neurons in the population

Fields:
- `graph::MetaDiGraph`: the connectivity graph of the population
- `events::PriorityQueue{Int, Integer}`: a queue of neuron indices indicating spike events
    with priority given by spike time
- `neurons::Array{NT}`: an array of the neurons in the population

Node Metadata:
- `:class`: the class of the neuron (`:input`, `:output`, or `:none`)

Edge Metadata:
- `:response`: the (pre-)synaptic response function
"""
struct Population{IT<:Integer, NT<:AbstractNeuron{<:Any, IT}} <: AbstractArray{Int, 1}
    graph::MetaDiGraph
    events::PriorityQueue{Int, IT}
    neurons::Array{NT, 1}
end

"""
    size(pop::Population)

Return the number of neurons in a population.
"""
Base.size(pop::Population) = length(pop.neurons)

Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.getindex(pop::Population, i::Int) = pop.neurons[i]
Base.setindex!(pop::Population{NT}, neuron::NT, i::Int) where {NT<:AbstractNeuron} =
    (pop.neurons[i] = neuron)

Base.show(io::IO, ::MIME"text/plain", pop::Population{NT}) where {NT<:AbstractNeuron} =
    print(io, "Population{$(nameof(NT))}($(size(pop))) with $(length(pop.events)) queued events:\n    $(collect(edges(pop.graph)))")
Base.show(io::IO, pop::Population{NT}) where {NT<:AbstractNeuron} =
    print(io, "Population{$(nameof(NT))}($(size(pop)))")

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
Optionally, specify the default synaptic response function.
"""
function Population(graph::SimpleDiGraph, neurons::Array{NT}; ϵ::Function = delta) where {IT<:Integer, NT<:AbstractNeuron{<:Any, IT}}
    mgraph = MetaDiGraph(graph)
    for vertex in vertices(mgraph)
        set_prop!(mgraph, vertex, :class, :none)
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
    end

    Population{IT, NT}(mgraph, PriorityQueue{Int, IT}(), neurons)
end

"""
    Population(weights::Array{Real, 2}, neurons::Array{NT<:AbstractNeuron})

Create a population based on the connectivity matrix, `weights`.
Optionally, specify the default synaptic response function.
"""
function Population(weights::Array{<:Real, 2}, neurons::Array{NT}; ϵ::Function = delta) where {IT<:Integer, NT<:AbstractNeuron{<:Any, IT}}
    if size(weights, 1) != size(weights, 2)
        error("Connectivity matrix of population must be a square.")
    end

    mgraph = MetaDiGraph(SimpleDiGraph(abs.(weights)))
    for vertex in vertices(mgraph)
        set_prop!(mgraph, vertex, :class, :none)
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
        set_prop!(mgraph, edge, :weight, weights[src(edge), dst(edge)])
    end

    Population{IT, NT}(mgraph, PriorityQueue{Int, IT}(), neurons)
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
findinputs(pop::Population) = filter_vertices(pop.graph, :class, :input)

"""
    inputs(pop::Population)

Return the input neurons in a population.
"""
inputs(pop::Population) = pop.neurons[collect(findinputs(pop))]

"""
    findoutputs(pop::Population)

Return iterator over the output neurons in a population.
"""
findoutputs(pop::Population) = filter_vertices(pop.graph, :class, :output)

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

"""
    excite!(pop::Population, neuron_ids::Array{Integer}, spikes::Array{Integer})

Excite the neurons in a population.

Fields:
- `pop::Population`: the population to excite
- `neuron_ids::Array{Integer}`: an array of neuron indices that should be excited
- `spikes::Array{Integer}`: an array of spike times
- `response::Function`: a response function applied to each spike
- `dt::Real`: the sample rate for the response function
"""
function excite!(pop::Population, neuron_ids::Array{<:Integer}, spikes::Array{<:Integer}; response = delta, dt::Real = 1.0)
    nrns = neurons(pop)
    for id in neuron_ids
        excite!(nrns[id], spikes; response = response, dt = dt)
        min_t = minimum(keys(nrns[id].spikes_in))
        if haskey(pop.events, id)
            pop.events[id] = min(min_t, pop.events[id])
        else
            enqueue!(pop.events, id => min_t)
        end
    end
end

"""
    simulate!(pop::Population, dt::Real = 1.0)

Simulate a population of neurons. Optionally specify a learner. The `prespike` and
`postspike` functions will be called immediately after either event occurs.

Fields:
- `pop::Population`: the population to simulate
- `dt::Real`: the simulation time step
- `learner::AbstractLearner`: a learning mechanism (see `AbstractLearner`)
- `cb::Function`: a callback function that is called after event evaluation (expects `(neuron_id, t)` as input)
- `dense::Bool`: set to `true` to evaluate every time step even in the absence of events
- `learner::AbstractLearner`: a learning mechanism
"""
function simulate!(pop::Population{IT, <:Any}, dt::Real = 1.0;
                   cb = (id::Int, t::IT) -> (), dense = false, learner::AbstractLearner = George()) where {IT<:Integer}
    spike_times = Dict([(i, IT[]) for i in 1:size(pop)])

    # for dense evaluation, add spikes with zero current to the queue
    max_t = 0
    if dense
        max_t = maximum([isempty(x.spikes_in) ? zero(keytype(x.spikes_in)) :
                                                maximum(keys(x.spikes_in)) for x in neurons(pop)])
        for neuron in neurons(pop)
            for t in setdiff(1:max_t, keys(neuron.spikes_in))
                inc!(neuron.spikes_in, t, 0)
            end
        end

        empty!(pop.events)
        for id in 1:size(pop)
            enqueue!(pop.events, id, 1)
        end
    end

    while !isempty(pop.events)
        # process spike event
        neuron_id, t = dequeue_pair!(pop.events)
        spike_time = step!(pop[neuron_id], dt)

        # push spike onto downstream neurons
        if spike_time > 0
            # record spike time
            record = get!(spike_times, neuron_id, IT[])
            push!(record, spike_time)

            # call postsynaptic spike functions for upstream neurons
            for src_id in inneighbors(pop.graph, neuron_id)
                w = get_prop(pop.graph, src_id, neuron_id, :weight)
                Δw = postspike(learner, w, dt * spike_time, src_id, neuron_id)
                set_prop!(pop.graph, src_id, neuron_id, :weight, w + Δw)
            end

            # process downstream neurons
            for dest_id in outneighbors(pop.graph, neuron_id)
                # call presynaptic spike function for downstream neuron
                w = get_prop(pop.graph, neuron_id, dest_id, :weight)
                Δw = prespike(learner, w, dt * spike_time, neuron_id, dest_id)
                set_prop!(pop.graph, neuron_id, dest_id, :weight, w + Δw)

                # process response function
                response = get_prop(pop.graph, neuron_id, dest_id, :response)
                h, N = sample_response(response, dt)
                currents = weights(pop.graph)[neuron_id, dest_id] .* h
                for (tt, current) in enumerate(currents)
                    inc!(pop[dest_id].spikes_in, spike_time + tt - 1, current)
                end
                min_t = minimum(keys(pop[dest_id].spikes_in))
                if haskey(pop.events, dest_id)
                    pop.events[dest_id] = min(min_t, pop.events[dest_id])
                else
                    enqueue!(pop.events, dest_id => min_t)
                end
            end
        elseif dense && t < max_t
            # add dummy current to downstream neurons
            for dest_id in outneighbors(pop.graph, neuron_id)
                inc!(pop[dest_id].spikes_in, t + 1, 0)
                min_t = minimum(keys(pop[dest_id].spikes_in))
                if haskey(pop.events, dest_id)
                    pop.events[dest_id] = min(min_t, pop.events[dest_id])
                else
                    enqueue!(pop.events, dest_id => min_t)
                end
            end
        end

        # evaluate callback
        cb(neuron_id, t)

        # add neuron back into queue if it still needs to process
        !isempty(pop[neuron_id].spikes_in) && enqueue!(pop.events, neuron_id, minimum(keys(pop[neuron_id].spikes_in)))
    end

    return spike_times
end