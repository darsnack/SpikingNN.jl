"""
    Population{NT<:AbstractNeuron}

A population of neurons is a directed graph with metadata.

Parameterized Types:
- `NT<:AbstractNeuron`: the type of the neurons in the population

Fields:
- `graph::MetaDiGraph{NT}`: the connectivity graph of the population
- `events::Queue{Int}`: a FIFO of neuron indices indicating spike events

Node Metadata:
- `:class`: the class of the neuron (`:input`, `:output`, or `:none`)

Edge Metadata:
- `:response`: the (pre-)synaptic response function
"""
struct Population{NT<:AbstractNeuron}
    graph::MetaDiGraph{NT}
    events::Queue{Int}
end

"""
    Population{NT<:AbstractNeuron}(graph::SimpleDiGraph)

Create a population based on the connectivity graph, `graph`.
Optionally, specify the default synaptic response function.
"""
function Population{NT<:AbstractNeuron}(graph::SimpleDiGraph, ϵ::Function = delta)
    mgraph = MetaGraph(graph)
    for node in nodes(mgraph)
        set_prop!(mgraph, node, :class, :none)
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
    end

    Population{NT}(mgraph, Queue{Int}())
end

"""
    Population{NT<:AbstractNeuron}(graph::SimpleWeightedDiGraph)

Create a population based on the connectivity graph, `graph`,
with weights as specified.
Optionally, specify the default synaptic response function.
"""
function Population{NT<:AbstractNeuron}(graph::SimpleWeightedDiGraph, ϵ::Function = delta)
    mgraph = MetaGraph(graph)
    for node in nodes(mgraph)
        set_prop!(mgraph, node, :class, :none)
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
        set_prop!(mgraph, edge, :weight, graph[edge])
    end

    Population{NT}(mgraph, Queue{Int}())
end

"""
    neurons(pop::Population)

Return an array of neurons within the population.
"""
neurons(pop::Population) = nodes(pop.graph)

"""
    synapses(pop::Population)

Return an array of edges representing the synapses within the population.
"""
synapses(pop::Population) = collect(edges(pop.graph))

"""
    inputs(pop::Population)

Return the input neurons in a population.
"""
inputs(pop::Population) = filter_vertices(pop.graph, :class, :input)

"""
    outputs(pop::Population)

Return the indices of the output neurons in a population.
"""
outputs(pop::Population) = filter_vertices(pop.graph, :class, :output)

"""
    excite!(pop::Population, spikes::Array{Integer})

Excite the input neurons in a population.

Fields:
- `pop::Population`: the population to excite
- `spikes::Array{Integer}`: an array of spike times
"""
function excite!(pop::Population, spikes::Array{<:Integer})
    events = repeat(inputs(pop), length(spikes))
    enqueue!(pop.events, events)

    for neuron in inputs(pop)
        excite!(neuron, spikes)
    end
end

"""
    simulate!(pop::Population, dt::Real = 1.0)

Fields:
- `pop::Population`: the population to simulate
- `dt::Real`: the simulation time step
"""
function simulate!(pop::Population, dt::Real = 1.0)
    spike_times = Tuple{Int, Int}[]
    neurons = neurons(pop)
    while !isempty(pop.events)
        # process spike event
        neuron_id = dequeue!(pop.events)
        spike_time = step!(neurons[neuron_id], dt)
        (spike_time > 0) && push!(spike_times, (neuron_id, spike_time))

        # push spike onto downstream neurons
        if spike_time > 0
            for dest_id in outneighbors(pop.graph, neuron_id)
                enqueue!(pop.events, dest_id)
                enqueue!(pop[dest_id], (spike_time, pop.graph[dest_id, neuron_id]))
            end
        end
    end

    return spike_times
end