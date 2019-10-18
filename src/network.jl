"""
    Network

A graph to describe connectivity between populations.
An edge from Pop A to Pop B signifies all-to-all connections between
output neurons from Pop A to input neurons from Pop B.

Fields:
- `graph::MetaDiGraph`: a connectivity graph of populations
- `pops::Array{PT, 1}`: populations in the graph

Node Metadata:
- `:name`: a name given to each population

Edge Metadata:
- `:response`: the (pre-)synaptic response function
"""
struct Network{PT<:Population} <: AbstractArray{Int, 1}
    graph::MetaDiGraph
    pops::Array{PT, 1}
end

"""
    size(net::Network)

Return the number of populations in a network.
"""
Base.size(net::Network) = length(net.pops)

Base.IndexStyle(::Type{<:Network}) = IndexLinear()
Base.getindex(net::Network, i::Int) = net.pops[i]
Base.setindex!(net::Network{PT}, pop::PT, i::Int) where {PT<:Population} = (net.pops[i] = pop)

Base.show(io::IO, net::Network) = print(io, "Network($(size(net)))")

"""
    Network(graph::SimpleDiGraph, pops::Array{PT<:Population}, names::Array{Symbol})

Create a network of populations (each labeled according to `names`)
based on the connectivity graph, `graph`.
Optionally, specify the default synaptic response function.

**Note:** the default response function assumes a simulation time step of 1 second.
"""
function Network(graph::SimpleDiGraph, pops::Array{PT}, names::Array{Symbol};
                 ϵ::AbstractSynapse = Synapse.Delta()) where {PT<:Population}
    mgraph = MetaDiGraph(graph)
    for vertex in vertices(mgraph)
        set_prop!(mgraph, vertex, :name, names[vertex])
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
    end

    Network{PT}(mgraph, pops)
end

"""
    Network(weights::Array{Real, 2}, pops::Array{PT<:Population}, names::Array{Symbol})

Create a network of populations (each labeled according to `names`)
based on the connectivity matrix, `weights`.
Optionally, specify the default synaptic response function.

**Note:** the default response function assumes a simulation time step of 1 second.
"""
function Network(weights::Array{<:Real, 2}, pops::Array{PT}, names::Array{Symbol};
                 ϵ::AbstractSynapse = Synapse.Delta()) where {PT<:Population}
    if size(weights, 1) != size(weights, 2)
        error("Connectivity matrix of network must be a square.")
    end

    mgraph = MetaDiGraph(SimpleDiGraph(abs.(weights)))
    for vertex in vertices(mgraph)
        set_prop!(mgraph, vertex, :name, names[vertex])
    end
    for edge in edges(mgraph)
        set_prop!(mgraph, edge, :response, ϵ)
        set_prop!(mgraph, edge, :weight, weights[src(edge), dst(edge)])
    end

    Network{PT}(mgraph, pops)
end

"""
    isdone(net::Network)

Returns true if all populations within the network are done.
"""
isdone(net::Network) = all(isdone, net.pops)

function _processspikes!(net::Network, pop_id::Integer, spikes::Array{Integer, 1}, t::Integer; dt::Real = 1.0)
    # process outputs
    for dest_pop in outneighbors(net.graph, pop_id)
        for dest_id in findinputs(pops[dest_pop])
            # process response function
            response = get_prop(net.graph, i, dest_pop, :response)
            w = weights(net.graph)[pop_id, dest_pop]

            # excite according to outputs from pop_id
            for src_id in findoutputs(pops[pop_id])
                if spikes[src_id] > 0
                    excite!(net[dest_pop][dest_id], t; response = response, dt = dt, weight = w)
                end
            end
        end
    end
end

function simulate!(net::Network, T::Integer; dt::Real = 1.0, cb = (name::Symbol, id::Int, t::IT) -> (), dense = false)
    spike_times = Dict{Symbol, Dict}()

    for t = 1:T
        for (i, pop) in enumerate(net.pops)
            # get population name
            name = get_prop(pop.graph, i, :name)

            # process one time step evaluation of population
            spikes = pop(t; dt = dt, dense = dense)
            _processspikes!(net, i, spikes, t; dt = dt)

            # record spikes
            dict = get!(spike_times, name, Dict([(i, Int[]) for i in 1:size(pop)]))
            _recordspikes!(dict, spikes)

            # update weights
            update!(pop)

            # evaluate callback
            cb.(name, ids, t)
        end
    end

    return spike_times
end