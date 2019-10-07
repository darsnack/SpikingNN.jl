"""
    Network

A graph to describe connectivity between populations.
An edge from Pop A to Pop B signifies all-to-all connections between
output neurons from Pop A to input neurons from Pop B.

Fields:
- `graph::MetaDiGraph`: a connectivity graph of populations
- `pops::Array{PT, 1}`: populations in the graph

Edge Metadata:
- `:response`: the (pre-)synaptic response function
"""
struct Network{IT<:Integer, PT<:Population{IT, NT}} <: AbstractArray{Int, 1} where NT <: AbstractNeuron{<:Any, IT}
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
Base.setindex!(net::Network{<:Integer, PT}, pop::PT, i::Int) where {NT<:AbstractNeuron, PT<:Population{<:Integer, NT}}} =
    (net.pops[i] = pop)

Base.show(io::IO, ::MIME"text/plain", net::Network) = print(io, "Network($(size(net)))")
Base.show(io::IO, net::Network) = print(io, "Network($(size(net)))")

function done(net::Network) = all(p -> isempty(p.events), net.pops)

function simulate!(net::Network{IT, <:Any}, dt::Real = 1.0;
                   cb = (id::Int, t::IT) -> (), dense = false, learner::AbstractLearner = George()) where {IT<:Integer}
    max_ts = dense ? densify.(net.pops) : zeros(size(net))

    while !done(net)
        # process a step! of every population in the network
        for (i, pop) in enumerate(net.pops)
            if !isempty(pop.events)
                # step! the most immediate neural event
                t, neuron_id, spike_time = step!(pop, dt; dense = dense, learner = learner, max_t = max_ts[i])

                # process outputs
                for dest_pop in outneighbors(pop)
                    for dest_id in findinputs(dest_pop)
                        # process response function
                        response = get_prop(net.graph, i, dest_pop, :response)
                        w = weights(net.graph)[neuron_id, dest_id]
                        wtrespose = (response |> (x -> w * x))
                        excite!(net[dest_pop][dest_id], spike_time; response = wtrespose, dt = dt)
                        min_t = minimum(keys(net[dest_pop][dest_id].spikes_in))
                        if haskey(net[dest_pop].events, dest_id)
                            net[dest_pop].events[dest_id] = min(min_t, net[dest_pop].events[dest_id])
                        else
                            enqueue!(net[dest_pop].events, dest_id => min_t)
                        end
                    end
                end
            end
        end
    end
end