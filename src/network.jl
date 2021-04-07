const PopOrInput = Union{Population, InputPopulation}

struct NetworkEdge{WT<:AbstractMatrix{<:Real}, ST<:AbstractArray{<:AbstractSynapse, 2}}
    weights::WT
    synapses::ST

    # cache
    synapse_currents::WT

    function NetworkEdge(weights::WT, synapses::ST) where {WT, ST}
        synapse_currents = similar(weights)

        synapse_mat = StructArray(synapses; unwrap = t -> t <: AbstractSynapse)
        synapse_mat = StructArrays.replace_storage(synapse_mat) do v
            if v isa Array{<:CircularArray}
                return ArrayOfCircularVectors{eltype(v[1])}(size(v), capacity(v[1]))
            else
                return v
            end
        end

        new{WT, typeof(synapse_mat)}(weights, synapse_mat, synapse_currents)
    end
end

struct Network <: AbstractDict{Symbol, PopOrInput}
    pops::Dict{Symbol, PopOrInput}
    fedgelist::Dict{Symbol, Vector{Symbol}}
    bedgelist::Dict{Symbol, Vector{Symbol}}
    connections::Dict{Tuple{Symbol, Symbol}, NetworkEdge}
end

"""
    size(net::Network)

Return the number of populations in a network.
"""
Base.length(net::Network) = length(net.pops)
Base.size(net::Network) = length(net)

Base.iterate(net::Network) = iterate(net.pops)
Base.getindex(net::Network, key) = net.pops[key]
Base.setindex!(net::Network, pop, key) = (net.pops[key] = pop)

Base.show(io::IO, net::Network) = print(io, "Network($(size(net)))")
function Base.show(io::IO, ::MIME"text/plain", net::Network)
    println(io, "Network($(size(net))):")
    for (name, pop) in net.pops
        println(io, "  $name => $pop")
    end
end

Network(pops::Dict = Dict{Symbol, PopOrInput}()) =
    Network(pops, Dict{Symbol, Vector{Symbol}}(),
                  Dict{Symbol, Vector{Symbol}}(),
                  Dict{Tuple{Symbol, Symbol}, NetworkEdge}())

function connect!(net::Network, src::Symbol, dst::Symbol;
                  weights::AbstractMatrix{<:Real}, synapse = Synapse.Delta)
    if !(haskey(net.pops, src) && haskey(net.pops, dst))
        error("Cannot find populations called $src and/or $dst in network.")
    end

    fedges = get!(net.fedgelist, src, Symbol[])
    push!(fedges, dst)
    bedges = get!(net.bedgelist, dst, Symbol[])
    push!(bedges, src)

    m = size(net.pops[src])
    n = size(net.pops[dst])
    synapses = [synapse() for i in 1:m, j in 1:n]

    net.connections[(src, dst)] = NetworkEdge(weights, synapses)
end

function _process_spikes!(net::Network, t::Integer, spikes; dt::Real = 1.0)
    for (pop, spikevec) in spikes
        dsts = get!(net.fedgelist, pop, Symbol[])
        for dst in dsts
            edge = net.connections[(pop, dst)]
            weights = edge.weights
            synapses = edge.synapses
            synapse_currents = edge.synapse_currents

            # excite synapses
            foreach((row, s) -> (s > 0) && excite!(row, s + 1), eachrow(synapses), spikevec)

            # compute current
            evaluate!(synapse_currents, synapses, t + 1; dt = dt)
            net.pops[dst].neuron_currents .= vec(sum(weights .* synapse_currents; dims = 1))
        end

        srcs = get!(net.bedgelist, pop, Symbol[])
        for src in srcs
            synapses = net.connections[(src, pop)].synapses
            neurons = view(net.pops[pop].neurons, :)

            # apply refactory period to synapses
            foreach((neuron, col, s) -> (s > 0) && refactor!(neuron, col, t; dt = dt),
                    neurons, eachcol(synapses), spikevec)
        end
    end
end

_resetnode!(node::Population) = reset!(node)
_resetnode!(::InputPopulation) = nothing

function update!(net::Network, t::Integer, spikes; dt::Real = 1.0)
    @inbounds for (name, pop) in net.pops
        (pop isa Population) && update!(pop, t, spikes[name]; dt = dt)
    end

    _record_spikes!(net, spikes; dt = dt)

    @inbounds for (_, edge) in net.connections
        update!(edge.learner, edge.weights, t; dt = dt)
    end
end

function evaluate!(spikes, net::Network, t::Integer; dt::Real = 1.0)
    @inbounds for (name, pop) in net.pops
        evaluate!(spikes[name], pop, t; dt = dt)
    end

    _process_spikes!(net, t, spikes; dt = dt)

    return spikes
end
evaluate!(net::Network, t; dt = 1.0) =
    evaluate!(Dict(name => zeros(Int, size(pop)) for (name, pop) in net.pops), net, t; dt = dt)
(net::Network)(t; dt = 1.0) = evaluate!(net, t; dt = dt)

# function step!(spikes, net::Network, learners, t; dt = 1.0)
#     evaluate!(spikes, net, t; dt = dt)

#     for (name, edge) in net.connections
#         update!(learners[name], edge.weights, spikes; dt = dt)
#     end
    
#     return spikes
# end
# function step!(net::Network, learners, t; dt = 1.0)
#     spikes = evaluate!(net, t; dt = dt)

#     for (name, edge) in net.connections
#         update!(learners[name], edge.weights, spikes; dt = dt)
#     end
    
#     return spikes
# end

function reset!(net::Network)
    _resetnode!.(values(net.pops))
    for (_, edge) in net.connections
        reset!(edge.synapses)
    end
end

# function simulate!(net::Network, T::Integer; dt::Real = 1.0, cb = () -> ())
#     spiketimes = Dict(name => )

#     for t = 1:T
#         cb()

#         spikes = net(t; dt = dt, dense = dense)

#         for (name, spikevec) in spikes
#              # record spikes
#             dict = get!(spiketimes, name, Dict([(i, Int[]) for i in 1:size(net.pops[name])]))
#             _recordspikes!(dict, spikevec)
#         end

#         update!(net, t, spikes; dt = dt)
#     end

#     return spiketimes
# end