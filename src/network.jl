const PopOrInput = Union{Population, InputPopulation}

struct NetworkEdge{WT<:AbstractMatrix{<:Real}, ST<:AbstractArray{<:AbstractSynapse, 2}, LT<:AbstractLearner}
    weights::WT
    synapses::ST
    learner::LT
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

Network(pops::Dict) = Network(pops, Dict(), Dict(), Dict())
Network() = Network(Dict())

function connect!(net::Network, src::Symbol, dst::Symbol; weights::AbstractMatrix{<:Real}, synapse = Synapse.Delta, learner = George())
    !(haskey(net.pops, src) && haskey(net.pops, dst)) && error("Cannot find populations called $src and/or $dst in network.")
    fedges = get!(net.fedgelist, src, Symbol[])
    push!(fedges, dst)
    bedges = get!(net.bedgelist, dst, Symbol[])
    push!(bedges, src)

    m = size(net.pops[src])
    n = size(net.pops[dst])
    synapses = StructArray(synapse() for i in 1:m, j in 1:n; unwrap = t -> t <: AbstractSynapse)

    net.connections[(src, dst)] = NetworkEdge(weights, synapses, learner)
end

function _processspikes!(net::Network, spikes, t::Integer; dt::Real = 1.0)
    for (pop, spikevec) in spikes
        dsts = get!(net.fedgelist, pop, Symbol[])
        for dst in dsts
            edge = net.connections[(pop, dst)]
            weights = edge.weights
            synapses = edge.synapses

            # excite synapses
            map((row, s) -> (s > 0) && excite!(row, s + 1), eachrow(synapses), spikevec)

            # compute current
            current = vec(reduce(+, weights .* evaluate!(synapses, t + 1; dt = dt); dims = 1))
            excite!(net.pops[dst].somas, current)

            # record pre-synaptic spikes
            prespike!(edge.learner, weights, spikevec; dt = dt)
        end

        srcs = get!(net.bedgelist, pop, Symbol[])
        for src in srcs
            # record post-synaptic spikes
            edge = net.connections[(src, pop)]
            postspike!(edge.learner, edge.weights, spikevec; dt = dt)

            # apply refactory period to synapses
            map((col, s) -> (s > 0) && spike!(col, s; dt = dt), eachcol(edge.synapses), spikevec)
        end
    end
end

_evalnode(node::Population, t; dt, dense) = node(t; dt = dt, dense = dense)
_evalnode(node::InputPopulation, t; dt, dense) = node(t; dt = dt)

_resetnode!(node::Population) = reset!(node)
_resetnode!(node::InputPopulation) = nothing

function update!(net::Network, t::Integer; dt::Real = 1.0)
    @inbounds for pop in values(net.pops)
        (pop isa Population) && update!(pop, t; dt = dt)
    end

    @inbounds for (_, edge) in net.connections
        update!(edge.learner, edge.weights, t; dt = dt)
    end
end

function (net::Network)(t::Integer; dt::Real = 1.0, dense = false)
    spikes = Dict{Symbol, Union{Vector, CuVector}}()

    @inbounds for (name, pop) in net.pops
        spikes[name] = _evalnode(pop, t; dt = dt, dense = dense)
    end

    _processspikes!(net, spikes, t; dt = dt)

    return spikes
end

function reset!(net::Network)
    _resetnode!.(values(net.pops))
    for (_, edge) in net.connections
        reset!(edge.synapses)
    end
end

function simulate!(net::Network, T::Integer; dt::Real = 1.0, cb = () -> (), dense = false)
    spiketimes = Dict{Symbol, Dict}()

    for t = 1:T
        cb()

        spikes = net(t; dt = dt, dense = dense)

        for (name, spikevec) in spikes
             # record spikes
            dict = get!(spiketimes, name, Dict([(i, Int[]) for i in 1:size(net.pops[name])]))
            _recordspikes!(dict, spikevec)
        end

        update!(net, t; dt = dt)
    end

    return spiketimes
end