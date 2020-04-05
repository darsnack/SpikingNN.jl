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
    print(io, "Population{$(nameof(T)), $(nameof(eltype(ST))), $(nameof(LT))}($(size(pop)))")
Base.show(io::IO, ::MIME"text/plain", pop::Population) = show(io, pop)

_checkweights(matrix) = (size(matrix, 1) == size(matrix, 2)) ? size(matrix, 1) : error("Connectivity matrix of population must be a square.")

# """
#     add_synapse(pop::Population, source::Integer, destination::Integer, weight::Real)

# Add a synapse between `source` and `destination` neurons with `weight`.
# """
# function add_synapse(pop::Population, source::Integer, destination::Integer, weight::Real; ϵ::AbstractSynapse = Synapse.Delta())
#     add_edge!(pop.graph, source, destination)
#     pop.weights[source, destination] = weight
#     pop.responses[(source, destination)] = ϵ
# end

# """
#     Population(weights::Array{Real, 2}, neurons::Array{NT<:AbstractNeuron})

# Create a population based on the connectivity matrix, `weights`.
# Optionally, specify the default synaptic response or learner.

# **Note:** the default response function assumes a simulation time step of 1 second.
# """
# function Population(weights::Array{<:Real, 2}, neurons::Vector{NT};
#                     ϵ = Synapse.Delta, learner::LT = George()) where {NT<:Union{AbstractInput, AbstractNeuron}, LT<:AbstractLearner}
#     if size(weights, 1) != size(weights, 2)
#         error("Connectivity matrix of population must be a square.")
#     end

#     graph = SimpleDiGraph(abs.(weights))
#     responses = Dict([((src(e), dst(e)), ϵ()) for e in edges(graph)])

#     Population{NT, LT}(graph, neurons, weights, responses, learner)
# end

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

# """
#     findinputs(pop::Population)

# Return a vector of the input neuron indices in a population.
# """
# findinputs(pop::Population) = filter!(x -> isempty(inneighbors(pop.graph, x)), collect(vertices(pop.graph)))

# """
#     inputs(pop::Population)

# Return the input neurons in a population.
# """
# inputs(pop::Population) = pop.neurons[findinputs(pop)]

# """
#     findoutputs(pop::Population)

# Return a vector of the output neuron indices in a population.
# """
# findoutputs(pop::Population) = filter!(x -> isempty(outneighbors(pop.graph, x)), collect(vertices(pop.graph)))

# """
#     outputs(pop::Population)

# Return the output neurons in a population.
# """
# outputs(pop::Population) = pop.neurons[findoutputs(pop)]

# """
#     isdone(pop::Population)

# Returns true if all neurons in the population have no current events left to process.
# """
# isdone(pop::Population) = all(isdone, neurons(pop))

function _processspikes!(pop::Population, spikes::AbstractVector{<:Integer}; dt::Real = 1.0)
    # record spikes with learner
    record!(pop.learner, pop.weights, spikes; dt = dt)

    @inbounds for (i, spike) in enumerate(spikes)
        # record spikes with neurons
        (spike > 0) && spike!(view(pop.neurons.body, i), spike; dt = dt)

        # excite post-synaptic neurons
        for j in 1:size(pop)
            (pop.weights[i, j] != 0) && Synapse.excite!(pop.synapses[i, j], spike)
        end
    end
end

_filteractive(pop::Population, neuronids, t::Integer) =
    filter(id -> isactive(pop[id], t), neuronids)

# _parents(pop::Population, neuron_ids::Vector{<:Integer}) =
#     reduce(vcat, [outneighbors(pop.graph, id) for id in neuron_ids])

"""
    (::Population)(t::Integer; dt::Real = 1.0, dense = false)

Evaluate a population of neurons at time step `t`.
Return time stamp if the neuron spiked and zero otherwise.
"""
function (pop::Population)(t::Integer; dt::Real = 1.0, dense = false, inputs = [])
    spikes = zeros(Int, size(pop))

    # evalute inputs
    excite!(view(pop.neurons.body, :), [input(t; dt = dt) for input in inputs])

    # filter inactive neurons for sparsity
    ids = collect(1:size(pop))
    evalids = dense ? ids : _filteractive(pop, ids, t)

    # evaluate synapses and excite neuron bodies w/ current
    current = vec(reduce(+, pop.weights[:, evalids] .* Synapse.evalsynapses(view(pop.synapses, :, evalids), t; dt = dt); dims = 1))
    excite!(view(pop.neurons.body, evalids), current)

    # evaluate neuron bodies
    voltage = evalcells(view(pop.neurons.body, evalids), t; dt = dt)

    # evaluate thresholds
    spikes[evalids] .= Threshold.evalthresholds(view(pop.neurons.threshold, evalids), t, voltage; dt = dt)

    # process spike events
    _processspikes!(pop, spikes; dt = dt)

    return spikes
end

"""
    update!(pop::Population)

Update synaptic weights within population according to learner.
"""
update!(pop::Population, t::Integer; dt::Real = 1.0) = update!(pop.learner, pop.weights, t; dt = dt)
update!(pop::Population{<:Any, <:Any, <:Any, <:Any, George}, t::Integer; dt::Real = 1.0) = return

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
function simulate!(pop::Population, T::Integer; dt::Real = 1.0, cb = (id::Int, t::Integer) -> (), dense = false, inputs = [])
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