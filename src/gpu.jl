cpu(x) = x
gpu(x) = x

cpu(x::CuArray) = adapt(Array, x)
gpu(x::Array) = CuArrays.cu(x)
cpu(x::StructArray) = replace_storage(Array, x)
gpu(x::StructArray) = replace_storage(x) do v
    typeof(v) <: StructArray ? gpu(v) :
    isbitstype(eltype(v)) ? gpu(v) : v
end

cpu(x::Neuron) = Neuron(cpu(x.synapses), x.soma)
gpu(x::Neuron) = Neuron(gpu(x.synapses), x.soma)

function cpu(x::STDP{T}) where T<:Real
    clastpre = cpu(x.lastpre)
    clastpost = cpu(x.lastpost)

    return STDP{T, typeof(clastpre)}(x.A₊, x.A₋, x.τ₊, x.τ₋, clastpre, clastpost)
end
function gpu(x::STDP{T}) where T<:Real
    glastpre = gpu(x.lastpre)
    glastpost = gpu(x.lastpost)

    return STDP{T, typeof(glastpre)}(x.A₊, x.A₋, x.τ₊, x.τ₋, glastpre, glastpost)
end

function cpu(x::Population)
    csynapses = cpu(x.synapses)
    csomas = cpu(x.somas)
    cweights = cpu(x.weights)
    clearner = cpu(x.learner)

    return Population(csomas, cweights, csynapses, clearner)
end

function gpu(x::Population)
    gsynapses = gpu(x.synapses)
    gsomas = gpu(x.somas)
    gweights = gpu(x.weights)
    glearner = gpu(x.learner)

    return Population(gsomas, gweights, gsynapses, glearner)
end

function cpu(x::InputPopulation)
    cinputs = cpu(x.inputs)

    return InputPopulation{typeof(cinputs)}(cinputs)
end

function gpu(x::InputPopulation)
    ginputs = gpu(x.inputs)

    return InputPopulation{typeof(ginputs)}(ginputs)
end

function cpu(x::Network)
    y = deepcopy(x)

    @inbounds for (k, v) in y.pops
        y.pops[k] = cpu(v)
    end

    @inbounds for (k, v) in y.connections
        y.pops[k] = NetworkEdge(cpu(v.weights), cpu(v.synapses), cpu(v.learner))
    end

    return y
end

function gpu(x::Network)
    y = deepcopy(x)

    @inbounds for (k, v) in y.pops
        y.pops[k] = gpu(v)
    end

    @inbounds for (k, v) in y.connections
        y.connections[k] = NetworkEdge(gpu(v.weights), gpu(v.synapses), gpu(v.learner))
    end

    return y
end