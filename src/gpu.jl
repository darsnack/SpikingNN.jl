cpu(x) = adapt(Array, x)
gpu(x) = adapt(CuArray, x)

cpu(x::AbstractArray{<:AbstractRNG}) = x
gpu(x::AbstractArray{<:AbstractRNG}) = x

# cpu(x::CuArray) = adapt(Array, x)
# gpu(x::AbstractArray) = adapt(CuArray, x)
cpu(x::StructArray) = replace_storage(x) do v
    typeof(v) <: StructArray ? cpu(v) :
    isbitstype(eltype(v)) ? cpu(v) : v
end
gpu(x::StructArray) = replace_storage(x) do v
    typeof(v) <: StructArray ? gpu(v) :
    isbitstype(eltype(v)) ? gpu(v) : v
end

cpu(x::Neuron) = Neuron(cpu(x.body), cpu(x.threshold))
gpu(x::Neuron) = Neuron(gpu(x.body), gpu(x.threshold))

function cpu(x::STDP{T}) where T<:Real
    clastpre = cpu(x.lastpre)
    clastpost = cpu(x.lastpost)

    return STDP{T, typeof(clastpre)}(x.Ap, x.An, x.τp, x.τn, clastpre, clastpost)
end
function gpu(x::STDP{T}) where T<:Real
    glastpre = gpu(x.lastpre)
    glastpost = gpu(x.lastpost)

    return STDP{T, typeof(glastpre)}(x.Ap, x.An, x.τp, x.τn, glastpre, glastpost)
end

function cpu(x::Population)
    csynapses = cpu(x.synapses)
    cneurons = cpu(x.neurons)
    cweights = cpu(x.weights)

    return Population(cneurons, cweights, csynapses)
end

function gpu(x::Population)
    gsynapses = gpu(x.synapses)
    gneurons = gpu(x.neurons)
    gweights = gpu(x.weights)

    return Population(gneurons, gweights, gsynapses)
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
        y.connections[k] = NetworkEdge(cpu(v.weights), cpu(v.synapses))
    end

    return y
end

function gpu(x::Network)
    y = deepcopy(x)

    @inbounds for (k, v) in y.pops
        y.pops[k] = gpu(v)
    end

    @inbounds for (k, v) in y.connections
        y.connections[k] = NetworkEdge(gpu(v.weights), gpu(v.synapses))
    end

    return y
end