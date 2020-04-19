cpu(x) = x
gpu(x) = x

cpu(x::CuArray) = adapt(Array, x)
gpu(x::Array) = CuArrays.cu(x)
cpu(x::StructArray) = replace_storage(Array, s)
gpu(x::StructArray) = replace_storage(s) do v
    isbitstype(eltype(v)) : gpu(v) : v
end

cpu(x::Neuron) = Neuron(cpu(x.synapses), x.body, x.threshold)
gpu(x::Neuron) = Neuron(gpu(x.synapses), x.body, x.threshold)

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
    cbody = cpu(x.neurons.body)
    cthresholds = cpu(x.neurons.threshold)
    cweights = cpu(x.weights)
    clearner = cpu(x.learner)

    viewsynapses = [view(csynapses, :, i) for i in eachcol(csynapses)]
    nt = (synapses = viewsynapses, body = cbody, threshold = cthresholds)
    gneurons = StructArray{Neuron{eltype(viewsynapses), eltype(cbody), eltype(cthresholds)}}(nt)

    return Population(cneurons, cweights, csynapses, clearner)
end

function gpu(x::Population)
    gsynapses = gpu(x.synapses)
    gbody = gpu(x.neurons.body)
    gthresholds = gpu(x.neurons.threshold)
    gweights = gpu(x.weights)
    glearner = gpu(x.learner)

    viewsynapses = [view(gsynapses, :, i) for i in eachcol(gsynapses)]
    nt = (synapses = viewsynapses, body = gbody, threshold = gthresholds)
    gneurons = StructArray{Neuron{eltype(viewsynapses), eltype(gbody), eltype(gthresholds)}}()

    return Population(gneurons, gweights, gsynapses, glearner)
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
        y.pops[k] = NetworkEdge(gpu(v.weights), gpu(v.synapses), gpu(v.learner))
    end

    return y
end