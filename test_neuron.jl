using SpikingNN
using UnicodePlots
using BenchmarkTools

##

neuron = Neuron(LIF(τm = 5f-3), Ideal(vth = 3f-1))
synapse = Delta()

##

dt = 1f-3
ts = ceil.(Int, (0:dt:0.2) ./ dt)
input = ConstantRate(freq = 1 / (5 * dt), dt = dt)

##

current = [0f0]
spike = [0]
voltage = Float32[]
output = Int[]
state, dstate = SpikingNN.init(neuron)

##

for t in ts
    global current, spike

    evaluate!(spike, input, t; dt = dt)
    excite!(synapse, spike)
    evaluate!(current, synapse, t; dt = dt)
    evaluate!(spike, dstate, state, neuron, t, current; dt = dt)
    refactor!(state, neuron, synapse, spike; dt = dt)

    push!(voltage, getvoltage(neuron, state)[1])
    (spike[1] > 0) && push!(output, t)
end

##

plt = lineplot(ts, voltage; width = 100, ylim = (0, 0.5))
scatterplot!(plt, output, fill(0.5, length(output)))
