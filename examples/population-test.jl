using SpikingNN
using Plots

# simulation parameters
T = 1000

# neuron parameters
vᵣ = 0
τᵣ = 1.0
vth = 1.0

# create population
# neuron 1, 2 excite neuron 3
# neuron 3 inhibits neuron 1, 2
weights = [ 0  0  1;
            0  0  1;
           -5 -5  0]
pop = Population(weights; cell = () -> LIF(τᵣ, vᵣ),
                          synapse = Synapse.Alpha,
                          threshold = () -> Threshold.Ideal(vth))

# create input currents
low = ConstantRate(0.1)
high = ConstantRate(0.99)
switch(t; dt = 1) = (t < Int(T/2)) ? low(t; dt = dt) : high(t; dt = dt)
n1synapse = Synapse.Alpha()
n2synapse = Synapse.Alpha()
excite!(n1synapse, filter(x -> x != 0, [low(t) for t = 1:T]))
excite!(n2synapse, filter(x -> x != 0, [switch(t) for t = 1:T]))

# simulate
voltages = Dict([(i, Float64[]) for i in 1:3])
cb = function(id::Int, t::Int)
    (t > length(voltages[id])) && push!(voltages[id], getvoltage(pop[id]))
end
@time outputs = simulate!(pop, T; cb = cb, inputs = [n1synapse, n2synapse, (t; dt) -> 0])

rasterplot(outputs, label = ["Input 1", "Input 2", "Inhibitor"])
title!("Raster Plot")
xlabel!("Time (sec)")