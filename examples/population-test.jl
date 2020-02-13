using SpikingNN
using Plots

# simulation parameters
T = 1000

# create three SRM0 neurons
vᵣ = 2.0
τᵣ = 1.0
v_th = 1.0
neurons = [LIF(τᵣ, vᵣ, v_th) for i = 1:3]

# create population
# neuron 1, 2 excite neuron 3
# neuron 3 inhibits neuron 1, 2
connectivity_matrix = [ 0  0  1;
                        0  0  1;
                       -5 -5  0]
pop = Population(connectivity_matrix, neurons; ϵ = Synapse.Alpha)
# setclass(pop, 1, :input)
# setclass(pop, 2, :input)

# create input currents
low = ConstantRate(0.1)
high = ConstantRate(0.99)
switch(t; dt) = (t < Int(T/2)) ? low(t) : high(t)

# excite neurons
excite!(pop[1], low, T; response = Synapse.Alpha())
excite!(pop[2], switch, T; response = Synapse.Alpha())

# simulate
voltages = Dict([(i, Float64[]) for i in 1:3])
cb = function(id::Int, t::Int)
    (t > length(voltages[id])) && push!(voltages[id], pop[id].voltage)
end
@time outputs = simulate!(pop, T; cb = cb)

rasterplot(outputs, label = ["Input 1", "Input 2", "Inhibitor"])
title!("Raster Plot")
xlabel!("Time (sec)")