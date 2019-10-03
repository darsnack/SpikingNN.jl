using SpikingNN
using Plots

# simulation parameters
T = 1000

# create three SRM0 neurons
η₀ = 5.0
τᵣ = 1.0
v_th = 1.0
neurons = [SRM0(η₀, τᵣ, v_th) for i = 1:3]

# create population
# neuron 1, 2 excite neuron 3
# neuron 3 inhibits neuron 1, 2
connectivity_matrix = [ 0  0  1;
                        0  0  1;
                       -5 -5  0]
pop = Population(connectivity_matrix, neurons; ϵ = SpikingNN.α)
setclass(pop, 1, :input)
setclass(pop, 2, :input)

# create step input currents
i₁ = [constant_rate(0.1, Int(T/2)); constant_rate(0.1, Int(T/2)) .+ Int(T/2)]
i₂ = [constant_rate(0.1, Int(T/2)); constant_rate(0.99, Int(T/2)) .+ Int(T/2)]

# excite input neurons
excite!(pop, [1], i₁; response = SpikingNN.α)
excite!(pop, [2], i₂; response = SpikingNN.α)

# simulate
# voltages = Dict([(i, Float64[]) for i in 1:3])
# cb = function(id::Int, t::Int)
#     (t > length(voltages[id])) && push!(voltages[id], pop[id].voltage)
# end
@time outputs = simulate!(pop)

rasterplot(outputs, label = ["Input 1", "Input 2", "Inhibitor"])
title!("Raster Plot")
xlabel!("Time (sec)")