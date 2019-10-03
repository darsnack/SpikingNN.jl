using SpikingNN
using Plots
using MetaGraphs

# simulation parameters
T = 1000

# create three SRM0 neurons
η₀ = 5.0
τᵣ = 1.0
v_th = 1.0
neurons = [SRM0(η₀, τᵣ, v_th) for i = 1:2]

# create population
connectivity_matrix = [ 0  5;
                        0  0]
pop = Population(connectivity_matrix, neurons; ϵ = SpikingNN.α)
setclass(pop, 1, :input)

# create step input currents
i = constant_rate(0.8, T)

# excite input neurons
excite!(pop, collect(findinputs(pop)), i; response = SpikingNN.α)

# simulate
times = Int[]
w = Float64[]
voltages = Dict([(i, Float64[]) for i in 1:2])
cb = function(id::Int, t::Int)
    push!(times, t)
    push!(w, get_prop(pop.graph, 1, 2, :weight))
    (t > length(voltages[id])) && push!(voltages[id], pop[id].voltage)
end
@time outputs = simulate!(pop; cb = cb, dense = true, learner = STDP(0.5, 0.5, size(pop)))

weight_plot = plot(times, w, label = "")
title!("Synaptic Weights Over Simulation")
xlabel!("Time (sec)")
ylabel!("Weight")

raster_plot = rasterplot(outputs)
title!("Raster Plot")
xlabel!("Time (sec)")

plot(voltages[1], label = "Input")
voltage_plot = plot!(voltages[2], label = "Neuron")
title!("Membrane Voltages")
xlabel!("Time (sec)")
ylabel!("Voltage (V)")

plot(weight_plot, raster_plot, voltage_plot, layout = grid(3, 1))