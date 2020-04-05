using SpikingNN
using Plots

# simulation parameters
T = 1000

# create three SRM0 neurons
η₀ = 5.0
τᵣ = 1.0
vth = 1.0

# create population
weights = Float32[ 0  5;
                   0  0]
pop = Population(weights; cell = () -> SRM0(η₀, τᵣ),
                          synapse = Synapse.Alpha,
                          threshold = () -> Threshold.Ideal(vth),
                          learner = STDP(0.5, 0.5, size(weights, 1)))

# create step input currents
i = ConstantRate(0.8)
input = Synapse.Alpha()

# excite input neurons
Synapse.excite!(input, filter(x -> x != 0, [i(t) for t in 1:T]))

# simulate
times = Int[]
w = Float64[]
voltages = Dict([(i, Float64[]) for i in 1:2])
cb = function(id::Int, t::Int)
    push!(times, t)
    push!(w, pop.weights[1, 2])
    push!(voltages[id], getvoltage(pop[id].body))
end
@time outputs = simulate!(pop, T; cb = cb, dense = true, inputs = [input, (t; dt) -> 0])

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