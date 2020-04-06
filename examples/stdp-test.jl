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
input = InputPopulation([ConstantRate(0.8)])

# create network
net = Network(Dict([:input => input, :pop => pop]))
connect!(net, :input, :pop; weights = [1 0], synapse = Synapse.Alpha)

# simulate
times = Int[]
w = Float64[]
voltages = Dict([(i, Float64[]) for i in 1:2])
cb = function(name::Symbol, id::Int, t::Int)
    if name == :pop
        if id == 1
            push!(times, t)
            push!(w, pop.weights[1, 2])
        end
        push!(voltages[id], getvoltage(pop[id].body))
    end
end
@time outputs = simulate!(net, T; cb = cb, dense = true)

weight_plot = plot(times, w, label = "")
title!("Synaptic Weights Over Simulation")
xlabel!("Time (sec)")
ylabel!("Weight")

raster_plot = rasterplot(outputs[:pop])
title!("Raster Plot")
xlabel!("Time (sec)")

plot(voltages[1], label = "Input")
voltage_plot = plot!(voltages[2], label = "Neuron")
title!("Membrane Voltages")
xlabel!("Time (sec)")
ylabel!("Voltage (V)")

plot(weight_plot, raster_plot, voltage_plot, layout = grid(3, 1))