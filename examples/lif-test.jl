using SpikingNN
using Plots

# LIF params
τm = 100
vreset = 0.0
vth = 0.1
R = 1.75

# Input spike train params
rate = 0.05
T = 1000

lif = Neuron(Synapse.Delta(), LIF(τm, vreset, R), Threshold.Ideal(vth))
input = ConstantRate(rate)
spikes = excite!(lif, input, T)

println("# of spikes equal: $(length(spikes) == length(lif.synapses[1].spikes))")

# callback to record voltages
voltages = Float64[]
record = let lif = lif
    () -> push!(voltages, getvoltage(lif.body))
end

# compile
simulate!(lif, T; cb = record)
reset!(lif.body)
voltages = Float64[]
excite!(lif, spikes)

# simulate
@time simulate!(lif, T; cb = record, dense = true)

# plot dense voltage recording
plot(collect(0:T), voltages,
    title = "LIF Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "Dense")

# repeat with dense simulation
reset!(lif.body)
voltages = Float64[]
excite!(lif, spikes)
@time output = simulate!(lif, T; cb = record)

# plot sparse voltage recording
voltage_plot = plot!([0; spikes], voltages,
    title = "LIF Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "Sparse (default)")

# plot raster plot
raster_plot = rasterplot(spikes, output, label = ["Input", "Output"], title = "Raster Plot", xlabel = "Time (sec)")

plot(raster_plot, voltage_plot, layout = grid(2, 1))