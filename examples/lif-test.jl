using SpikingNN
using Plots

# LIF params
τ_m = 100
v_reset = 0.0
v_th = 0.1
R = 1.75

# Input spike train params
rate = 0.05
T = 1000

lif = LIF(τ_m, v_reset, v_th, R)
input = ConstantRate(rate)
spikes = excite!(lif, input, T)

println("# of spikes equal: $(length(spikes) == length(lif.current_in))")

# callback to record voltages
voltages = Float64[]
record = function ()
    push!(voltages, lif.voltage)
end

# simulate
@time output = simulate!(lif; cb = record)

# plot raster plot
raster_plot = rasterplot(spikes, output, label = ["Input", "Output"], title = "Raster Plot", xlabel = "Time (sec)")

# # plot sparse voltage recording
plot(collect(0:maximum(spikes)), voltages,
    title = "LIF Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "Sparse (default)")

# repeat with dense simulation
reset!(lif)
voltages = Float64[]
excite!(lif, spikes)
@time simulate!(lif; cb = record, dense = true)

# plot dense voltage recording
voltage_plot = plot!(collect(0:maximum(spikes)), voltages,
    title = "LIF Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "Dense")

plot(raster_plot, voltage_plot, layout = grid(2, 1))