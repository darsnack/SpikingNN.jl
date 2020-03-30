using SpikingNN
using Plots

# SRM0 params
η₀ = 5.0
τᵣ = 1.0
vth = 1.0

# Input spike train params
rate = 0.01
T = 15
∂t = 0.01
n = convert(Int, ceil(T / ∂t))

srm = Neuron(Synapse.Alpha(q = 2), SRM0(η₀, τᵣ), Threshold.Ideal(vth))
input = ConstantRate(rate)
spikes = excite!(srm, input, n)

# callback to record voltages
voltages = Float64[]
record = function ()
    push!(voltages, getvoltage(srm.body))
end

# simulate
@time output = simulate!(srm, n; dt = ∂t, cb = record, dense = true)

# plot raster plot
raster_plot = rasterplot(∂t .* spikes, ∂t .* output, label = ["Input", "Output"], xlabel = "Time (sec)",
                title = "Raster Plot (\\alpha response)")
xlims!(0, T)

# plot dense voltage recording
plot(∂t .* collect(0:n), voltages,
    title = "SRM Membrane Potential with Varying Presynaptic Responses", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "\\alpha response")

# resimulate using presynaptic response
voltages = Float64[]
srm = Neuron(Synapse.EPSP(ϵ₀ = 2.0, τm = 0.5, τs = 2.0), SRM0(η₀, τᵣ), Threshold.Ideal(vth))
excite!(srm, spikes)
@time simulate!(srm, n; dt = ∂t, cb = record, dense = true)

# plot voltages with response function
voltage_plot = plot!(∂t .* collect(0:n), voltages, label = "EPSP response")
xlims!(0, T)

plot(raster_plot, voltage_plot, layout = grid(2, 1))
xticks!(0:T)