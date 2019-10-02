using SpikingNN
using Plots

# SRM0 params
η₀ = 5.0
τᵣ = 1.0
v_th = 2.0

# Input spike train params
rate = 0.01
T = 15
∂t = 0.01
n = convert(Int, ceil(T / ∂t))

srm = SRM0{Float64}(η₀, τᵣ, (Δ, v) -> SpikingNN.poisson(Δ, v; dt = ∂t))
spikes = constant_rate(rate, n)
excite!(srm, spikes; response = (t -> SpikingNN.delta(t; q = 2)), dt = ∂t)

# println("spike times:\n  $spikes")
println("# of spikes equal: $(length(spikes) == length(srm.spikes_in))")

# callback to record voltages
voltages = Float64[]
record = function ()
    push!(voltages, srm.voltage)
end

# simulate
@time output = simulate!(srm, ∂t; cb = record, dense = true)

# plot raster plot
scatter(∂t .* spikes, ones(length(spikes)), label = "Input")
raster_plot = scatter!(∂t .* output, 2*ones(length(output)), title = "Raster Plot (\\delta response)", xlabel = "Time (sec)", label = "Output")
xlims!(0, T)

# plot dense voltage recording
plot(∂t .* collect(0:maximum(spikes)), voltages,
    title = "SRM Membrane Potential with Varying Presynaptic Responses", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "\\delta response")

# resimulate using presynaptic response
reset!(srm)
voltages = Float64[]
excite!(srm, spikes; response = SpikingNN.α, dt = ∂t)
@time simulate!(srm, ∂t; cb = record, dense = true)

# plot voltages with response function
voltage_plot = plot!(∂t .* collect(0:maximum(spikes)), voltages, label = "\\alpha response")
xlims!(0, T)

plot(raster_plot, voltage_plot, layout = grid(2, 1))
xticks!(0:T)
;