using SpikingNN
using Plots

# LIF params
η₀ = 2.0
τᵣ = 1.0
v_th = 1.0

# Input spike train params
rate = 0.01
T = 20
∂t = 0.01
n = convert(Int, ceil(T / ∂t))

srm = SRM0(η₀, τᵣ, v_th)
spikes = constant_current(rate, n)
excite!(srm, spikes)

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
scatter(spikes, ones(length(spikes)), label = "Input")
scatter!(output, 2*ones(length(output)), title = "Raster Plot", xlabel = "Time (sec)", label = "Output")
savefig("srm-test-raster.png")

# plot dense voltage recording
plot(∂t .* collect(0:maximum(spikes)), voltages,
    title = "SRM Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "")
savefig("srm-test-voltage.png")