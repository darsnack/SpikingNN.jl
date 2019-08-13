using SpikingNN
using Plots

# LIF params
τ_m = 100
v_reset = 0.0
v_th = 1.5
R = 1.75

# Input spike train params
rate = 0.9
T = 1000

lif = LIF(:none, τ_m, v_reset, v_th, R)
spikes = constant_current(rate, T)
excite!(lif, spikes)

# println("spike times:\n  $spikes")
println("# of spikes equal: $(length(spikes) == length(lif.spikes_in))")

output = simulate!(lif)

scatter(spikes, ones(length(spikes)), label = "Input")
scatter!(output, 2*ones(length(output)), title = "Raster Plot", xlabel = "Time (sec)", label = "Output")
savefig("lif-test.png")