using SpikingNN
using Plots

dt = 0.1
spikes = [3, 20, 23, 30]
N = 100
synapse = Synapse.EPSP(ϵ₀ = 1, τm = 2, τs = 0.5)

# excite the synapse
excite!(synapse, spikes)

# get response
y = [synapse(t; dt = dt) for t in 1:N]

# plot results
scatter(dt .* spikes .- dt, zeros(length(spikes)), label = "Spikes")
plot!(dt .* collect(0:(N - 1)), y, label = "Output", minorticks = true)