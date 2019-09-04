using SpikingNN
using Plots, DSP

dt = 0.01
spikes = Int.(ceil.([3, 20, 23, 30] ./ dt))
N = 1000
response = SpikingNN.α

# sample the response function
h = SpikingNN.sample_response(response, dt)

# construct a dense version of the spike train
n = maximum(spikes)
x = zeros(n)
x[spikes] .= 1

# convolve the the response with the spike train
y = conv(x, h)
y = y[1:n]

scatter(dt .* spikes, zeros(length(spikes)), label = "Spikes")
plot!(dt .* collect(1:N) .- dt, h, label = "Response")
plot!(dt .* collect(0:(n - 1)), x, label = "Input")
plot!(dt .* collect(0:(n - 1)), y, label = "Output", minorticks = true)