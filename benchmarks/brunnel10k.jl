using SpikingNN
using Distributions
using Distributions: sample

# simulation parameters
T = 1.0
dt = 0.1f-3
n = Int(ceil(T / dt))

# populations parameters
Ne = 800
Ni = 200

# connectivity parameters
w = 0.1
wexcite = 0.1
winhibit = -0.5
sparsity = 0.1
λ = 1f-2

# neuron parameters
τm = 20f-3
τr = 2f-3
vth = 20f-3
delay = Int(ceil(1.5f-3 / dt))

# input parameters
ρ0 = 20f0
Ninput = Ne + Ni

# create system
D = Bernoulli(1 - sparsity)
inputs = InputPopulation([ConstantRate(ρ0, dt) for _ in 1:Ninput])
W_EE = Float32.(rand(D, Ne, Ne) .* wexcite)
W_II = Float32.(rand(D, Ni, Ni) .* winhibit)
W_EI = Float32.(rand(D, Ne, Ni) .* wexcite)
W_IE = Float32.(rand(D, Ni, Ne) .* winhibit)
W_input_E = Float32.(rand(D, Ninput, Ne) .* wexcite)
W_input_I = Float32.(rand(D, Ninput, Ni) .* wexcite)
E = Population(W_EE; cell = () -> LIF(τm = τm, τr = τr),
                     threshold = () -> Threshold.Ideal(vth),
                     synapse = () -> DelayedSynapse(Synapse.Delta(), delay))
I = Population(W_II; cell = () -> LIF(τm = τm, τr = τr),
                     threshold = () -> Threshold.Ideal(vth),
                     synapse = () -> DelayedSynapse(Synapse.Delta(), delay))
net = Network(Dict(:input => inputs, :E => E, :I => I))
connect!(net, :E, :I; weights = W_EI, synapse = () -> DelayedSynapse(Synapse.Delta(), delay))
connect!(net, :I, :E; weights = W_IE, synapse = () -> DelayedSynapse(Synapse.Delta(), delay))
connect!(net, :input, :E; weights = W_input_E, synapse = () -> DelayedSynapse(Synapse.Delta(), delay))
connect!(net, :input, :I; weights = W_input_I, synapse = () -> DelayedSynapse(Synapse.Delta(), delay))

# recording callback
nsample = min(Ne, Ni, 25)
const record_excite = sample(1:Ne, nsample; replace = false)
const record_inhibit = sample(1:Ni, nsample; replace = false)
Ve = Dict{Int, Vector{Float32}}()
Vi = Dict{Int, Vector{Float32}}()
function record()
    global Ve, Vi, record_excite, record_inhibit
    for idx in record_excite
        push!(get!(Ve, idx, Float32[]), getvoltage(net[:E][idx]))
    end
    for idx in record_inhibit
        push!(get!(Vi, idx, Float32[]), getvoltage(net[:I][idx]))
    end
end

# simulate
@time simulate!(net, n; cb = record, dt = dt)