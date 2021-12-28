using SpikingNN
using LinearAlgebra
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)

##

N = 10
T = 100
device = SpikingNN.cpu

weights = rand(Float32, N, N)
spikes = Dict([:input => device(zeros(Int, N)), :pop => device(zeros(Int, N))])

cell = () -> SRM0(η₀ = 1, τᵣ = 10)
threshold = () -> Poisson(ρ₀ = 1, Θ = 0.15, Δᵤ = 0.001)
synapse = BiExponential
inputs = InputPopulation([ConstantRate(0.8) for _ in 1:N]) #|> device
pop = Population(weights; cell, synapse, threshold) #|> device
learner = Dict([:pop => device(STDP(A = 0.2, τ = 4, n = size(pop)))])
net = Network(Dict([:input => inputs, :pop => pop]))
connect!(net, :input, :pop; weights = Float32.(I(N)) |> collect, synapse)
# net = net |> device

##

function run!(spikes, net, t)
    evaluate!(spikes, net, t)
    update!(learner[:pop], net[:pop].weights, t, spikes[:pop], spikes[:pop])
end

@benchmark CUDA.@sync run!($spikes, $net, $(rand(1:T)))

##

@benchmark CUDA.@sync SpikingNN.step!($spikes, $net, $learner, $(rand(1:T)))

##

@benchmark CUDA.@sync simulate!($net, $learner, $T)
