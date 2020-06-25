using SpikingNN
using Statistics
using Test

@testset "LIF spike count" begin
    # LIF params
    τm = 100
    vreset = 0.0
    vth = 0.1
    R = 1.75

    # Input spike train params
    rate = 0.05
    T = 1000

    lif = Neuron(QueuedSynapse(Synapse.Delta()), LIF(τm, vreset, R), Threshold.Ideal(vth))
    input = ConstantRate(rate)


    spikescount = []
    for i in 1:T
        spikes = excite!(lif, input, T)
        push!(spikescount, length(spikes))
    end

    avgspikecount = mean(spikescount)/T


    @test Float32(round(avgspikecount,digits = 2)) == input.rate
end

@testset "SRM0 spike count" begin

    # SRM0 params
    η₀ = 5.0
    τᵣ = 1.0
    vth = 0.5

    # Input spike train params
    rate = 0.01
    T = 1000

    srm = Neuron(QueuedSynapse(Synapse.Alpha()), SRM0(η₀, τᵣ), Threshold.Ideal(vth))
    input = ConstantRate(rate)


    spikescount = []
    for i in 1:T
        spikes = excite!(srm, input, T)
        push!(spikescount, length(spikes))
    end

    avgspikecount = mean(spikescount)/T


    @test Float32(round(avgspikecount,digits = 2)) == input.rate
end

