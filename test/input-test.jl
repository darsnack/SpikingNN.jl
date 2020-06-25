using SpikingNN
using Statistics
using Test

@testset "input" begin

    @testset "LIF SpikeCount" begin
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

    @testset "SRM SpikeCount" begin

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

   # Assert frequency constructor matches the rate constructor
   @testset "ConstantRate Constructor" begin

    freq = 0.05
    dt = 1.0
    rate = freq * dt

    @test ConstantRate(freq, dt).rate == ConstantRate(rate).rate

    end

    @testset "StepCurrent" begin

        τ = 10

        stepcurrent = StepCurrent(τ)

        let
            flag = true

            for t in 0:100
                if t <= τ
                    evaluate!(stepcurrent, t) != 0 ? flag = false : flag = true
                        
                else
                    (evaluate!(stepcurrent, t) > 0) != true ? flag = false : flag = true
                end
            end     

            @test flag == true
        end
    end


end

