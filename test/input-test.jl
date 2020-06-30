@testset "ConstantRate" begin
    T = 10_000
    freq = rand(1:100)
    dt = rand() * (1/freq)
    rate = freq * dt
    input = ConstantRate(rate)
    # Averaging the spike count of ConstantRate over a fixed window of time should be ConstantRate.rate
    @test isapprox(count(x -> x > 0, [input(t) for t in 1:T]) / T, rate, atol=0.005)
    # Assert frequency constructor matches the rate constructor
    @test ConstantRate(freq, dt).rate == input.rate
end

# A StepCurrent should be zero < threshold time and positive > threshold time
@testset "StepCurrent" begin
    τ = 10
    T = 25
    stepcurrent = StepCurrent(τ)
    prethresh = [stepcurrent(t) for t in 1:τ]
    postthresh = [stepcurrent(t) for t in (τ + 1):T]

    @test all(x -> x == 0, prethresh)
    @test all(x -> x > 0, postthresh)    
end