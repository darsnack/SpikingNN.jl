@testset "Threshold Poisson" begin
    baserate = rand()
    theta = rand()
    deltav = rand()
    v = rand()
    t = 1    
    dt = 1
    threshold = Threshold.Poisson(baserate, theta, deltav; rng = MersenneTwister())
    rho = baserate * exp((v - theta) / deltav)
    if rand(MersenneTwister()) < rho * dt
        @test evaluate!(threshold,t,v) == t
    else
        @test evaluate!(threshold,t,v) == zero(t)
    end      
end

@testset "Threshold Ideal" begin
    vth = rand()
    v = rand()
    t = rand()
    threshold = Threshold.Ideal(vth)
    if v >= threshold.vth
        @test evaluate!(threshold, t, v) == t
    else
        @test evaluate!(threshold, t, v) == zero(t)
    end      
end