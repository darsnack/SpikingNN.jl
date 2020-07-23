@testset "Threshold Poisson" begin
    baserate = rand()
    theta = rand()
    deltav = rand()
    v = rand()
    t = 1    
    dt = 1
    @testset "Threshold Poisson without rng" begin
        threshold = Threshold.Poisson(baserate, theta, deltav)
        rho = baserate * exp((v - theta) / deltav)
        if rand(threshold.rng) < rho * dt
            @test evaluate!(threshold,t,v) == t
        else
            @test evaluate!(threshold,t,v) == zero(t)
        end 
    end
    @testset "Threshold Poisson with rng" begin
        threshold = Threshold.Poisson(baserate, theta, deltav; rng = MersenneTwister())
        rho = baserate * exp((v - theta) / deltav)
        if rand(MersenneTwister()) < rho * dt
            @test evaluate!(threshold,t,v) == t
        else
            @test evaluate!(threshold,t,v) == zero(t)
        end 
    end

end

@testset "Threshold Ideal" begin
    t = rand()
    v = rand()
    @testset "when v < vth" begin
        vth = v .+ (1-v)*(1 .- rand()) # Setting the (v,1] bound for vth    
        threshold = Threshold.Ideal(vth) 
        @test evaluate!(threshold, t, v) == zero(t) # Condition for v < threshold.vth
    end
    @testset "when v > vth" begin
        vth = v * rand() # Setting the [0,v) bound for vth
        threshold = Threshold.Ideal(vth)
        @test evaluate!(threshold, t, v) == t # Condition for v >= threshold.vth
    end         
end