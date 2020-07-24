@testset "Threshold Poisson" begin
    for rng in [nothing, Random.MersenneTwister()]
        baserate = rand()
        theta = rand()
        deltav = rand()
        v = rand()
        t = 1    
        dt = 1
        threshold = isnothing(rng) ? Threshold.Poisson(baserate, theta, deltav) : Threshold.Poisson(baserate, theta, deltav; rng = rng)
        rho = baserate * exp((v - theta) / deltav)
        if rand(threshold.rng) < rho * dt
            @test evaluate!(threshold,t,v) == t
        else
            @test evaluate!(threshold,t,v) == zero(t)
        end 
    end
end

@testset "Threshold Ideal" begin
    let v = rand()
        for vth in [v .+ (1-v)*(1 .- rand()), v * rand()] # Setting the (v,1] and [0,v) bound for vth
            t = rand()
            threshold = Threshold.Ideal(vth)
            if v < threshold.vth
                @test evaluate!(threshold, t, v) == zero(t)
            else
                @test evaluate!(threshold, t, v) == t
            end
        end
    end      
end