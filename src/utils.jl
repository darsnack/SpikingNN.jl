function sample_response(response::Function, dt::Real = 1.0)
    N = Int(10 / dt) # number of samples to acquire (defaults to 10 samples when dt = 1.0)
    response.(dt .* collect(1:N) .- dt), N
end