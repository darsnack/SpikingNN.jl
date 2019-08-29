"""
    constant_current(rate::Real, T::Integer)

Create an array of spike times corresponding to a rate-coded neuron firing
at a fixed rate.
"""
function constant_current(rate::Real, T::Integer)
    if T < 1
        error("Cannot create a spike train of length < 1 (supplied T = $T).")
    elseif rate < 0
        error("Cannot create a spike train with rate < 0 (supplied rate = $rate).")
    end

    dist = Bernoulli(rate)
    spikes = rand(dist, T)
    times = findall(x -> x == 1, spikes)
    return times
end