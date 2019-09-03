"""
    constant_current(rate::Real, n::Integer)

Create an array (of length `n`) of spike times corresponding to a
rate-coded neuron firing at a fixed rate.
"""
function constant_current(rate::Real, n::Integer; response = delta)
    if n < 1
        error("Cannot create a spike train of length < 1 (supplied n = $n).")
    elseif rate < 0
        error("Cannot create a spike train with rate < 0 (supplied rate = $rate).")
    end

    dist = Bernoulli(rate)
    spikes = rand(dist, n)
    times = findall(x -> x == 1, spikes)
    return times
end