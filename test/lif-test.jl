@plottest begin
    # LIF params
    τm = 100
    vreset = 0.0
    vth = 0.1
    R = 1.75

    # Input spike train params
    rate = 0.05
    T = 1000

    rng = StableRNG(123)

    lif = Neuron(QueuedSynapse(Synapse.Delta()), LIF(τm, vreset, R), Threshold.Ideal(vth))
    input = ConstantRate(rate; rng = rng)
    spikes = excite!(lif, input, T)

    # callback to record voltages
    voltages = Float64[]
    record = let lif = lif
        () -> push!(voltages, getvoltage(lif))
    end

    # compile
    simulate!(lif, T; cb = record)
    reset!(lif)
    voltages = Float64[]
    excite!(lif, spikes)

    # simulate
    simulate!(lif, T; cb = record, dense = true)

    # plot dense voltage recording
    plot(collect(1:T), voltages, title = "LIF Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "Dense")

    # repeat with dense simulation
    reset!(lif)
    voltages = Float64[]
    excite!(lif, spikes)
    output = simulate!(lif, T; cb = record)

    # plot sparse voltage recording
    voltage_plot = plot!(spikes, voltages,
        title = "LIF Membrane Potential Over Time", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "Sparse (default)")

    # plot raster plot
    raster_plot = rasterplot(spikes, output, label = ["Input", "Output"], title = "Raster Plot", xlabel = "Time (sec)")

    plot(raster_plot, voltage_plot, layout = grid(2, 1))
end joinpath(datadir, "Lif-Test.png") !isci