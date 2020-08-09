@plottest begin
    # SRM0 params
    η₀ = 5.0
    τᵣ = 1.0

    # Input spike train params
    rate = 0.01
    T = 20
    ∂t = 0.01
    n = convert(Int, ceil(T / ∂t))

    rng = StableRNG(123)

    srm = Neuron(QueuedSynapse(Synapse.Delta()), SRM0(η₀, τᵣ), Threshold.Poisson(5.0, 0.5, 0.1))
    input = ConstantRate(rate; rng = rng)
    spikes = excite!(srm, input, n)

    # callback to record voltages
    voltages = Float64[]
    record = function ()
        push!(voltages, getvoltage(srm))
    end

    # simulate
    output = simulate!(srm, n; dt = ∂t, cb = record, dense = true)

    # plot raster plot
    raster_plot = rasterplot(∂t .* spikes, ∂t .* output, label = ["Input", "Output"], xlabel = "Time (sec)",
                    title = "Raster Plot (δ response)")
    xlims!(0, T)

    # plot dense voltage recording
    plot(∂t .* collect(1:n), voltages,
        title = "SRM Membrane Potential with Varying Presynaptic Responses", xlabel = "Time (sec)", ylabel = "Potential (V)", label = "δ response")

    # resimulate using presynaptic response
    srm = Neuron(QueuedSynapse(Synapse.Alpha()), SRM0(η₀, τᵣ), Threshold.Poisson(5.0, 0.5, 0.1; rng = rng))
    voltages = Float64[]
    excite!(srm, spikes)
    simulate!(srm, n; dt = ∂t, cb = record, dense = true)

    # plot voltages with response function
    voltage_plot = plot!(∂t .* collect(1:n), voltages, label = "α response")
    xlims!(0, T)

    plot(raster_plot, voltage_plot, layout = grid(2, 1), xticks = 0:T)
    
end joinpath(datadir, "Srm0-Threshold-Test.png") !isci