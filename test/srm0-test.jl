@plottest begin
    # SRM0 params
    η₀ = 5.0
    τᵣ = 1.0
    vth = 0.5

    # Input spike train params
    rate = 0.01
    T = 15
    ∂t = 0.01
    n = convert(Int, ceil(T / ∂t))

    rng = StableRNG(123)

    srm = Neuron(QueuedSynapse(Synapse.Alpha()), SRM0(η₀, τᵣ), Threshold.Ideal(vth))
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
                             title = "Raster Plot (α response)")

    # plot dense voltage recording
    t = ∂t:∂t:(∂t * n)
    voltage_plot = plot(t, voltages, title = "SRM Membrane Potential with Varying Presynaptic Responses",
                                     xlabel = "Time (sec)", ylabel = "Potential (V)", label = "α response")

    # resimulate using presynaptic response
    voltages = Float64[]
    srm = Neuron(QueuedSynapse(Synapse.EPSP(ϵ₀ = 2, τm = 0.5, τs = 1)), SRM0(η₀, τᵣ), Threshold.Ideal(vth))
    excite!(srm, spikes)
    simulate!(srm, n; dt = ∂t, cb = record, dense = true)

    # plot voltages with response function
    voltage_plot = plot!(t, voltages, label = "EPSP response")

    plot(raster_plot, voltage_plot, layout = grid(2, 1))
end joinpath(datadir, "Srm0-Test.png") !isci