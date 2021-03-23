var documenterSearchIndex = {"docs":
[{"location":"input/#Inputs","page":"Inputs","title":"Inputs","text":"","category":"section"},{"location":"input/","page":"Inputs","title":"Inputs","text":"SpikingNN provides several synthetic inputs:","category":"page"},{"location":"input/","page":"Inputs","title":"Inputs","text":"constant rate\nstep current\ninhomogenous Poisson process","category":"page"},{"location":"input/","page":"Inputs","title":"Inputs","text":"Inputs can also form a InputPopulation that behaves similar to a Population.","category":"page"},{"location":"input/#Constant-Rate","page":"Inputs","title":"Constant Rate","text":"","category":"section"},{"location":"input/","page":"Inputs","title":"Inputs","text":"A constant rate input fires at a fixed frequency.","category":"page"},{"location":"input/","page":"Inputs","title":"Inputs","text":"ConstantRate\nevaluate!(::ConstantRate, ::Integer; ::Real)","category":"page"},{"location":"input/#SpikingNN.ConstantRate","page":"Inputs","title":"SpikingNN.ConstantRate","text":"ConstantRate(rate::Real)\nConstantRate{T}(rate::Real)\nConstantRate(freq::Real, dt::Real)\n\nCreate a constant rate input where the probability a spike occurs is Bernoulli(rate). rate-coded neuron firing at a fixed rate. Alternatively, specify freq in Hz at a simulation time step of dt.\n\n\n\n\n\n","category":"type"},{"location":"input/#SpikingNN.evaluate!-Tuple{ConstantRate,Integer}","page":"Inputs","title":"SpikingNN.evaluate!","text":"evaluate!(input::ConstantRate, t::Integer; dt::Real = 1.0)\n(::ConstantRate)(t::Integer; dt::Real = 1.0)\nevaluate!(inputs::AbstractArray{<:ConstantRate}, t::Integer; dt::Real = 1.0)\n\nEvaluate a constant rate-code input at time t.\n\n\n\n\n\n","category":"method"},{"location":"input/#Step-Current","page":"Inputs","title":"Step Current","text":"","category":"section"},{"location":"input/","page":"Inputs","title":"Inputs","text":"A step current input is low until a fixed time step, then it is high.","category":"page"},{"location":"input/","page":"Inputs","title":"Inputs","text":"StepCurrent\nevaluate!(::StepCurrent, ::Integer; ::Real)","category":"page"},{"location":"input/#SpikingNN.StepCurrent","page":"Inputs","title":"SpikingNN.StepCurrent","text":"StepCurrent(τ::Real)\n\nCreate a step current input that turns on at time τ seconds.\n\n\n\n\n\n","category":"type"},{"location":"input/#SpikingNN.evaluate!-Tuple{StepCurrent,Integer}","page":"Inputs","title":"SpikingNN.evaluate!","text":"evaluate!(input::StepCurrent, t::Integer; dt::Real = 1.0)\n(::StepCurrent)(t::Integer; dt::Real = 1.0)\nevaluate!(inputs::AbstractArray{<:StepCurrent}, t::Integer; dt::Real = 1.0)\n\nEvaluate a step current input at time t.\n\n\n\n\n\n","category":"method"},{"location":"input/#Inhomogenous-Poisson-Input-Process","page":"Inputs","title":"Inhomogenous Poisson Input Process","text":"","category":"section"},{"location":"input/","page":"Inputs","title":"Inputs","text":"An input that behaves like an inhomogenous Poisson process given by a provided instantaneous rate function.","category":"page"},{"location":"input/","page":"Inputs","title":"Inputs","text":"PoissonInput\nevaluate!(::PoissonInput, ::Integer; ::Real)","category":"page"},{"location":"input/#SpikingNN.PoissonInput","page":"Inputs","title":"SpikingNN.PoissonInput","text":"PoissonInput(ρ₀::Real, λ::Function)\n\nCreate a inhomogenous Poisson input function according to\n\nX  mathrmdt rho_0 lambda(t)\n\nwhere X sim mathrmUnif(0 1). Note that dt must be appropriately specified to ensure correct behavior.\n\nFields:\n\nρ₀::Real: baseline firing rate\nλ::(Integer; dt::Integer) -> Real: a function that returns the   instantaneous firing rate at time t\n\n\n\n\n\n","category":"type"},{"location":"input/#SpikingNN.evaluate!-Tuple{PoissonInput,Integer}","page":"Inputs","title":"SpikingNN.evaluate!","text":"evaluate!(input::PoissonInput, t::Integer; dt::Real = 1.0)\n(::PoissonInput)(t::Integer; dt::Real = 1.0)\nevaluate!(inputs::AbstractArray{<:PoissonInput}, t::Integer; dt::Real = 1.0)\n\nEvaluate a inhomogenous Poisson input at time t.\n\n\n\n\n\n","category":"method"},{"location":"input/#Input-Population","page":"Inputs","title":"Input Population","text":"","category":"section"},{"location":"input/","page":"Inputs","title":"Inputs","text":"InputPopulation\nevaluate!(::InputPopulation, ::Integer; ::Real)","category":"page"},{"location":"input/#SpikingNN.InputPopulation","page":"Inputs","title":"SpikingNN.InputPopulation","text":"InputPopulation{IT<:StructArray{<:AbstractInput}}\n\nAn InputPopulation is a population of AbstractInputs.\n\n\n\n\n\n","category":"type"},{"location":"input/#SpikingNN.evaluate!-Tuple{InputPopulation,Integer}","page":"Inputs","title":"SpikingNN.evaluate!","text":"evaluate!(pop::InputPopulation, t::Integer; dt::Real = 1.0)\n(::InputPopulation)(t::Integer; dt::Real = 1.0)\n\nEvaluate a population of inputs at time t.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#Synapse-Models","page":"Synapse Models","title":"Synapse Models","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"The following synapse models are supported:","category":"page"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"Dirac-Delta\nAlpha\nExcitatory post-synaptic potential (EPSP)","category":"page"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"The following wrapper types are also supported:","category":"page"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"QueuedSynapse\nDelayedSynapse","category":"page"},{"location":"synapse-models/#Default-function-implementations","page":"Synapse Models","title":"Default function implementations","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"Some function implementations apply to all synapses (unless they override them) that inherit from AbstractSynapse.","category":"page"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"excite!(::AbstractSynapse, ::Vector{<:Integer})\nspike!(::AbstractSynapse, ::Integer; ::Real)","category":"page"},{"location":"synapse-models/#SpikingNN.excite!-Tuple{AbstractSynapse,Array{var\"#s1\",1} where var\"#s1\"<:Integer}","page":"Synapse Models","title":"SpikingNN.excite!","text":"excite!(synapse::AbstractSynapse, spikes::Vector{<:Integer})\nexcite!(synapses::AbstractArray{<:AbstractSynapse}, spikes::Vector{<:Integer})\n\nExcite a synapse with a vector of spikes by calling excite!(synapse, spike) for spike in spikes.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.spike!-Tuple{AbstractSynapse,Integer}","page":"Synapse Models","title":"SpikingNN.spike!","text":"spike!(synapse::AbstractSynapse, spike::Integer; dt::Real = 1.0)\nspike!(synapse::AbstractArray{<:AbstractSynapse}, spikes::AbstractArray{<:Integer}; dt::Real = 1.0)\n\nNotify a synapse that the post-synaptic neuron has released a spike. The default implmentation is to do nothing. Override this behavior by dispatching on your synapse type.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#Dirac-Delta","page":"Synapse Models","title":"Dirac-Delta","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"Synapse.Delta\nexcite!(::Synapse.Delta, ::Integer)\nevaluate!(::Synapse.Delta, ::Integer; ::Real)\nreset!(::Synapse.Delta)","category":"page"},{"location":"synapse-models/#SpikingNN.Synapse.Delta","page":"Synapse Models","title":"SpikingNN.Synapse.Delta","text":"Delta{IT<:Integer, VT<:Real}\n\nA synapse representing a Dirac-delta at lastspike with amplitude q.\n\n\n\n\n\n","category":"type"},{"location":"synapse-models/#SpikingNN.excite!-Tuple{SpikingNN.Synapse.Delta,Integer}","page":"Synapse Models","title":"SpikingNN.excite!","text":"excite!(synapse::Delta, spike::Integer)\nexcite!(synapses::AbstractArray{<:Delta}, spike::Integer)\n\nExcite synapse with a spike (spike == time step of spike).\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.evaluate!-Tuple{SpikingNN.Synapse.Delta,Integer}","page":"Synapse Models","title":"SpikingNN.evaluate!","text":"evaluate!(synapse::Delta, t::Integer; dt::Real = 1.0)\n(synapse::Delta)(t::Integer; dt::Real = 1.0)\nevaluate!(synapses::AbstractArray{<:Delta}, t::Integer; dt::Real = 1.0)\n\nReturn synapse.q if t == synapse.lastspike otherwise return zero.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.reset!-Tuple{SpikingNN.Synapse.Delta}","page":"Synapse Models","title":"SpikingNN.reset!","text":"reset!(synapse::Delta)\nreset!(synapses::AbstractArray{<:Delta})\n\nReset synapse.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#Alpha","page":"Synapse Models","title":"Alpha","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"Synapse.Alpha\nexcite!(::Synapse.Alpha, ::Integer)\nevaluate!(::Synapse.Alpha, ::Integer; ::Real)\nreset!(::Synapse.Alpha)","category":"page"},{"location":"synapse-models/#SpikingNN.Synapse.Alpha","page":"Synapse Models","title":"SpikingNN.Synapse.Alpha","text":"Alpha{IT<:Integer, VT<:Real}\n\nSynapse that returns (t - lastspike) * (q / τ) * exp(-(t - lastspike - τ) / τ) Θ(t - lastspike) (where Θ is the Heaviside function).\n\n\n\n\n\n","category":"type"},{"location":"synapse-models/#SpikingNN.excite!-Tuple{SpikingNN.Synapse.Alpha,Integer}","page":"Synapse Models","title":"SpikingNN.excite!","text":"excite!(synapse::Alpha, spike::Integer)\nexcite!(synapses::AbstractArray{<:Alpha}, spike::Integer)\n\nExcite synapse with a spike (spike == time step of spike).\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.evaluate!-Tuple{SpikingNN.Synapse.Alpha,Integer}","page":"Synapse Models","title":"SpikingNN.evaluate!","text":"evaluate!(synapse::Alpha, t::Integer; dt::Real = 1.0)\n(synapse::Alpha)(t::Integer; dt::Real = 1.0)\nevaluate!(synapses::AbstractArray{<:Alpha}, t::Integer; dt::Real = 1.0)\n\nEvaluate an alpha synapse. See Synapse.Alpha.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.reset!-Tuple{SpikingNN.Synapse.Alpha}","page":"Synapse Models","title":"SpikingNN.reset!","text":"reset!(synapse::Alpha)\nreset!(synapses::AbstractArray{<:Alpha})\n\nReset synapse.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#Exitatory-Post-Synaptic-Potential","page":"Synapse Models","title":"Exitatory Post-Synaptic Potential","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"Synapse.EPSP\nexcite!(::Synapse.EPSP, ::Integer)\nspike!(::Synapse.EPSP, ::Integer; ::Real)\nevaluate!(::Synapse.EPSP, ::Integer; ::Real)\nreset!(::Synapse.EPSP)","category":"page"},{"location":"synapse-models/#SpikingNN.Synapse.EPSP","page":"Synapse Models","title":"SpikingNN.Synapse.EPSP","text":"EPSP{T<:Real}\n\nSynapse that returns (ϵ₀ / τm - τs) * (exp(-Δ / τm) - exp(-Δ / τs)) Θ(Δ) (where Θ is the Heaviside function and Δ = t - lastspike).\n\nSpecifically, this is the EPSP time course for the SRM0 model introduced by Gerstner. Details: Spiking Neuron Models: Single Neurons, Populations, Plasticity\n\n\n\n\n\n","category":"type"},{"location":"synapse-models/#SpikingNN.excite!-Tuple{SpikingNN.Synapse.EPSP,Integer}","page":"Synapse Models","title":"SpikingNN.excite!","text":"excite!(synapse::EPSP, spike::Integer)\nexcite!(synapses::AbstractArray{<:EPSP}, spike::Integer)\n\nExcite synapse with a spike (spike == time step of spike).\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.spike!-Tuple{SpikingNN.Synapse.EPSP,Integer}","page":"Synapse Models","title":"SpikingNN.spike!","text":"spike!(synapse::EPSP, spike::Integer; dt::Real = 1.0)\nspike!(synapses::AbstractArray{<:EPSP}, spikes; dt::Real = 1.0)\n\nReset synapse when the post-synaptic neuron spikes.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.evaluate!-Tuple{SpikingNN.Synapse.EPSP,Integer}","page":"Synapse Models","title":"SpikingNN.evaluate!","text":"evaluate!(synapse::EPSP, t::Integer; dt::Real = 1.0)\n(synapse::EPSP)(t::Integer; dt::Real = 1.0)\nevaluate!(synapses::AbstractArray{<:EPSP}, t::Integer; dt::Real = 1.0)\n\nEvaluate an EPSP synapse. See Synapse.EPSP.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.reset!-Tuple{SpikingNN.Synapse.EPSP}","page":"Synapse Models","title":"SpikingNN.reset!","text":"reset!(synapse::EPSP)\nreset!(synapses::AbstractArray{<:EPSP})\n\nReset synapse.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#Queued-Synapse","page":"Synapse Models","title":"Queued Synapse","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"QueuedSynapse\nexcite!(::QueuedSynapse, ::Integer)\nevaluate!(::QueuedSynapse, ::Integer; ::Real)\nreset!(::QueuedSynapse)","category":"page"},{"location":"synapse-models/#SpikingNN.Synapse.QueuedSynapse","page":"Synapse Models","title":"SpikingNN.Synapse.QueuedSynapse","text":"QueuedSynapse{ST<:AbstractSynapse, IT<:Integer}\n\nA QueuedSynapse excites its internal synapse when the timestep matches the head of the queue. Wrapping a synapse in this type allows you to pre-load several spike excitation times, and the   internal synapse will be excited as those time stamps are evaluated. This can be useful for cases where it is more efficient to load all the input spikes before simulation.\n\nNote: currently only supported on CPU.\n\n\n\n\n\n","category":"type"},{"location":"synapse-models/#SpikingNN.excite!-Tuple{QueuedSynapse,Integer}","page":"Synapse Models","title":"SpikingNN.excite!","text":"excite!(synapse::QueuedSynapse, spike::Integer)\nexcite!(synapses::AbstractArray{<:QueuedSynapse}, spike::Integer)\n\nExcite synapse with a spike (spike == time step of spike) by pushing   spike onto synapse.queue.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.evaluate!-Tuple{QueuedSynapse,Integer}","page":"Synapse Models","title":"SpikingNN.evaluate!","text":"evaluate!(synapse::QueuedSynapse, t::Integer; dt::Real = 1.0)\n(synapse::QueuedSynapse)(t::Integer; dt::Real = 1.0)\nevaluate!(synapses::AbstractArray{<:QueuedSynapse}, t::Integer; dt::Real = 1.0)\n\nEvaluate synapse at time t by first exciting synapse.core with a spike if   there is one to process, then evaluating synapse.core.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.reset!-Tuple{QueuedSynapse}","page":"Synapse Models","title":"SpikingNN.reset!","text":"reset!(synapse::QueuedSynapse)\nreset!(synapses::AbstractArray{<:QueuedSynapse})\n\nClear synapse.queue and reset synapse.core.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#Delayed-Synapse","page":"Synapse Models","title":"Delayed Synapse","text":"","category":"section"},{"location":"synapse-models/","page":"Synapse Models","title":"Synapse Models","text":"DelayedSynapse\nexcite!(::DelayedSynapse, ::Integer)\nevaluate!(::DelayedSynapse, ::Integer; ::Real)\nreset!(::DelayedSynapse)","category":"page"},{"location":"synapse-models/#SpikingNN.Synapse.DelayedSynapse","page":"Synapse Models","title":"SpikingNN.Synapse.DelayedSynapse","text":"DelayedSynapse\n\nA DelayedSynapse adds a fixed delay to spikes when exciting its internal synapse.\n\n\n\n\n\n","category":"type"},{"location":"synapse-models/#SpikingNN.excite!-Tuple{DelayedSynapse,Integer}","page":"Synapse Models","title":"SpikingNN.excite!","text":"excite!(synapse::DelayedSynapse, spike::Integer)\nexcite!(synapses::AbstractArray{<:DelayedSynapse}, spike::Integer)\n\nExcite synapse.core with a spike + synapse.delay (spike == time step of spike).\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.evaluate!-Tuple{DelayedSynapse,Integer}","page":"Synapse Models","title":"SpikingNN.evaluate!","text":"evaluate!(synapse::DelayedSynapse, t::Integer; dt::Real = 1.0)\n(synapse::DelayedSynapse)(t::Integer; dt::Real = 1.0)\nevaluate!(synapses::AbstractArray{<:DelayedSynapse}, t::Integer; dt::Real = 1.0)\n\nEvaluate synapse.core at time t.\n\n\n\n\n\n","category":"method"},{"location":"synapse-models/#SpikingNN.reset!-Tuple{DelayedSynapse}","page":"Synapse Models","title":"SpikingNN.reset!","text":"reset!(synapse::DelayedSynapse)\nreset!(synapses::AbstractArray{<:DelayedSynapse})\n\nReset synapse.core.\n\n\n\n\n\n","category":"method"},{"location":"population/#Populations","page":"Populations","title":"Populations","text":"","category":"section"},{"location":"population/","page":"Populations","title":"Populations","text":"A population of neurons is a collection of a single type of neuron connected by a single type of synapse. A learning mechanism is associated with the synapses of a population.","category":"page"},{"location":"population/","page":"Populations","title":"Populations","text":"Population\nsize(::Population)\nneurons\nsynapses\nevaluate!(::Population, ::Integer; ::Real, ::Any, ::Any)\nupdate!(::Population, ::Integer; ::Real)\nreset!(::Population)\nsimulate!(::Population, ::Integer; ::Real, ::Any, ::Any, ::Any)","category":"page"},{"location":"population/#SpikingNN.Population","page":"Populations","title":"SpikingNN.Population","text":"Population{T<:Soma,\n           NT<:AbstractArray{T, 1},\n           WT<:AbstractMatrix{<:Real},\n           ST<:AbstractArray{<:AbstractSynapse, 2},\n           LT<:AbstractLearner} <: AbstractArray{T, 1}\n\nA population of neurons is an array of Somas,   a weighted matrix of synapses, and a learner.\n\nFields:\n\nsomas::AbstractArray{<:Soma, 1}: a vector of somas\nweights::AbstractMatrix{<:Real}: a weight matrix\nsynapses::AbstractArray{<:AbstractSynapse, 2}: a matrix of synapses\nlearner::AbstractLearner: a learning mechanism\n\n\n\n\n\n","category":"type"},{"location":"population/#Base.size-Tuple{Population}","page":"Populations","title":"Base.size","text":"size(pop::Population)\n\nReturn the number of neurons in a population.\n\n\n\n\n\n","category":"method"},{"location":"population/#SpikingNN.neurons","page":"Populations","title":"SpikingNN.neurons","text":"neurons(pop::Population)\n\nReturn an array of neurons within the population.\n\n\n\n\n\n","category":"function"},{"location":"population/#SpikingNN.synapses","page":"Populations","title":"SpikingNN.synapses","text":"synapses(pop::Population)\n\nReturn an array of edges representing the synapses within the population.\n\n\n\n\n\n","category":"function"},{"location":"population/#SpikingNN.evaluate!-Tuple{Population,Integer}","page":"Populations","title":"SpikingNN.evaluate!","text":"evaluate!(pop::Population, t::Integer; dt::Real = 1.0, dense = false, inputs = nothing)\n(::Population)(t::Integer; dt::Real = 1.0, dense = false)\n\nEvaluate a population of neurons at time step t. Return a vector of time stamps (t if the neuron spiked and zero otherwise).\n\n\n\n\n\n","category":"method"},{"location":"population/#SpikingNN.reset!-Tuple{Population}","page":"Populations","title":"SpikingNN.reset!","text":"reset!(pop::Population)\n\nReset pop.synapses and pop.somas.\n\n\n\n\n\n","category":"method"},{"location":"population/#SpikingNN.simulate!-Tuple{Population,Integer}","page":"Populations","title":"SpikingNN.simulate!","text":"simulate!(pop::Population, dt::Real = 1.0)\n\nSimulate a population of neurons. Optionally specify a learner. The prespike and postspike functions will be called immediately after either event occurs.\n\nFields:\n\npop::Population: the population to simulate\nT::Integer: number of time steps to simulate\ndt::Real: the simulation time step\ncb::Function: a callback function that is called after event evaluation (expects (neuron_id, t) as input)\ndense::Bool: set to true to evaluate every time step even in the absence of events\n\n\n\n\n\n","category":"method"},{"location":"threshold-models/#Threshold-Models","page":"Threshold Models","title":"Threshold Models","text":"","category":"section"},{"location":"threshold-models/","page":"Threshold Models","title":"Threshold Models","text":"The following threshold models are supported:","category":"page"},{"location":"threshold-models/","page":"Threshold Models","title":"Threshold Models","text":"Ideal\nInhomogenous Poisson process","category":"page"},{"location":"threshold-models/#Ideal","page":"Threshold Models","title":"Ideal","text":"","category":"section"},{"location":"threshold-models/","page":"Threshold Models","title":"Threshold Models","text":"Threshold.Ideal\nevaluate!(::Threshold.Ideal, ::Real, ::Real; ::Real)","category":"page"},{"location":"threshold-models/#SpikingNN.Threshold.Ideal","page":"Threshold Models","title":"SpikingNN.Threshold.Ideal","text":"Ideal(vth::Real)\n\nAn ideal threshold spikes when v > vth.\n\n\n\n\n\n","category":"type"},{"location":"threshold-models/#SpikingNN.evaluate!-Tuple{SpikingNN.Threshold.Ideal,Real,Real}","page":"Threshold Models","title":"SpikingNN.evaluate!","text":"evaluate!(threshold::Ideal, t::Real, v::Real; dt::Real = 1.0)\n(threshold::Ideal)(t::Real, v::Real; dt::Real = 1.0)\nevaluate!(thresholds::AbstractArray{<:Ideal}, t::Integer, v; dt::Real = 1.0)\n\nReturn t when v > threshold.vth.\n\n\n\n\n\n","category":"method"},{"location":"threshold-models/#Inhomogenous-Poisson-Process","page":"Threshold Models","title":"Inhomogenous Poisson Process","text":"","category":"section"},{"location":"threshold-models/","page":"Threshold Models","title":"Threshold Models","text":"Threshold.Poisson\nevaluate!(::Threshold.Poisson, ::Integer, ::Real; ::Real)","category":"page"},{"location":"threshold-models/#SpikingNN.Threshold.Poisson","page":"Threshold Models","title":"SpikingNN.Threshold.Poisson","text":"Poisson(ρ₀::Real, Θ::Real, Δᵤ::Real, rng::AbstractRNG)\n\nChoose to output a spike based on a inhomogenous Poisson process given by\n\nX  mathrmdt  rho_0 expleft(fracv - ThetaDelta_uright)\n\nwhere X sim mathrmUnif(0 1). Accordingly, dt must be set correctly so that the neuron does not always spike.\n\nFields:\n\nρ₀::Real: baseline firing rate at threshold\nΘ::Real: firing threshold\nΔᵤ::Real: voltage resolution\nrng: random number generation\n\n\n\n\n\n","category":"type"},{"location":"threshold-models/#SpikingNN.evaluate!-Tuple{SpikingNN.Threshold.Poisson,Integer,Real}","page":"Threshold Models","title":"SpikingNN.evaluate!","text":"evaluate!(threshold::Poisson, t::Integer, v::Real; dt::Real = 1.0)\n(::Poisson)(t::Integer, v::Real; dt::Real = 1.0)\nevaluate!(thresholds::AbstractArray{<:Poisson}, t::Integer, v; dt::Real = 1.0)\n\nEvaluate Poisson threshold function. See Threshold.Poisson.\n\n\n\n\n\n","category":"method"},{"location":"neuron/#Neurons","page":"Neurons","title":"Neurons","text":"","category":"section"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"A neuron in SpikingNN is compromised of a soma (cell body) and vector of synapses (dendrites), much like a biological neuron. The soma itself is broken into the cell and threshold function. Beyond making SpikingNN flexible and extensible, this breakdown of a neuron into parts allows for efficient evaluation.","category":"page"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"A Neuron can be simulated with simulate!.","category":"page"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"simulate!(::Neuron, ::Integer; ::Real, ::Any, ::Any)","category":"page"},{"location":"neuron/#SpikingNN.simulate!-Tuple{Neuron,Integer}","page":"Neurons","title":"SpikingNN.simulate!","text":"simulate!(neuron::Neuron, T::Integer; dt::Real = 1.0, cb = () -> (), dense = false)\n\nFields:\n\nneuron::Neuron: the neuron to simulate\nT::Integer: number of time steps to simulate\ndt::Real: the length of simulation time step\ncb::Function: a callback function that is called at the start of each time step\ndense::Bool: set to true to evaluate every time step even in the absence of events\n\n\n\n\n\n","category":"method"},{"location":"neuron/#Soma","page":"Neurons","title":"Soma","text":"","category":"section"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"A Soma is made up of a cell body (found in Neuron Models) and theshold function (found in Threshold Models).","category":"page"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"Soma\nexcite!(::Union{Soma, AbstractArray{<:Soma}}, ::Any)\nevaluate!(::Soma, t::Integer; dt::Real = 1.0)\nreset!(::Union{Soma, AbstractArray{<:Soma}})","category":"page"},{"location":"neuron/#SpikingNN.Soma","page":"Neurons","title":"SpikingNN.Soma","text":"Soma{BT<:AbstractCell, TT<:AbstractThreshold}\n\nA Soma is a cell body + a threshold.\n\n\n\n\n\n","category":"type"},{"location":"neuron/#SpikingNN.excite!-Tuple{Union{AbstractArray{var\"#s1\",N} where N where var\"#s1\"<:Soma, Soma},Any}","page":"Neurons","title":"SpikingNN.excite!","text":"excite!(soma, current)\n\nInject current into soma.body.\n\n\n\n\n\n","category":"method"},{"location":"neuron/#SpikingNN.evaluate!-Tuple{Soma,Integer}","page":"Neurons","title":"SpikingNN.evaluate!","text":"evaluate!(soma::Soma, t::Integer; dt::Real = 1.0)\n(::Soma)(t::Integer; dt::Real = 1.0)\nevaluate!(somas::AbstractArray{<:Soma}, t::Integer; dt::Real = 1.0)\n\nEvaluate the soma's cell body, decide whether to spike according to the  threshold, then register the spike event with the cell body. Return the spike event (0 for no spike or t for a spike).\n\n\n\n\n\n","category":"method"},{"location":"neuron/#SpikingNN.reset!-Tuple{Union{AbstractArray{var\"#s1\",N} where N where var\"#s1\"<:Soma, Soma}}","page":"Neurons","title":"SpikingNN.reset!","text":"reset!(soma::T) where T<:Union{Soma, AbstractArray{<:Soma}}\n\nReset soma.body.\n\n\n\n\n\n","category":"method"},{"location":"neuron/#Neuron","page":"Neurons","title":"Neuron","text":"","category":"section"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"Neurons are the smallest simulated block in SpikingNN.","category":"page"},{"location":"neuron/","page":"Neurons","title":"Neurons","text":"Neuron\nconnect!(::Neuron, ::AbstractSynapse)\ngetvoltage(::Neuron)\nexcite!(::Neuron, ::Integer)\nevaluate!(::Neuron, ::Integer; ::Real)\nreset!(::Union{Neuron, AbstractArray{<:Neuron}})","category":"page"},{"location":"neuron/#SpikingNN.Neuron","page":"Neurons","title":"SpikingNN.Neuron","text":"Neuron{ST<:AbstractArray{<:AbstractSynapse}, CT<:Soma}\n\nA Neuron is a vector of synapses feeding into a soma.\n\n\n\n\n\n","category":"type"},{"location":"neuron/#SpikingNN.connect!-Tuple{Neuron,AbstractSynapse}","page":"Neurons","title":"SpikingNN.connect!","text":"connect!(neuron::Neuron, synapse::AbstractSynapse)\n\nConnect synapse into neuron.\n\n\n\n\n\n","category":"method"},{"location":"neuron/#SpikingNN.getvoltage-Tuple{Neuron}","page":"Neurons","title":"SpikingNN.getvoltage","text":"getvoltage(neuron::Neuron)\n\nGet the current membrane potential of neuron.soma.\n\n\n\n\n\n","category":"method"},{"location":"neuron/#SpikingNN.excite!-Tuple{Neuron,Integer}","page":"Neurons","title":"SpikingNN.excite!","text":"excite!(neuron::Neuron, spike::Integer)\nexcite!(neuron::Neuron, spikes::Array{<:Integer})\nexcite!(neuron::Neuron, input, T::Integer; dt::Real = 1.0)\n\nExcite a neuron's synapses with spikes. Or excite a neuron's synapses with an arbitrary input function evaluated from 1:T. input must satisfy the following signature: input(t; dt).\n\n\n\n\n\n","category":"method"},{"location":"neuron/#SpikingNN.evaluate!-Tuple{Neuron,Integer}","page":"Neurons","title":"SpikingNN.evaluate!","text":"evaluate!(neuron::Neuron, t::Integer; dt::Real = 1.0)\n(::Neuron)(t::Integer; dt::Real = 1.0)\n\nEvaluate a neuron at time t by evaluating all its synapses,  exciting the soma with current, then registering post-synaptic  spikes with the synapses. Return the spike event (0 for no spike or t for spike).\n\n\n\n\n\n","category":"method"},{"location":"neuron/#SpikingNN.reset!-Tuple{Union{AbstractArray{var\"#s1\",N} where N where var\"#s1\"<:Neuron, Neuron}}","page":"Neurons","title":"SpikingNN.reset!","text":"reset!(neuron::T) where T<:Union{Neuron, AbstractArray{<:Neuron}}\n\nReset neuron.synapses and neuron.soma.\n\n\n\n\n\n","category":"method"},{"location":"#SpikingNN.jl","page":"Home","title":"SpikingNN.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SpikingNN is spiking neural network (SNN) simulator written in Julia that targets multiple platforms. The span of appropriate hardware/software combinations for an arbitrary SNNs is quite wide. Emulating an SNN involves simulating a large, interconnected dynamical system — this allows for sparsity in space and time. If a network is dense in both dimensions, a GPU (or multi-GPU) target is most appropriate. If a network uses temporal coding, its activity is sparse over time, so event-driven simulation is well-suited. Targeting these various platforms should not require code changes or complete knowledge of low-level programming requirements. On the other hand, using glue-languages, like Python, to generate low-level code can leave performance on the table and become difficult to extend and debug.","category":"page"},{"location":"","page":"Home","title":"Home","text":"SpikingNN attempts to address these issues by leveraging the rich Julia ecosystem (StructArrays.jl, CuArrays.jl, to name a few packages). These packages allow us to remap the underlying representation of the network to different platforms. This packages provides a framework for building spiking neural networks and simple functions like gpu to map the network to supported targets. It is also designed to be extensible, so that implementing a simple interface allows your model to enjoy the same remapping functionality.","category":"page"},{"location":"#Suported-Platforms","page":"Home","title":"Suported Platforms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Available platforms:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Dense CPU\nDense single-node GPU","category":"page"},{"location":"","page":"Home","title":"Home","text":"Planned platforms:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Sparse CPU\nEvent-driven (sparse in time) CPU\nBlock-sparse in time GPU\nDistributed CPU/GPU","category":"page"},{"location":"network/#Networks","page":"Networks","title":"Networks","text":"","category":"section"},{"location":"network/","page":"Networks","title":"Networks","text":"Multiple Populations and InputPopulations can be connected together with different synapse types and different learners. This is done using a Network.","category":"page"},{"location":"neuron-models/#Neuron-Models","page":"Neuron Models","title":"Neuron Models","text":"","category":"section"},{"location":"neuron-models/","page":"Neuron Models","title":"Neuron Models","text":"The following neuron (cell body) models are supported:","category":"page"},{"location":"neuron-models/","page":"Neuron Models","title":"Neuron Models","text":"Leaky integrate-and-fire (LIF)\nSimplified spike response model (SRM0)","category":"page"},{"location":"neuron-models/#Leaky-Integrate-and-Fire","page":"Neuron Models","title":"Leaky Integrate-and-Fire","text":"","category":"section"},{"location":"neuron-models/","page":"Neuron Models","title":"Neuron Models","text":"LIF\nexcite!(::LIF, ::Any)\nspike!(::LIF, ::Integer; ::Real)\nevaluate!(::LIF, ::Integer; ::Real)\nreset!(::LIF)","category":"page"},{"location":"neuron-models/#SpikingNN.LIF","page":"Neuron Models","title":"SpikingNN.LIF","text":"LIF\n\nA leaky-integrate-fire neuron described by the following differential equation\n\nfracmathrmdvmathrmdt = fracRtau I - lambda\n\nFields\n\nvoltage::VT: membrane potential\ncurrent::VT: injected (unprocessed) current\nlastt::IT: the last time this neuron processed a spike\nτm::VT: membrane time constant\nvreset::VT: reset voltage potential\nR::VT: resistive constant (typically = 1)\n\n\n\n\n\n","category":"type"},{"location":"neuron-models/#SpikingNN.excite!-Tuple{LIF,Any}","page":"Neuron Models","title":"SpikingNN.excite!","text":"excite!(neuron::LIF, current)\nexcite!(neurons::AbstractArray{<:LIF}, current)\n\nExcite a neuron with external current.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#SpikingNN.spike!-Tuple{LIF,Integer}","page":"Neuron Models","title":"SpikingNN.spike!","text":"spike!(neuron::LIF, t::Integer; dt::Real = 1.0)\nspike!(neurons::AbstractArray{<:LIF}, spikes; dt::Real = 1.0)\n\nRecord a output spike from the threshold function with the neuron body. Sets neuron.lastt.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#SpikingNN.evaluate!-Tuple{LIF,Integer}","page":"Neuron Models","title":"SpikingNN.evaluate!","text":"evaluate!(neuron::LIF, t::Integer; dt::Real = 1.0)\n(neuron::LIF)(t::Integer; dt::Real = 1.0)\nevaluate!(neurons::AbstractArray{<:LIF}, t::Integer; dt::Real = 1.0)\n\nEvaluate the neuron model at time t. Return the resulting membrane potential.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#SpikingNN.reset!-Tuple{LIF}","page":"Neuron Models","title":"SpikingNN.reset!","text":"reset!(neuron::LIF)\nreset!(neurons::AbstractArray{<:LIF})\n\nReset neuron by setting the membrane potential to neuron.vreset.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#Simplified-Spike-Response-Model","page":"Neuron Models","title":"Simplified Spike Response Model","text":"","category":"section"},{"location":"neuron-models/","page":"Neuron Models","title":"Neuron Models","text":"SRM0\nexcite!(::SRM0, ::Any)\nspike!(::SRM0, ::Integer; ::Real)\nevaluate!(::SRM0, ::Integer; ::Real)\nreset!(::SRM0)","category":"page"},{"location":"neuron-models/#SpikingNN.SRM0","page":"Neuron Models","title":"SpikingNN.SRM0","text":"SRM0\n\nA SRM0 neuron described by\n\nv(t) = eta(t - t^f) Theta(t - t^f) + I\n\nwhere Theta is the Heaviside function and   t^f is the last output spike time.\n\nNote: The SRM0 model normally includes an EPSP term which   is modeled by Synapse.EPSP\n\nFor more details see:   Spiking Neuron Models: Single Neurons, Populations, Plasticity\n\nFields:\n\nvoltage::VT: membrane potential\ncurrent::VT: injected (unprocessed) current\nlastspike::VT: last time this neuron spiked\nη::F: refractory response function\n\n\n\n\n\n","category":"type"},{"location":"neuron-models/#SpikingNN.excite!-Tuple{SRM0,Any}","page":"Neuron Models","title":"SpikingNN.excite!","text":"excite!(neuron::SRM0, current)\nexcite!(neurons::AbstractArray{<:SRM0}, current)\n\nExcite an SRM0 neuron with external current.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#SpikingNN.spike!-Tuple{SRM0,Integer}","page":"Neuron Models","title":"SpikingNN.spike!","text":"spike!(neuron::SRM0, t::Integer; dt::Real = 1.0)\nspike!(neurons::AbstractArray{<:SRM0}, spikes; dt::Real = 1.0)\n\nRecord a output spike from the threshold function with the neuron body. Sets neuron.lastspike.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#SpikingNN.evaluate!-Tuple{SRM0,Integer}","page":"Neuron Models","title":"SpikingNN.evaluate!","text":"evaluate!(neuron::SRM0, t::Integer; dt::Real = 1.0)\n(neuron::SRM0)(t::Integer; dt::Real = 1.0)\nevaluate!(neurons::AbstractArray{<:SRM0}, t::Integer; dt::Real = 1.0)\n\nEvaluate the neuron model at time t. Return resulting membrane potential.\n\n\n\n\n\n","category":"method"},{"location":"neuron-models/#SpikingNN.reset!-Tuple{SRM0}","page":"Neuron Models","title":"SpikingNN.reset!","text":"reset!(neuron::SRM0)\nreset!(neurons::AbstractArray{<:SRM0})\n\nReset neuron.\n\n\n\n\n\n","category":"method"}]
}
