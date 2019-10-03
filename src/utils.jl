"""
    sample_response(response::Function, dt::Real = 1.0)

Sample the provided response function.
"""
function sample_response(response::Function, dt::Real = 1.0)
    N = Int(10 / dt) # number of samples to acquire (defaults to 10 samples when dt = 1.0)
    return response.(dt .* collect(1:N) .- dt), N
end

@userplot RasterPlot

@recipe function f(h::RasterPlot)
    seriestype := :scatter
    if length(h.args) == 1 && isa(h.args[1], Dict)
        ylabel := "Neuron ID"
        for (i, x) in pairs(h.args[1])
            @series begin
                if !haskey(plotattributes, :label) || isempty(plotattributes[:label])
                    label := ""
                else
                    label := plotattributes[:label][i]
                end
                x, i .* ones(length(x))
            end
        end
    else
        for (i, x) in enumerate(h.args)
            yticks := :none
            @series x, i .* ones(length(x))
        end
    end
end