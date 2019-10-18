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