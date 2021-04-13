_checksquare(matrix) =
    (size(matrix, 1) == size(matrix, 2)) ? size(matrix, 1) :
                                           error("Connectivity (weight) matrix must be a square.")

conv_impulses(f, t, impulses::AbstractVector{<:Number}) = mapreduce(t̂ -> f(t - t̂), +, impulses)
function conv_impulses(f, t, impulses::AbstractArray{<:Number, N}; dims = N) where N
    @inline function fbuffered!(t, Δ, t̂)
        Δ .= t .- t̂
        return f(Δ)
    end
    Δ = similar(first(eachslice(impulses; dims = dims)))

    return mapreduce(t̂ -> fbuffered!(t, Δ, t̂), +, eachslice(impulses; dims = dims))
end
function conv_impulses!(f!, buffer, t, impulses::AbstractArray{<:Number, N}; dims = N) where N
    @inline function fbuffered!(t, Δ, t̂)
        Δ .= t .- t̂
        f!(buffer, Δ)
        return buffer
    end
    Δ = similar(first(eachslice(impulses; dims = dims)))

    return mapreduce(t̂ -> fbuffered!(t, Δ, t̂), +, eachslice(impulses; dims = dims))
end
# function conv_impulses(f, t, impulses::AbstractArray{<:CircularBuffer})
#     @inline function fbuffered!(t, Δ, i)
#         Δ .= t .- getindex.(impulses, i)

#         return f(Δ)
#     end
#     Δ = similar(getindex.(impulses, 1))
    
#     return mapreduce(i -> fbuffered!(t, Δ, i), +, 1:length(first(impulses)))
# end
# function conv_impulses!(f!, buffer, t, impulses::AbstractArray{<:CircularBuffer})
#     @inline function fbuffered!(t, Δ, i)
#         Δ .= t .- getindex.(impulses, i)
#         f!(buffer, Δ)

#         return buffer
#     end
#     Δ = similar(getindex.(impulses, 1))
    
#     return mapreduce(i -> fbuffered!(t, Δ, i), +, 1:length(first(impulses)))
# end
conv_impulses(f, t, impulses::CircularArray) =
    conv_impulses(f, t, impulses.buffer)
conv_impulses!(f!, buffer, t, impulses::CircularArray) =
    conv_impulses!(f!, buffer, t, impulses.buffer)
conv_impulses(f, t, impulses::ArrayOfCircularVectors) =
    conv_impulses(f, t, impulses.buffer)
conv_impulses!(f!, buffer, t, impulses::ArrayOfCircularVectors) =
    conv_impulses!(f!, buffer, t, impulses.buffer)
