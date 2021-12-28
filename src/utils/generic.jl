_checksquare(matrix) =
    (size(matrix, 1) == size(matrix, 2)) ? size(matrix, 1) :
                                           error("Connectivity (weight) matrix must be a square.")

_fillmemaybe(x::AbstractArray, sz) = x
_fillmemaybe(x, sz) = fill(x, sz)

inf(::Type{T}) where {T<:AbstractFloat} = T(Inf)
inf(::Type{T}) where {T<:Number} = inf(Float32)
inf(x::T) where T = inf(T)

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
conv_impulses(f, t, impulses::ImpulseBuffer) =
    conv_impulses(f, t, impulses.buffer)
conv_impulses(f, t, impulses::ArrayOfImpulseBuffers) =
    conv_impulses(f, t, impulses.buffer)
conv_impulses!(f!, buffer, t, impulses::ArrayOfImpulseBuffers) =
    conv_impulses!(f!, buffer, t, impulses.buffer)
