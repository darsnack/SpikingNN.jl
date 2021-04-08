_default(::Type{<:Integer}) = -1
_default(::Type{<:Real}) = -Inf

struct CircularArray{T, N, S<:AbstractArray{T, N}, I} <: AbstractArray{T, N}
    buffer::S
    first::I
    usage::I
    default::T
end

CircularArray{T}(size::NTuple{N, <:Integer}; default = _default(T)) where {T, N} =
    CircularArray{T, N, Array{T, N}, Array{Int, N - 1}}(fill(default, size...),
                                                        ones(Int, size[1:(end - 1)]),
                                                        zeros(Int, size[1:(end - 1)]),
                                                        default)
CircularArray{T}(size::Vararg{<:Integer, N}; default = _default(T)) where {T, N} =
    CircularArray{T}(size; default = default)

Base.eltype(::CircularArray{T}) where T = T
Base.size(A::CircularArray) = size(A.buffer)
capacity(A::CircularArray) = size(A)[end]
capacity(A::SubArray{<:Any, <:Any, <:CircularArray}) = capacity(parent(A))
Base.firstindex(A::CircularArray) = A.first
Base.firstindex(A::SubArray{<:Any, <:Any, <:CircularArray}) =
    view(firstindex(parent(A)), parentindices(A)[1:(end - 1)]...)
usage(A::CircularArray) = A.usage
usage(A::SubArray{<:Any, <:Any, <:CircularArray}) =
    view(usage(parent(A)), parentindices(A)[1:(end - 1)]...)

function Base.similar(A::CircularArray{T, N, S, I}, ::Type{R}, dims::Dims) where {T, N, S, I, R}
    default = convert(R, A.default)
    buffer = fill(default, dims)
    first = ones(eltype(I), dims)
    usage = zeros(eltype(I), dims)

    return CircularArray{R, length(dims), typeof(buffer), typeof(first)}(buffer, first, usage, default)
end

function Adapt.adapt_structure(to, A::CircularArray)
    buffer = adapt(to, A.buffer)
    first = adapt(to, A.first)
    usage = adapt(to, A.usage)

    T = eltype(buffer)
    N = ndims(buffer)
    S = typeof(buffer)
    I = typeof(first)

    return CircularArray{T, N, S, I}(buffer, first, usage, A.default)
end

Base.axes(A::CircularArray) = axes(A.buffer)

@inline function _buffer_index(A::CircularArray{<:Any, N}, I::NTuple{N, <:Integer}) where N
    ibuffer = mod1.(A.first[I[1:(end - 1)]...] + I[end] - 1, capacity(A))
    
    return ntuple(i -> (i == N) ? ibuffer : I[i], N)
end
@inline _buffer_index(A::CircularArray{<:Any, N}, I::CartesianIndex{N}) where N =
    _buffer_index(A, Tuple(I))
@inline function _buffer_index(A::CircularArray{<:Any, N}, I) where N
    C = CartesianIndices(ntuple(i -> (I[i] isa Colon) ? axes(A, i) :
                                     (I[i] isa AbstractRange) ? I[i] : (I[i]:I[i]), N))

    return CartesianIndices(map(c -> _buffer_index(A, c), C))
end

@inline Base.getindex(A::CircularArray{<:Any, N}, I::Vararg{Int, N}) where N =
    A.buffer[_buffer_index(A, I)...]

@inline Base.getindex(A::CircularArray{<:Any, N}, I::Vararg{<:Any, N}) where N =
    A.buffer[_buffer_index(A, I)]

@inline function Base.setindex!(A::CircularArray{<:Any, N}, x, I::Vararg{Int, N}) where N
    @boundscheck if I[end] < 1 || I[end] > usage(A)[I[1:(end - 1)]...]
        throw(BoundsError(A, I))
    end

    A.buffer[_buffer_index(A, I)...] = x
end

_cartesian(I) = [CartesianIndex(c, I[c]) for c in CartesianIndices(I)]

function Base.push!(A::CircularArray, x)
    U = usage(A)
    overflow = (U .== capacity(A))
    @. A.first += overflow
    @. A.usage += 1 - overflow
    A[_cartesian(U)] .= x

    return A
end
function Base.push!(A::SubArray{<:Any, <:Any, <:CircularArray}, x)
    @assert parentindices(A)[end] == 1:capacity(A)

    P = view(parent(A).buffer, parentindices(A)...)
    U = usage(A)
    F = firstindex(A)
    
    overflow = (U .== capacity(A))
    @. F += overflow
    @. U += 1 - overflow
    P[_cartesian(U)] .= x

    return A
end

function Base.empty!(A::CircularArray)
    A.first .= 1
    A.usage .= 0
    A.buffer .= A.default

    return A
end
function Base.empty!(A::SubArray{<:Any, <:Any, <:CircularArray})
    @assert parentindices(A)[end] == 1:capacity(A)

    is = parentindices(A)[1:(end - 1)]
    P = parent(A)
    F = firstindex(A)
    U = usage(A)
    
    F .= 1
    U .= 0
    P.buffer[is..., :] .= P.default

    return A
end

struct ArrayOfCircularVectors{T, N, S} <: AbstractArray{CircularArray{T}, N}
    buffer::S

    function ArrayOfCircularVectors{T}(size::NTuple{N, <:Integer}, capacity) where {T, N}
        buffer = CircularArray{T}((size..., capacity))
        S = typeof(buffer)
        # P = Core.Compiler.return_type(similar, Tuple{S, NTuple{1, Int}})
        
        new{T, N, S}(buffer)
    end
end
ArrayOfCircularVectors{T}(size...; capacity) where T = ArrayOfCircularVectors{T}(size, capacity)

Base.size(A::ArrayOfCircularVectors) = size(A.buffer)[1:(end - 1)]

function Base.getindex(A::ArrayOfCircularVectors{T, N}, I::Vararg{Int, N}) where {T, N}
    buffer_view = view(A.buffer.buffer, I..., :)
    first_view = view(A.buffer.first, I...)
    usage_view = view(A.buffer.usage, I...)
    n = ndims(buffer_view)
    S = typeof(buffer_view)
    I = typeof(first_view)

    CircularArray{T, n, S, I}(buffer_view, first_view, usage_view, A.buffer.default)
end

Base.push!(A::ArrayOfCircularVectors, x) = push!(A.buffer, x)
Base.push!(A::SubArray{<:CircularArray, <:Any, <:ArrayOfCircularVectors}, x) =
    push!(view(parent(A).buffer, parentindices(A)..., :), x)
Base.empty!(A::ArrayOfCircularVectors) = empty!(A.buffer)
Base.empty!(A::SubArray{<:CircularArray, <:Any, <:ArrayOfCircularVectors}) =
    empty!(view(parent(A).buffer, parentindices(A)..., :))
