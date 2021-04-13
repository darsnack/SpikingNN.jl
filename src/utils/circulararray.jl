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

@inline function _buffer_index_raw(F, C, I::NTuple{1, <:Integer})
    ibuffer = mod1.(F .+ I[end] .- 1, C)
    
    return (ibuffer...,)
end
@inline function _buffer_index_raw(F, C, I::NTuple{N, <:Integer}) where N
    ibuffer = mod1.(F[I[1:(end - 1)]...] + I[end] - 1, C)
    
    return ntuple(i -> (i == N) ? ibuffer : I[i], N)
end
@inline _buffer_index_raw(F, C, I::CartesianIndex) = _buffer_index_raw(F, C, Tuple(I))

@inline _buffer_index(A::CircularArray{<:Any, N}, I::NTuple{N, <:Integer}) where N =
    _buffer_index_raw(firstindex(A), capacity(A), I)
@inline _buffer_index(A::CircularArray{<:Any, N}, I::CartesianIndex{N}) where N =
    CartesianIndex(_buffer_index(A, Tuple(I)))
@inline _buffer_index(A::CircularArray, I::AbstractArray{<:CartesianIndex}) =
    CartesianIndex.(_buffer_index.(Ref(A), I))
@inline function _buffer_index(A::CircularArray{<:Any, N}, I) where N
    C = CartesianIndices(ntuple(i -> (I[i] isa Colon) ? axes(A, i) :
                                     (I[i] isa AbstractRange) ? I[i] : (I[i]:I[i]), N))

    return map(c -> _buffer_index(A, c), C)
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

_cartesian(I::AbstractArray{<:Any, 0}) = [CartesianIndex((I...))]
_cartesian(I) = CartesianIndex.(CartesianIndices(I), I)
# _view_cartesian(parent_indices, first_indices, capacity, indices) =
#     CartesianIndex.(Base.reindex.(Ref(parent_indices), Tuple.(_cartesian(indices))))
function _view_cartesian(parent_indices,
                         first_indices,
                         buffer_capacity,
                         indices)
    I = _buffer_index_raw.(Ref(first_indices), buffer_capacity, _cartesian(indices))
    
    return CartesianIndex.(Base.reindex.(Ref(parent_indices), I))
end

function Base.push!(A::CircularArray, x)
    U = usage(A)
    overflow = (U .== capacity(A))
    @. A.first += overflow
    @. A.usage += 1 - overflow
    indices = _buffer_index(A, _cartesian(U))
    A.buffer[indices] .= x

    return A
end
function Base.push!(A::SubArray{<:Any, <:Any, <:CircularArray}, x)
    @assert parentindices(A)[end] == 1:capacity(A)

    P = parent(A)
    U = usage(A)
    F = firstindex(A)
    
    overflow = (U .== capacity(A))
    @. F += overflow
    @. U += 1 - overflow
    indices = _view_cartesian(parentindices(A), F, capacity(A), U)
    P.buffer[indices] .= x

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

    function ArrayOfCircularVectors{T, N, S}(buffer::S) where {T, N, S}
        if (ndims(buffer) != N + 1) || !(buffer isa CircularArray{T})
            throw(ArgumentError("ArrayOfCircularVectors{$T, $N} expects a buffer of type CircularArray{$T, $(N + 1)}"))
        end

        new{T, N, S}(buffer)
    end
end
function ArrayOfCircularVectors{T}(size::NTuple{N, <:Integer}, capacity) where {T, N}
    buffer = CircularArray{T}(size..., capacity)
    S = typeof(buffer)

    ArrayOfCircularVectors{T, N, S}(buffer)
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

function Adapt.adapt_structure(to, A::ArrayOfCircularVectors{T, N}) where {T, N}
    buffer = adapt(to, A.buffer)
    S = typeof(buffer)

    return ArrayOfCircularVectors{T, N, S}(buffer)
end

Base.push!(A::ArrayOfCircularVectors, x) = push!(A.buffer, x)
Base.push!(A::SubArray{<:CircularArray, <:Any, <:ArrayOfCircularVectors}, x) =
    push!(view(parent(A).buffer, parentindices(A)..., :), x)
Base.empty!(A::ArrayOfCircularVectors) = empty!(A.buffer)
Base.empty!(A::SubArray{<:CircularArray, <:Any, <:ArrayOfCircularVectors}) =
    empty!(view(parent(A).buffer, parentindices(A)..., :))
