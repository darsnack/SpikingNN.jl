_default(::Type{<:Integer}) = -1
_default(::Type{<:Real}) = -Inf

@inline function _buffer_index_raw(head, capacity, I::NTuple{1})
    ibuffer = mod1.(head .+ last(I) .- 1, capacity)
    
    return (ibuffer...,)
end
@inline function _buffer_index_raw(head, capacity, I::NTuple{N}) where N
    ibuffer = mod1.(head[Base.front(I)...] + last(I) - 1, capacity)
    
    return ntuple(i -> (i == N) ? ibuffer : I[i], N)
end
@inline _buffer_index_raw(head, capacity, I::CartesianIndex) =
    _buffer_index_raw(head, capacity, Tuple(I))

mutable struct ImpulseBuffer{T} <: AbstractVector{T}
    buffer::Vector{T}
    head::Int
    usage::Int
    default::T
end
ImpulseBuffer{T}(N; default = _default(T)) where T = 
    ImpulseBuffer{T}(fill(default, N), one(Int), zero(Int), default)

Base.eltype(::ImpulseBuffer{T}) where T = T
Base.size(A::ImpulseBuffer) = size(A.buffer)
headptr(A::ImpulseBuffer) = A.head
usage(A::ImpulseBuffer) = A.usage
capacity(A::ImpulseBuffer) = length(A)

@inline _buffer_index(A::ImpulseBuffer, i::Int) = _buffer_index_raw(headptr(A), capacity(A), (i,))

@inline Base.getindex(A::ImpulseBuffer, i::Int) = A.buffer[_buffer_index(A, i)...]

@inline function Base.setindex!(A::ImpulseBuffer, x, i::Int)
    @boundscheck if i < 1 || i > usage(A)
        throw(BoundsError(A, i))
    end

    A.buffer[_buffer_index(A, i)...] = x
end

function Base.push!(A::ImpulseBuffer, x)
    overflow = (usage(A) == capacity(A))
    A.head += overflow
    A.usage += 1 - overflow
    indices = _buffer_index(A, usage(A))
    A.buffer[indices...] = x

    return A
end

function Base.empty!(A::ImpulseBuffer)
    A.head = 1
    A.usage = 0
    A.buffer .= A.default

    return A
end

struct ArrayOfImpulseBuffers{T, N, S, I} <: AbstractArray{ImpulseBuffer{T}, N}
    buffer::S
    head::I
    usage::I
    default::T

    function ArrayOfImpulseBuffers{T, N, S, I}(buffer, head, usage, default) where {T, N, S, I}
        @assert (ndims(S) == N + 1) "Cannot create ArrayOfCircularVectors{$T, $N} with buffer $S (need ndims(buffer) == $N)"

        new{T, N, S, I}(buffer, head, usage, default)
    end
end
ArrayOfImpulseBuffers{T}(sz::NTuple{N, <:Integer}; capacity, default = _default(T)) where {T, N} =
    ArrayOfImpulseBuffers{T, N, Array{T, N + 1}, Array{Int, N}}(fill(default, sz..., capacity),
                                                                ones(Int, sz),
                                                                zeros(Int, sz),
                                                                default)
ArrayOfImpulseBuffers{T}(sz::Vararg{<:Integer, N}; kwargs...) where {T, N} =
    ArrayOfImpulseBuffers{T}(sz; kwargs...)
function ArrayOfImpulseBuffers(buffers::AbstractArray{<:ImpulseBuffer, N}) where N
    default = first(buffers).default
    A = ArrayOfImpulseBuffers{typeof(default)}(size(buffers);
                                               capacity = capacity(first(buffers)),
                                               default = default)
    A .= buffers

    return A
end

Base.eltype(::ArrayOfImpulseBuffers{T}) where T = T
Base.size(A::ArrayOfImpulseBuffers) = Base.front(size(A.buffer))
# Base.axes(A::ArrayOfImpulseBuffers) = Base.front(axes(A.buffer))
headptr(A::ArrayOfImpulseBuffers) = A.head
headptr(A::SubArray{<:Any, <:Any, <:ArrayOfImpulseBuffers}) = view(headptr(parent(A)), parentindices(A)...)
usage(A::ArrayOfImpulseBuffers) = A.usage
usage(A::SubArray{<:Any, <:Any, <:ArrayOfImpulseBuffers}) = view(usage(parent(A)), parentindices(A)...)
capacity(A::ArrayOfImpulseBuffers) = last(size(A.buffer))
capacity(A::SubArray{<:Any, <:Any, <:ArrayOfImpulseBuffers}) = capacity(parent(A))

function Base.similar(A::ArrayOfImpulseBuffers{T, N, S, I}, ::Type{R}, dims::Dims) where {T, N, S, I, R}
    default = convert(eltype(R), A.default)
    buffer = fill(default, dims..., capacity(A))
    head = ones(eltype(I), dims)
    usage = zeros(eltype(I), dims)

    _T = eltype(R)
    _N = length(dims)
    _S = typeof(buffer)
    _I = typeof(head)

    return ArrayOfImpulseBuffers{_T, _N, _S, _I}(buffer, head, usage, default)
end

function Adapt.adapt_structure(to, A::ArrayOfImpulseBuffers)
    buffer = adapt(to, A.buffer)
    head = adapt(to, A.head)
    usage = adapt(to, A.usage)

    T = eltype(buffer)
    N = ndims(buffer) - 1
    S = typeof(buffer)
    I = typeof(head)

    return ArrayOfImpulseBuffers{T, N, S, I}(buffer, head, usage, A.default)
end

@inline _buffer_index(A::ArrayOfImpulseBuffers, I::NTuple{<:Any, <:Integer}) =
    _buffer_index_raw(headptr(A), capacity(A), I)    

@inline _buffer_index_cartesian(A::ArrayOfImpulseBuffers, I::NTuple{<:Any, <:Integer}) =
    CartesianIndex(_buffer_index(A, I))
@inline function _buffer_index_cartesian(A::ArrayOfImpulseBuffers, I)
    Is = ntuple(length(I)) do i
        (I[i] isa Colon) ? axes(A.buffer, i) :
        (I[i] isa AbstractRange) ? I[i] :
        (I[i]:I[i])
    end

    return map(i -> _buffer_index_cartesian(A, i), Base.Iterators.product(Is...))
end
@inline _buffer_index_cartesian(A::ArrayOfImpulseBuffers, I::CartesianIndex) =
    _buffer_index_cartesian(A, Tuple(I))
@inline _buffer_index_cartesian(A::ArrayOfImpulseBuffers, I::AbstractArray{<:CartesianIndex}) =
    _buffer_index_cartesian.(Ref(A), I)

@inline _buffer_index_slice(A::ArrayOfImpulseBuffers{<:Any, N}, I::NTuple{N, <:Integer}) where N =
    circshift(1:capacity(A), -(headptr(A)[I...] - 1))

@inline function Base.getindex(A::ArrayOfImpulseBuffers{T, N}, I::Vararg{<:Integer, N}) where {T, N}
    H = headptr(A)[I...]
    U = usage(A)[I...]
    C = capacity(A)
    slice = A.buffer[I..., :]
    
    return ImpulseBuffer{T}(slice, H, U, A.default)
end

@inline function Base.setindex!(A::ArrayOfImpulseBuffers{<:Any, N}, x, I::Vararg{<:Integer, N}) where N
    A.buffer[I..., _buffer_index_slice(A, I)] .= x
end
@inline function Base.setindex!(A::ArrayOfImpulseBuffers{<:Any, N},
                                x::ImpulseBuffer,
                                I::Vararg{<:Integer, N}) where N
    A.head[I...] = headptr(x)
    A.usage[I...] = usage(x)
    A.buffer[I..., :] .= x.buffer
end

_cartesian(I) = CartesianIndex.(CartesianIndices(I), I)
# _view_cartesian(parent_indices, first_indices, capacity, indices) =
#     CartesianIndex.(Base.reindex.(Ref(parent_indices), Tuple.(_cartesian(indices))))
function _view_cartesian(parent_indices, head, capacity, indices)
    I = _buffer_index_raw.(Ref(head), capacity, _cartesian(indices))
    
    return CartesianIndex.(Base.reindex.(Ref(parent_indices), I))
end

function Base.push!(A::ArrayOfImpulseBuffers, x)
    U = usage(A)
    overflow = (U .== capacity(A))
    @. A.head += overflow
    @. A.usage += 1 - overflow
    indices = _buffer_index_cartesian(A, _cartesian(U))
    A.buffer[indices] .= x

    return A
end
function Base.push!(A::SubArray{<:Any, <:Any, <:ArrayOfImpulseBuffers}, x)
    P = parent(A)
    PI = parentindices(A)
    U = usage(A)
    H = headptr(A)
    
    overflow = (U .== capacity(A))
    @. H += overflow
    @. U += 1 - overflow
    indices = _view_cartesian((PI..., 1:capacity(A)), H, capacity(A), U)
    P.buffer[indices] .= x

    return A
end

function Base.empty!(A::ArrayOfImpulseBuffers)
    A.head .= 1
    A.usage .= 0
    A.buffer .= A.default

    return A
end
function Base.empty!(A::SubArray{<:Any, <:Any, <:ArrayOfImpulseBuffers})
    P = parent(A)
    H = headptr(A)
    U = usage(A)
    
    H .= 1
    U .= 0
    P.buffer[parentindices(A)..., :] .= P.default

    return A
end

# struct CircularArray{T, N, S<:AbstractArray{T, N}, I} <: AbstractArray{T, N}
#     buffer::S
#     first::I
#     usage::I
#     default::T
# end

# CircularArray{T}(size::NTuple{N, <:Integer}; default = _default(T)) where {T, N} =
#     CircularArray{T, N, Array{T, N}, Array{Int, N - 1}}(fill(default, size...),
#                                                         ones(Int, Base.front(size)),
#                                                         zeros(Int, Base.front(size)),
#                                                         default)
# CircularArray{T}(size::Vararg{<:Integer, N}; default = _default(T)) where {T, N} =
#     CircularArray{T}(size; default = default)

# Base.eltype(::CircularArray{T}) where T = T
# Base.size(A::CircularArray) = size(A.buffer)
# capacity(A::CircularArray) = last(size(A))
# capacity(A::SubArray{<:Any, <:Any, <:CircularArray}) = capacity(parent(A))
# Base.firstindex(A::CircularArray) = A.first
# Base.firstindex(A::SubArray{<:Any, <:Any, <:CircularArray}) =
#     view(firstindex(parent(A)), Base.front(parentindices(A))...)
# usage(A::CircularArray) = A.usage
# usage(A::SubArray{<:Any, <:Any, <:CircularArray}) =
#     view(usage(parent(A)), Base.front(parentindices(A))...)

# function Base.similar(A::CircularArray{T, N, S, I}, ::Type{R}, dims::Dims) where {T, N, S, I, R}
#     default = convert(R, A.default)
#     buffer = fill(default, dims)
#     first = ones(eltype(I), dims)
#     usage = zeros(eltype(I), dims)

#     return CircularArray{R, length(dims), typeof(buffer), typeof(first)}(buffer, first, usage, default)
# end

# function Adapt.adapt_structure(to, A::CircularArray)
#     buffer = adapt(to, A.buffer)
#     first = adapt(to, A.first)
#     usage = adapt(to, A.usage)

#     T = eltype(buffer)
#     N = ndims(buffer)
#     S = typeof(buffer)
#     I = typeof(first)

#     return CircularArray{T, N, S, I}(buffer, first, usage, A.default)
# end

# Base.axes(A::CircularArray) = axes(A.buffer)



# @inline Base.getindex(A::CircularArray{<:Any, N}, I::Vararg{Int, N}) where N =
#     A.buffer[_buffer_index(A, I)...]

# @inline Base.getindex(A::CircularArray{<:Any, N}, I::Vararg{<:Any, N}) where N =
#     A.buffer[_buffer_index(A, I)]

# @inline function Base.setindex!(A::CircularArray{<:Any, N}, x, I::Vararg{Int, N}) where N
#     @boundscheck if last(I) < 1 || last(I) > usage(A)[Base.front(I)...]
#         throw(BoundsError(A, I))
#     end

#     A.buffer[_buffer_index(A, I)...] = x
# end



# function Base.push!(A::CircularArray, x)
#     U = usage(A)
#     overflow = (U .== capacity(A))
#     @. A.first += overflow
#     @. A.usage += 1 - overflow
#     indices = _buffer_index(A, _cartesian(U))
#     A.buffer[indices] .= x

#     return A
# end
# function Base.push!(A::SubArray{<:Any, <:Any, <:CircularArray}, x)
#     @assert parentindices(A)[end] == 1:capacity(A)

#     P = parent(A)
#     U = usage(A)
#     F = firstindex(A)
    
#     overflow = (U .== capacity(A))
#     @. F += overflow
#     @. U += 1 - overflow
#     indices = _view_cartesian(parentindices(A), F, capacity(A), U)
#     P.buffer[indices] .= x

#     return A
# end

# function Base.empty!(A::CircularArray)
#     A.first .= 1
#     A.usage .= 0
#     A.buffer .= A.default

#     return A
# end
# function Base.empty!(A::SubArray{<:Any, <:Any, <:CircularArray})
#     @assert last(parentindices(A)) == 1:capacity(A)

#     is = Base.front(parentindices(A))
#     P = parent(A)
#     F = firstindex(A)
#     U = usage(A)
    
#     F .= 1
#     U .= 0
#     P.buffer[is..., :] .= P.default

#     return A
# end

# struct ArrayOfCircularVectors{T, N, S, P} <: AbstractArray{P, N}
#     buffer::S

#     function ArrayOfCircularVectors{T, N, S}(buffer::S) where {T, N, S}
#         if (ndims(buffer) != N + 1) || !(buffer isa CircularArray{T})
#             throw(ArgumentError("ArrayOfCircularVectors{$T, $N} expects a buffer of type CircularArray{$T, $(N + 1)}"))
#         end
#         # T_buffer = Core.Compiler.return_type(typeof(S.buffer)
#         # T_indices = typeof(usage(S))
#         # P = Core.Compiler.return_type(view, Tuple{S, Int, Int, Colon})
#         P = Core.Compiler.return_type(similar, Tuple{S, NTuple{1, Int}})

#         new{T, N, S, P}(buffer)
#     end
# end
# function ArrayOfCircularVectors{T}(size::NTuple{N, <:Integer}, capacity) where {T, N}
#     buffer = CircularArray{T}(size..., capacity)
#     S = typeof(buffer)

#     ArrayOfCircularVectors{T, N, S}(buffer)
# end
# ArrayOfCircularVectors{T}(size...; capacity) where T = ArrayOfCircularVectors{T}(size, capacity)

# Base.size(A::ArrayOfCircularVectors) = Base.front(size(A.buffer))

# function _unsafe_view(A, i::Int)


# function Base.getindex(A::ArrayOfCircularVectors{T, N}, I::Vararg{Int, N}) where {T, N}
#     # buffer_view = view(A.buffer.buffer, I..., :)
#     # first_view = view(A.buffer.first, I...)
#     # usage_view = view(A.buffer.usage, I...)
#     # n = ndims(buffer_view)
#     # S = typeof(buffer_view)
#     # I = typeof(first_view)

#     # CircularArray{T, n, S, I}(buffer_view, first_view, usage_view, A.buffer.default)
#     buffer_view = Array(view(A.buffer.buffer, I..., :))
#     first_view = Array(view(A.buffer.first, I...))
#     usage_view = Array(view(A.buffer.usage, I...))
#     n = ndims(buffer_view)
#     S = typeof(buffer_view)
#     I = typeof(first_view)

#     CircularArray{T, n, S, I}(buffer_view, first_view, usage_view, A.buffer.default)
# end
# # Base.getindex(A::ArrayOfCircularVectors{T, N}, I::Vararg{Int, N}) where {T, N} = view(A.buffer, I..., :)

# function Adapt.adapt_structure(to, A::ArrayOfCircularVectors{T, N}) where {T, N}
#     buffer = adapt(to, A.buffer)
#     S = typeof(buffer)

#     return ArrayOfCircularVectors{T, N, S}(buffer)
# end

# Base.push!(A::ArrayOfCircularVectors, x) = push!(A.buffer, x)
# Base.push!(A::SubArray{<:CircularArray, <:Any, <:ArrayOfCircularVectors}, x) =
#     push!(view(parent(A).buffer, parentindices(A)..., :), x)
# Base.empty!(A::ArrayOfCircularVectors) = empty!(A.buffer)
# Base.empty!(A::SubArray{<:CircularArray, <:Any, <:ArrayOfCircularVectors}) =
#     empty!(view(parent(A).buffer, parentindices(A)..., :))
