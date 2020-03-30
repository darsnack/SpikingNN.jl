abstract type AbstractCell end

_errorstring(x, f) = """
    x is $(typeof(x)) <: AbstractCell but $f(...) is not implemented.
    If this is a custom cell body, then maybe you forgot to implement the interface?
    """

getvoltage(x::AbstractCell) = error(_errorstring(x, "getvoltage"))
excite!(x::AbstractCell, current) = error(_errorstring(x, "excite!"))
spike!(x::AbstractCell, t::Integer; dt::Real = 1.0) = error(_errorstring(x, "spike!"))
reset!(x::AbstractCell) = error(_errorstring(x, "reset!"))
(x::AbstractCell)(t::Integer; dt::Real = 1.0) = error(_errorstring(x, "(x::$(typeof(x)))"))