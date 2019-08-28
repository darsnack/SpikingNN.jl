"""
    delta(t::Real)

Return 1.0 whenever t == 0 and 0.0 otherwise. (i.e. Dirac delta function)
"""
delta(t::Real) = (t == 0.0) ? 1.0 : 0.0