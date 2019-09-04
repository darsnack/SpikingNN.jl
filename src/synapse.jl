"""
    delta(Δ::Real; q::Real = 1)

Return 1.0 whenever Δ == 0 and 0.0 otherwise. (i.e. Dirac delta function)
"""
delta(Δ::Real; q::Real = 1.0) = (Δ == 0) ? q : zero(q)

"""
    α(Δ::Real; q::Real = 1, τ::Real = 1)

Return (q / τ) * exp(-Δ / τ) Θ(Δ) (where Θ is the Heaviside function).
"""
function α(Δ::Real; q::Real = 1.0, τ::Real = 1.0)
    v = Δ * (q / τ) * exp(-(Δ - τ) / τ)
    (Δ >= 0) ? v : zero(v)
end