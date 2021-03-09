using StaticArrays, LinearAlgebra

@doc raw"""
    QuadraticCost

Represents a stage cost function of the form

```math
\frac{1}{2} x^T Q x + q^T x + \frac{1}{2} u^T R u + r^T u + c
```

# Methods
    stagecost(cost, x, u)  # evaluate the stage cost at x,u
    termcost(cost, x)      # evaluate the terminal cost (assumes u is zero)
"""
struct QuadraticCost{n,m,T}
    Q::Diagonal{T,SVector{n,T}}
    R::Diagonal{T,SVector{m,T}}
    q::SVector{n,T}
    r::SVector{m,T}
    c::T
end

@doc raw"""
    LQRCost(Q, R, xf, uf)

Creates a `QuadraticCost` cost function of the form:

```math
\frac{1}{2} (x^T - x_f) Q (x - x_f) + \frac{1}{2} (u^T - u_f) R (u - u_f)
```
"""
function LQRCost(Q::AbstractMatrix, R::AbstractMatrix, xf, uf=zeros(size(R,1)))
    n,m = length(xf), length(uf)
    Q = Diagonal(SVector{n}(diag(Q)))
    R = Diagonal(SVector{m}(diag(R)))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'Q*xf + 0.5*uf'R*uf
    T = promote_type(eltype(Q), eltype(R), eltype(xf), eltype(uf))
    QuadraticCost{n,m,T}(Q, R, SVector{n}(q), SVector{m}(r), c)
end

function stagecost(cost::QuadraticCost, x, u)
    Q,R,q,r,c = cost.Q, cost.R, cost.q, cost.r, cost.c
    return 0.5*x'Q*x + q'x + 0.5*u'R*u + r'u + c
end

function termcost(cost::QuadraticCost, x, u=nothing)
    Q,R,q,r,c = cost.Q, cost.R, cost.q, cost.r, cost.c
    return 0.5*x'Q*x + q'x + c 
end

function cost(obj::Vector{<:QuadraticCost{n,m,T}}, X, U) where {n,m,T}
    J = zero(T)
    for k = 1:length(U)
        J += stagecost(obj[k], X[k], U[k])
    end
    J += termcost(obj[end], X[end])
    return J
end