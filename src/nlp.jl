using RobotDynamics
using StaticArrays
using LinearAlgebra

struct QuadraticCost{n,m,T}
    Q::Diagonal{T,SVector{n,T}}
    R::Diagonal{T,SVector{m,T}}
    q::SVector{n,T}
    r::SVector{m,T}
    c::T
end

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


struct NLP{n,m,L,Q}
    model::L
    obj::Vector{QuadraticCost{n,m,Float64}}
    T::Int  # number of knot points
    tf::Float64
    x0::MVector{n,Float64}  # initial condition
    xf::MVector{n,Float64}  # final condition
    xinds::Vector{SVector{n,Int}}
    uinds::Vector{SVector{m,Int}}
    times::Vector{Float64}
    function NLP(model::AbstractModel, obj::Vector{<:QuadraticCost{n,m}},
            tf::Real, T::Integer, x0::AbstractVector, xf::AbstractVector, integration::Type{<:QuadratureRule}=RK4
        ) where {n,m}
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:T]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:T-1]
        times = collect(range(0, tf, length=T))
        new{n,m,typeof(model), integration}(
            model, obj,
            T, tf, x0, xf, xinds, uinds, times
        )
    end
end
Base.size(nlp::NLP{n,m}) where {n,m} = (n,m,nlp.T)
num_primals(nlp::NLP{n,m}) where {n,m} = n*nlp.T + m*(nlp.T-1)
num_duals(nlp::NLP) = num_eq(nlp) + num_ineq(nlp)
num_eq(nlp::NLP{n,m}) where {n,m} = n*nlp.T + n
num_ineq(nlp::NLP) = 0

function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1:nlp.T-1
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
    Z[nlp.xinds[end]] = X[end]
    return Z
end

function eval_f(nlp::NLP, Z)
    J = 0.0
    xi,ui = nlp.xinds, nlp.uinds
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        J += stagecost(nlp.obj[k], x, u)
    end
    J += termcost(nlp.obj[end], Z[xi[end]])
    return J
end

function grad_f!(nlp::NLP{n,m}, grad, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        grad[xi[k]] = obj[k].Q*x + obj[k].q
        grad[ui[k]] = obj[k].R*u + obj[k].r
    end
    grad[xi[end]] = obj[end].Q*Z[xi[end]] + obj[end].q
    return nothing
end

function hess_f!(nlp::NLP{n,m}, hess, Z, rezero=true) where {n,m}
    if rezero
        for i = 1:size(hess,1)
            hess[i,i] = 0
        end
    end
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    i = 1
    for k = 1:nlp.T
        for j = 1:n
            hess[i,i] += nlp.obj[k].Q[j,j]
            i += 1
        end
        if k < nlp.T
            for j = 1:m
                hess[i,i] += nlp.obj[k].R[j,j]
                i += 1
            end
        end
    end
end

function eval_c!(nlp::NLP{n,m,<:Any,Q}, c, Z) where {n,m,Q}
    T = nlp.T
    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]

    # initial condition
    c[idx] = Z[xi[1]] - nlp.x0

    # dynamics
    for k = 1:T-1
        idx = idx .+ n
        x,u = Z[xi[k]], Z[ui[k]]
        x⁺ = Z[xi[k+1]]
        dt = nlp.times[k+1] - nlp.times[k]
        c[idx] = discrete_dynamics(Q, nlp.model, x, u, nlp.times[k], dt) - x⁺
    end

    # terminal constraint
    idx = idx .+ n
    c[idx] = Z[xi[T]] - nlp.xf
    return nothing
end

function jac_c!(nlp::NLP{n,m,<:Any,Q}, jac, Z) where {n,m,Q}
    for i = 1:n
        jac[i,i] = 1
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    for k = 1:nlp.T-1
        idx = idx .+ n 
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        J = view(jac,idx,zi)
        discrete_jacobian!(Q, J, nlp.model, z, nothing)
        for i = 1:n
            jac[idx[i], zi[end]+i] = -1
        end
    end
    idx = idx .+ n 
    for i = 1:n
        jac[idx[i], xi[end][i]] = 1
    end
end

function jvp!(nlp::NLP{n,m,<:Any,Q}, jac, Z, λ, rezero::Bool=true, tmp=zeros(n+m)) where {n,m,Q}
    for i = 1:n
        rezero && (jac[i] = 0)
        jac[i] += λ[i]
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = [xi[1]; ui[1]]
    idx2 = xi[1]
    for k = 1:nlp.T-1
        idx2 = idx2 .+ n
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        λ_ = λ[idx2]
        RobotDynamics.discrete_jvp!(Q, tmp, nlp.model, z, λ_)
        rezero && (jac[idx[end]] = 0)
        jac[idx] += tmp 

        idx = idx .+ (n + m)
        for i = 1:n
            rezero && (jac[idx[i]] = 0)
            jac[idx[i]] += -λ_[i]
        end
    end
    λT = λ[idx2 .+ n]
    for i = 1:n
        jac[idx[i]] += λT[i]
    end
end

function ∇jvp!(nlp::NLP{n,m,<:Any,Q}, hess, Z, λ) where {n,m,Q}
    xi,ui = nlp.xinds, nlp.uinds
    idx = [xi[1]; ui[1]]
    idx2 = xi[1]
    for k = 1:nlp.T-1
        idx2 = idx2 .+ n
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        λ_ = λ[idx2]
        ∇f = view(hess, idx, idx)
        RobotDynamics.∇discrete_jacobian!(Q, ∇f, nlp.model, z, λ_)
        idx = idx .+ (n + m)
    end
    for i = 1:n
        hess[end-i+1,end-i+1] = 0
    end
end

############################################################################################
#                                 LAGRANGIAN
############################################################################################
function lagrangian(nlp::NLP{n,m}, Z, λ, c=zeros(eltype(Z),length(λ))) where {n,m}
    J = eval_f(nlp, Z)
    eval_c!(nlp, c, Z)
    return J - dot(λ,c)
end

function grad_lagrangian!(nlp::NLP{n,m}, grad, Z, λ, tmp=zeros(eltype(Z), n+m)) where {n,m}
    grad_f!(nlp, grad, Z)
    grad .*= -1
    jvp!(nlp, grad, Z, λ, false, tmp)
    grad .*= -1
    return nothing
end

function hess_lagrangian!(nlp::NLP{n,m}, hess, Z, λ) where {n,m}
    ∇jvp!(nlp, hess, Z, λ)
    hess .*= -1
    hess_f!(nlp, hess, Z, false)
end

function primal_residual(nlp::NLP, Z, λ, g=zeros(num_primals(nlp)); p=2)
    grad_lagrangian!(nlp, g, Z, λ)
    return norm(g, p)
end

function dual_residual(nlp::NLP, Z, λ, c=zeros(num_eq(nlp)); p=2)
    eval_c!(nlp, c, Z)
    norm(c, p)
end