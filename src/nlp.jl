using RobotDynamics
using StaticArrays
using LinearAlgebra


"""
    NLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem,
with a dynamics model of type `L`, a quadratic cost function, horizon `T`, 
and initial and final state `x0`, `xf`.

# Constructor
    NLP(model, obj, tf, T, x0, xf, [integration])

# Basic Methods
    Base.size(nlp)    # returns (n,m,T)
    num_ineq(nlp)     # number of inequality constraints
    num_eq(nlp)       # number of equality constraints
    num_primals(nlp)  # number of primal variables
    num_duals(nlp)    # total number of dual variables
    packZ(nlp, X, U)  # Stacks state `X` and controls `U` into one vector `Z`

# Evaluating the NLP
The NLP supports the following API for evaluating various pieces of the NLP:

    eval_f(nlp, Z)         # evaluate the objective
    grad_f!(nlp, grad, Z)  # gradient of the objective
    hess_f!(nlp, hess, Z)  # Hessian of the objective
    eval_c!(nlp, c, Z)     # evaluate the constraints
    jac_c!(nlp, c, Z)      # constraint Jacobian
"""
struct NLP{n,m,L,Q}
    model::L                                 # dynamics model
    obj::Vector{QuadraticCost{n,m,Float64}}  # objective function
    T::Int                                   # number of knot points
    tf::Float64                              # total time (sec)
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    times::Vector{Float64}                   # vector of times
    function NLP(model::AbstractModel, obj::Vector{<:QuadraticCost{n,m}},
            tf::Real, T::Integer, x0::AbstractVector, xf::AbstractVector, integration::Type{<:QuadratureRule}=RK4
        ) where {n,m}
        mT = integration <: RobotDynamics.Explicit ? T-1 : T 
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:T]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:mT]
        times = collect(range(0, tf, length=T))
        new{n,m,typeof(model), integration}(
            model, obj,
            T, tf, x0, xf, xinds, uinds, times
        )
    end
end
Base.size(nlp::NLP{n,m}) where {n,m} = (n,m,nlp.T)
num_primals(nlp::NLP{n,m}) where {n,m} = n*nlp.T + m*num_controls(nlp)
num_duals(nlp::NLP) = num_eq(nlp) + num_ineq(nlp)
num_eq(nlp::NLP{n,m}) where {n,m} = n*nlp.T + n
num_ineq(nlp::NLP) = 0
ishs(nlp::NLP{<:Any,<:Any,<:Any,Q}) where Q = Q == HermiteSimpson
num_controls(nlp::NLP) = ishs(nlp) ? nlp.T : nlp.T - 1

"""
    packZ(nlp, X, U)

Take a vector state vectors `X` and controls `U` and stack them into a single vector Z.
"""
function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1: num_controls(nlp)
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
    ishs(nlp) || (Z[nlp.xinds[end]] = X[end])
    return Z
end

"""
    unpackZ(nlp, Z)

Take a vector of all the states and controls and return a vector of state vectors `X` and
controls `U`.
"""
function unpackZ(nlp, Z)
    X = [Z[xi] for xi in nlp.xinds]
    U = [Z[ui] for ui in nlp.uinds]
    return X, U
end

"""
    eval_f(nlp, Z)

Evaluate the objective, returning a scalar.
"""
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

"""
    grad_f!(nlp, grad, Z)

Evaluate the gradient of the objective at `Z`, storing the result in `grad`.
"""
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

"""
    hess_f!(nlp, hess, Z)

Evaluate the Hessian of the objective at `Z`, storing the result in `hess`.
Should work with `hess` sparse.
"""
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

"""
    eval_c!(nlp, c, Z)

Evaluate the equality constraints at `Z`, storing the result in `c`.
The constraints should be ordered as follows: 
1. Initial condition ``x_1 = x_\\text{init}``
2. Dynamics ``f(x_k,u_k) - x_{k+1} = 0``
3. Terminal constraint ``x_T = x_\\text{goal}``
"""
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
        if ishs(nlp) 
            u⁺ = Z[ui[k+1]]
            c[idx] = hermite_simpson(nlp.model, x, u, x⁺, u⁺, dt)
        else
            c[idx] = discrete_dynamics(Q, nlp.model, x, u, nlp.times[k], dt) - x⁺
        end
    end

    # terminal constraint
    idx = idx .+ n
    c[idx] = Z[xi[T]] - nlp.xf
    return nothing
end


"""
    jac_c!(nlp, jac, Z)

Evaluate the constraint Jacobian, storing the result in the (potentially sparse) matrix `jac`.
"""
function jac_c!(nlp::NLP{n,m,<:Any,Q}, jac, Z) where {n,m,Q}
    # TODO: Initial condition
    # SOLUTION
    for i = 1:n
        jac[i,i] = 1
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    for k = 1:nlp.T-1
        idx = idx .+ n 
        zi = [xi[k];ui[k]]
        zi2 = k < T-1 || ishs(nlp) ? zi .+ (n+m) : xi[T]
        x = Z[xi[k]]
        u = Z[ui[k]]
        t = nlp.times[k]
        dt = nlp.times[k+1] - nlp.times[k]

        ∇f = view(jac, idx, zi)
        ∇f2 = view(jac, idx, zi2)
        
        # TODO: Dynamics constraint
        if Q == HermiteSimpson
            x2,u2 = Z[xi[k+1]], Z[ui[k+1]]
            i1 = [xi[1]; ui[1]]
            i2 = [xi[2]; ui[2]]
            ∇f12 = hs_jacobian(model, x, u, x2, u2, dt)
            ∇f .= ∇f12[xi[1],i1]
            ∇f2 .= ∇f12[xi[1],i2]
        else
            discrete_jacobian!(Q, ∇f, nlp.model, x, u, t, dt)
            for i = 1:n
                ∇f2[i,i] = -1
            end
        end
    end
    idx = idx .+ n 
    
    # TODO: Terminal constraint
    # SOLUTION
    for i = 1:n
        jac[idx[i], xi[end][i]] = 1
    end
end

"""
    jvp!(nlp, jac, Z, λ)

Evaluate the constraint Jacobian-transpose vector product ``\\nabla c^T \\lambda``, storing the result in the vector `jac`.
"""
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

"""
    ∇jvp!(nlp, hess, Z, λ)

Evaluate the Jacobian of the constraint Jacobian-transpose vector product, e.g. ``\\frac{\\partial}{\\partial z} \\nabla c^T \\lambda``,
storing the result in the (potentially sparse) matrix `hess`.
"""
function ∇jvp!(nlp::NLP{n,m,<:Any,Q}, hess, Z, λ) where {n,m,Q}
    xi,ui = nlp.xinds, nlp.uinds
    idx = [xi[1]; ui[1]]
    idx2 = xi[1]
    
    # TODO: Initial Constraint
    ishs(nlp) && (hess .= 0)
    
    # Dynamics constraints
    for k = 1:nlp.T-1
        idx2 = idx2 .+ n
        zi2 = k < T-1 ? idx .+ (n+m) : xi[T]
        x = Z[xi[k]]
        u = Z[ui[k]]
        λk = λ[idx2]
        t = nlp.times[k]
        dt = nlp.times[k+1] - nlp.times[k]
        
        ∇f = view(hess, idx, idx)
        ∇f2 = view(hess, idx, zi2)
        
        # TODO: Calculate second derivative the dynamics
        # SOLUTION
        if ishs(nlp)
            ib = [idx; idx .+ (n+m)]
            ∇f = view(hess, ib, ib)
            x2,u2 = Z[xi[k+1]], Z[ui[k+1]]
            ∇f .+= hs_∇jvp(nlp.model, x, u, x2, u2, λk, dt)
        else
            ∇discrete_jacobian!(Q, ∇f, nlp.model, x, u, t, dt, λk)
        end
        
        # Advance indices
        idx = idx .+ (n + m)
    end
    # TODO: Terminal constraint
end

############################################################################################
#                                 LAGRANGIAN
############################################################################################
"""
    lagrangian(nlp, Z, λ, c)

Evaluate the Lagrangian at `Z` and `λ`. Calculates the constraints, storing the result in `c`.
"""
function lagrangian(nlp::NLP{n,m}, Z, λ, c=zeros(eltype(Z),length(λ))) where {n,m}
    J = eval_f(nlp, Z)
    eval_c!(nlp, c, Z)
    return J - dot(λ,c)
end

"""
    grad_lagrangian(nlp, grad, Z, λ)

Evaluate the gradient of the Lagrangian.
"""
function grad_lagrangian!(nlp::NLP{n,m}, grad, Z, λ, tmp=zeros(eltype(Z), n+m)) where {n,m}
    jac = spzeros(length(λ), length(Z))
    jac_c!(nlp, jac, Z)
    grad_f!(nlp, grad, Z)
    grad .-= jac'λ
    return nothing

    grad_f!(nlp, grad, Z)
    grad .*= -1
    jvp!(nlp, grad, Z, λ, false, tmp)
    grad .*= -1
    return nothing
end

"""
    hess_lagrangian(nlp, grad, Z, λ)

Evaluate the Hessian of the Lagrangian.
"""
function hess_lagrangian!(nlp::NLP{n,m}, hess, Z, λ) where {n,m}
    ∇jvp!(nlp, hess, Z, λ)
    hess .*= -1
    hess_f!(nlp, hess, Z, false)
end

"""
    primal_residual(nlp, Z, λ, [g; p])

Evaluate the `p`-norm of the primal residual (stationarity condition).
"""
function primal_residual(nlp::NLP, Z, λ, g=zeros(num_primals(nlp)); p=2)
    grad_lagrangian!(nlp, g, Z, λ)
    return norm(g, p)
end

"""
    dual_residual(nlp, Z, λ, [c; p])

Evaluate the `p`-norm of the dual residual (constraint violation).
"""
function dual_residual(nlp::NLP, Z, λ, c=zeros(num_eq(nlp)); p=2)
    eval_c!(nlp, c, Z)
    norm(c, p)
end