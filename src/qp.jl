
@doc raw"""
    TOQP

A type specifying a (T)rajectory (O)ptimization (Q)uadratic (P)rogram, of the form


``\begin{aligned} &\text{minimize} &&\frac{1}{2} z^T Q z + q^T z \\ 
&\text{subject to} && A z = b \\ 
&&& l \leq C z \leq u \end{aligned}``

where ``z = [x_1^T \; u_1^T \; \dots \; x_{T-1}^T \; u_{T-1}^T \; x_T^T]^T`` and 
``x \in \mathbb{R}^n`` is the state vector and ``u \in \mathbb{R}^m`` is the control vector.

# Constructors

    TOQP(n,m,T,M,P)

where `n` is the number of states, `m` is the number of controls, `T` is the horizon, `M` is the number of equality 
constraints, and `P` is the number of inequality constraints.

# Methods

    num_ineq(qp)     # number of inequality constraints
    num_eq(qp)       # number of equality constraints
    num_primals(qp)  # number of primal variables
    num_duals(qp)    # total number of dual variables


"""
struct TOQP
    Q::SparseMatrixCSC{Float64,Int}  # quadratic cost
    q::Vector{Float64}               # linear cost
    A::SparseMatrixCSC{Float64,Int}  # equality constraint Ax = b
    b::Vector{Float64}               # equality constraint 
    C::SparseMatrixCSC{Float64,Int}  # inequality constraint l ≤ Cx ≤ u
    l::Vector{Float64}               # inequality constraint lower bound
    u::Vector{Float64}               # inequality constraint upper bound
    n::Int
    m::Int
    T::Int

    function TOQP(n,m,T,M,P)
        N = n*T + (T-1)*m
        Q = spzeros(N,N)
        q = zeros(N)
        A = spzeros(M,N)
        b = zeros(M) 
        C = spzeros(P,N)
        l = fill(-Inf,P)
        u = fill(Inf,P)

        new(Q,q,A,b,C,l,u,n,m,T)
    end
end


function TOQP(nlp::NLP{n,m}) where {n,m}
    TOQP(n,m,nlp.T, num_eq(nlp), num_ineq(nlp))
end

num_ineq(qp::TOQP) = length(qp.l)
num_eq(qp::TOQP) = length(qp.b)
num_primals(qp::TOQP) = length(qp.q)
num_duals(qp::TOQP) = num_ineq(qp) + num_eq(qp)


"""
    build_qp!(qp, nlp, Z, λ; [gn=true])

Build a QP from the NLP, evaluated at primal variables `Z` and dual variables `λ`, 
optionally using either the Hessian of the cost function (`gn = true`) or the Hessian of the Lagrangian (`gn = false`).
"""
function build_qp!(qp::TOQP, nlp::NLP, Z, λ; gn::Bool=true) 
    jac_c!(nlp, qp.A, Z)
    eval_c!(nlp, qp.b, Z)
    qp.b .*= -1  # reverse sign
    grad_lagrangian!(nlp, qp.q, Z, λ)

    if gn
        hess_f!(nlp, qp.Q, Z)
    else
        hess_lagrangian!(nlp, qp.Q, Z, λ)
    end
    return nothing
end

"""
    solve_qp!(qp, [reg])

Solve the QP, optionally applying regularization `reg`.
"""
function solve_qp!(qp::TOQP, reg=0.0)
    N,M = num_primals(qp), num_duals(qp)
    K = [qp.Q + reg*I qp.A'; qp.A -reg*I]
    t = [-qp.q; qp.b]
    dY = K\t
    dZ = dY[1:N]
    dλ = dY[N+1:N+M]
    return dZ, dλ
end