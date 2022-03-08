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
    N::Int

    function TOQP(n,m,N,M,P)
        N = n*N + (N)*m
        Q = spzeros(N,N)
        q = zeros(N)
        A = spzeros(M,N)
        b = zeros(M) 
        C = spzeros(P,N)
        l = fill(-Inf,P)
        u = fill(Inf,P)

        new(Q,q,A,b,C,l,u,n,m,N)
    end
end


function TOQP(nlp::NLP{n,m}) where {n,m}
    TOQP(n,m,nlp.N, num_eq(nlp), num_ineq(nlp))
end

num_ineq(qp::TOQP) = length(qp.l)
num_eq(qp::TOQP) = length(qp.b)
num_primals(qp::TOQP) = length(qp.q)
num_duals(qp::TOQP) = num_ineq(qp) + num_eq(qp)


# TASK: Complete the following method to build the QP sub-problem
"""
    build_qp!(qp, nlp, Z, λ; [gn=true])

Build a QP from the NLP, evaluated at primal variables `Z` and dual variables `λ`, 
optionally using either the Hessian of the cost function (`gn = true`) or the Hessian of the Lagrangian (`gn = false`).
"""
function build_qp!(qp::TOQP, nlp::NLP, Z, λ; gn::Bool=true)
    # TODO: Build the qp, filling in qp.Q, qp.q, qp.A, qp.b
    jac_c!(nlp, qp.A, Z)
    eval_c!(nlp, qp.b, Z)
    qp.b .*= -1  # reverse sign
    # grad_lagrangian!(nlp, qp.q, Z, λ)
    grad_f!(nlp, qp.q, Z)

    if gn
        hess_f!(nlp, qp.Q, Z)
    else
        hess_lagrangian!(nlp, qp.Q, Z, λ)
    end
    return nothing
end


# TASK: Complete the function to solve the QP
"""
    solve_qp!(qp, [reg])

Solve the QP, optionally applying regularization `reg`.
"""
function solve_qp!(qp::TOQP, reg=0.0)
    # TODO: Solve the QP sub-problem
    # HINT: Form the KKT system and solve with a single linear solve
    N,M = num_primals(qp), num_duals(qp)
    
    # SOLUTION
    K = [qp.Q + reg*I qp.A'; qp.A -reg*I]
    t = [-qp.q; qp.b]
    dY = K\t
    dZ = dY[1:N]
    dλ = dY[N+1:N+M]
    return dZ, dλ
end

function solve_osqp!(qp::TOQP)
    model = OSQP.Model()
    OSQP.setup!(model, P=qp.Q, q=qp.q, A=qp.A, l=qp.b, u=qp.b)
    res = OSQP.solve!(model)
    return res.x, res.y 
end

function minimum_penalty(qp::TOQP, p)
    # σ = isposdef(qp.Q)
    σ = 1.0
    ρ = 0.5
    (qp.q'p + σ/2 * dot(p, qp.Q, p)) / ((1 - ρ) * norm(qp.b, 1))
end