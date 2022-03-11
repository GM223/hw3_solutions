"""
    NLP{n,m,L,Q}

Represents a (N)on(L)inear (P)rogram of a trajectory optimization problem,
with a dynamics model of type `L`, a quadratic cost function, horizon `T`, 
and initial and final state `x0`, `xf`.

The kth state and control can be extracted from the concatenated state vector `Z` using
`Z[nlp.xinds[k]]`, and `Z[nlp.uinds[k]]`.

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
struct NLP{n,m,L} <: MOI.AbstractNLPEvaluator
    model::L                                 # dynamics model
    stagecost::QuadraticCost{n,m,Float64}    # stage cost function
    termcost::QuadraticCost{n,m,Float64}     # terminal cost function
    N::Int                                   # number of knot points
    tf::Float64                              # total time (sec)
    x0::MVector{n,Float64}                   # initial condition
    xf::MVector{n,Float64}                   # final condition
    xinds::Vector{SVector{n,Int}}            # Z[xinds[k]] gives states for time step k
    uinds::Vector{SVector{m,Int}}            # Z[uinds[k]] gives controls for time step k
    times::Vector{Float64}                   # vector of times
    f::Vector{SVector{4,Float64}}
    A::Vector{SMatrix{4,4,Float64,16}}
    B::Vector{SMatrix{4,1,Float64,4}}
    xm::Vector{SVector{4,Float64}}
    um::Vector{SVector{1,Float64}}
    fm::Vector{SVector{4,Float64}}
    Am::Matrix{SMatrix{4,4,Float64,16}}
    Bm::Matrix{SMatrix{4,1,Float64,4}}
    use_sparse_jacobian::Bool
    function NLP(prob::CartpoleProblem; use_sparse_jacobian::Bool=false)
        n,m,N = prob.n, prob.m, prob.N
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:N]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:N]
        times = collect(range(0, prob.tf, length=N))
        costfun = LQRCost(prob.Q, prob.R, prob.xf)
        costterm = LQRCost(prob.Qf, prob.R, prob.xf)
        f = [@SVector zeros(n) for k = 1:N]
        A = [@SMatrix zeros(n,n) for k = 1:N]
        B = [@SMatrix zeros(n,m) for k = 1:N]
        xm = deepcopy(f)
        um = [@SVector zeros(m) for k = 1:N]
        fm = deepcopy(f)
        Am = [@SMatrix zeros(n,n) for k = 1:N, i = 1:3]
        Bm = [@SMatrix zeros(n,m) for k = 1:N, i = 1:3]
        Np = (n + m) * N
        Nd = n*(N + 1)
        new{n,m,typeof(prob.model)}(
            prob.model, costfun, costterm,
            N, prob.tf, prob.x0, prob.xf, xinds, uinds, times,
            f,A,B,xm,um,fm,Am,Bm,
            use_sparse_jacobian
        )
    end
end
Base.size(nlp::NLP{n,m}) where {n,m} = (n,m,nlp.N)
num_primals(nlp::NLP{n,m}) where {n,m} = (n + m)*nlp.N
num_duals(nlp::NLP) = num_eq(nlp) + num_ineq(nlp)
num_eq(nlp::NLP{n,m}) where {n,m} = n*nlp.N + n
num_ineq(nlp::NLP) = 0

"""
    packZ(nlp, X, U)

Take a vector state vectors `X` and controls `U` and stack them into a single vector Z.
"""
function packZ(nlp, X, U)
    Z = zeros(num_primals(nlp))
    for k = 1:nlp.N
        Z[nlp.xinds[k]] = X[k]
        Z[nlp.uinds[k]] = U[k]
    end
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




