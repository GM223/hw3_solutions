
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
Build a QP from the NLP, optionally using either the Hessian of the cost function 
or the Hessian of the Lagrangian.
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