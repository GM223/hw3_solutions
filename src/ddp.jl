using Printf

"""
    Problem{n,m,L}

Describes a trajectory optimization problem with `n` states, `m` controls, and 
a model of type `L`. 

# Constructor
    Problem(model::L, obj::Vector{<:QuadraticCost{n,m}}, tf, x0, xf) where {n,m,L}

where `tf` is the final time, and `x0` is the initial state. 
"""
struct Problem{n,m,L}
    model::L
    obj::Vector{QuadraticCost{n,m,Float64}}
    T::Int
    tf::Float64
    x0::MVector{n,Float64}
    times::Vector{Float64}
    function Problem(model::L, obj::Vector{<:QuadraticCost{n,m}}, tf, x0) where {n,m,L}
        @assert length(x0) == n == length(obj[1].q) == state_dim(model)
        @assert length(obj[1].r) == m == control_dim(model)
        T = length(obj)
        times = range(0, tf, length=T)
        new{n,m,L}(model, obj, T, tf, x0, times)
    end
end
Base.size(prob::Problem{n,m}) where {n,m} = (n,m,prob.T)

"""
    solve_ilqr(prob, X, U; kwargs...)

Solve the trajectory optimization problem specified by `prob` using iterative LQR.
Returns the optimized state and control trajectories, as well as the local control gains,
`K` and `d`.
"""
function solve_ilqr(prob::Problem{n,m}, X0, U0; 
        iters=100,     # max iterations
        ls_iters=10,   # max line search iterations
        reg_min=1e-6,  # minimum regularizatio for the backwardpass
        verbose=0,     # print verbosity
        eps=1e-5,      # termination tolerance
        eps_ddp=eps    # tolerance to switch to ddp
    ) where {n,m}
    t_start = time_ns()
    Nx,Nu,Nt = size(prob)

    T = prob.T
    p = [zeros(n) for k = 1:T]      # ctg gradient
    P = [zeros(n,n) for k = 1:T]    # ctg hessian
    d = [zeros(m) for k = 1:T-1]    # feedforward gains
    K = [zeros(m,n) for k = 1:T-1]  # feedback gains
    ΔJ = 0.0

    X = deepcopy(X0)
    U = deepcopy(U0)

    Xbar = [@SVector zeros(n) for k = 1:T]
    Ubar = [@SVector zeros(m) for k = 1:T-1]


    J = cost(prob.obj, X, U)
    Jn = Inf
    iter = 0
    tol = 1.0
    β = 1e-6
    while tol > eps 
        iter += 1
        
        ddp = tol < eps_ddp
        ΔJ, failed = backwardpass!(prob, P, p, K, d, X, U, ddp=ddp, β=β)

        if failed
            β *= 2
            continue
        end

        Jn, α = forwardpass!(prob, X, U, Xbar, Ubar, K, d, ΔJ, J, max_iters=ls_iters)
        
        tol = maximum(norm.(d, Inf))
        β = max(0.9*β, 1e-6)

        if verbose > 0
            @printf("Iter: %3d, Cost: % 6.2f → % 6.2f (% 7.2e), res: % .2e, β= %.2e, α = %.3f\n",
                iter, J, Jn, J-Jn, tol, β, α
            )
        end
        J = Jn

        if iter >= iters
            @warn "Reached max iterations"
            break
        end

    end
    println("Total Time: ", (time_ns() - t_start)*1e-6, " ms")
    return X,U,K,d
end

"""
    backwardpass!(prob, P, p, K, d, X, U)

Evaluate the iLQR backward pass at state and control trajectories `X` and `U`, 
storing the cost-to-go expansion in `P` and `p` and the gains in `K` and `d`.
"""
function backwardpass!(prob::Problem{n,m}, P, p, K, d, X, U; 
        β=1e-6, ddp::Bool=false
    ) where {n,m}
    T = prob.T
    obj = prob.obj
    ΔJ = 0.0

    ∇f = RobotDynamics.DynamicsJacobian(prob.model) 
    ∇jac = zeros(n+m,n+m) 

    p[T] = obj[end].Q*X[T] + obj[end].q
    P[T] = obj[end].Q
    
    #Backward Pass
    failed = false
    for k = (T-1):-1:1
        # Cost Expansion
        q = obj[k].Q*X[k] + obj[k].q
        Q = obj[k].Q
        r = obj[k].R*U[k] + obj[k].r
        R = obj[k].R

        # Dynamics derivatives
        dt = prob.times[k+1] - prob.times[k]
        z = KnotPoint(SVector{n}(X[k]), SVector{m}(U[k]), dt, prob.times[k])
        discrete_jacobian!(RK4, ∇f, model, z)
        A = RobotDynamics.get_static_A(∇f)
        B = RobotDynamics.get_static_B(∇f)
    
        gx = q + A'*p[k+1]
        gu = r + B'*p[k+1]
    
        Gxx = Q + A'*P[k+1]*A
        Guu = R + B'*P[k+1]*B
        Gux = B'*P[k+1]*A
        
        if ddp 
            # #Add full Newton terms
            RobotDynamics.∇discrete_jacobian!(RK4, ∇jac, model, z, p[k+1])
            Gxx .+= ∇jac[1:n, 1:n]
            Guu .+= ∇jac[n+1:end, n+1:end]
            Gux .+= ∇jac[n+1:end, 1:n]
        end
    
        # Regularization
        Gxx_reg = Gxx + A'*β*I*A
        Guu_reg = Guu + B'*β*I*B
        Gux_reg = Gux + B'*β*I*A
        C = cholesky(Symmetric([Gxx_reg Gux_reg'; Gux_reg Guu_reg]), check=false)
        if !issuccess(C)
            β = 2*β
            failed = true
            break
        end
    
        d[k] .= Guu_reg\gu
        K[k] .= Guu_reg\Gux_reg
    
        p[k] .= gx - K[k]'*gu + K[k]'*Guu*d[k] - Gux'*d[k]
        P[k] .= Gxx + K[k]'*Guu*K[k] - Gux'*K[k] - K[k]'*Gux
    
        ΔJ += gu'*d[k]
    end
    return ΔJ, failed
end

"""
    forwardpass!(prob, X, U, K, d, ΔJ, J)

Evaluate the iLQR forward pass at state and control trajectories `X` and `U`, using
the gains `K` and `d` to simulate the system forward. The new cost should be less than 
the current cost `J` together with the expected cost decrease `ΔJ`.
"""
function forwardpass!(prob::Problem{n,m}, X, U, K, d, ΔJ, J,
        Xbar = deepcopy(X), Ubar = deepcopy(U);
        max_iters=10,
    ) where {n,m}
    T = prob.T

    # Line Search
    Xbar[1] = X[1]
    α = 1.0
    Jn = Inf
    for i = 1:max_iters
        for k = 1:(T-1)
            t = prob.times[k]
            dt = prob.times[k+1] - prob.times[k]
            Ubar[k] = U[k] - α*d[k] - K[k]*(Xbar[k]-X[k])
            Xbar[k+1] = discrete_dynamics(RK4, model, Xbar[k], Ubar[k], t, dt) 
        end
        Jn = cost(prob.obj, Xbar, Ubar)

        if Jn <= J - 1e-2*α*ΔJ
            break
        else
            α *= 0.5
        end
        if i == max_iters 
            @warn "Line Search failed"
            α = 0
        end
    end
    
    # Accept direction
    for k = 1:T-1
        X[k] = Xbar[k]
        U[k] = Ubar[k]
    end
    X[T] = Xbar[T]
    
    return Jn, α
end
