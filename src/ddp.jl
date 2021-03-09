using Printf

struct Problem{n,m,L}
    model::L
    obj::Vector{QuadraticCost{n,m,Float64}}
    T::Int
    tf::Float64
    x0::MVector{n,Float64}
    xf::MVector{n,Float64}
    times::Vector{Float64}
end
Base.size(prob::Problem{n,m}) where {n,m} = (n,m,prob.T)

function cost(obj::Vector{<:QuadraticCost{n,m,T}}, X, U) where {n,m,T}
    J = zero(T)
    for k = 1:length(U)
        J += stagecost(obj[k], X[k], U[k])
    end
    J += termcost(obj[end], X[end])
    return J
end

function solve_ddp(prob::Problem{n,m}, X, U; 
        iters=100,
        ls_iters=10,
        reg_min=1e-6,
        verbose=0,
        eps=1e-5,
        eps_ddp=eps
    ) where {n,m}
    t_start = time_ns()
    Nx,Nu,Nt = size(prob)

    T = prob.T
    p = [zeros(n) for k = 1:T]      # ctg gradient
    P = [zeros(n,n) for k = 1:T]    # ctg hessian
    j = [zeros(m) for k = 1:T-1]    # feedforward gains
    K = [zeros(m,n) for k = 1:T-1]  # feedback gains
    ΔJ = 0.0

    Xbar = [@SVector zeros(n) for k = 1:T]
    Ubar = [@SVector zeros(m) for k = 1:T-1]

    ∇f = RobotDynamics.DynamicsJacobian(prob.model) 
    ∇jac = zeros(n+m,n+m) 

    J = cost(xtraj, utraj)
    Jn = Inf
    iter = 0
    tol = 1.0
    β = 1e-6
    while tol > eps 
        iter += 1
        
        ΔJ = 0.0

        p[T] = obj[end].Q*X[T] + obj[end].q
        P[T] = obj[end].Q
        
        #Backward Pass
        failed = false
        for k = (Nt-1):-1:1
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
            
            if tol < eps_ddp 
                # #Add full Newton terms
                RobotDynamics.∇discrete_jacobian!(RK4, ∇jac, model, z, p[k+1])
                Gxx .+= ∇jac[1:n, 1:n]
                Guu .+= ∇jac[n+1:end, n+1:end]
                Gux .+= ∇jac[n+1:end, 1:n]
                Gxu .+= ∇jac[1:n, n+1:end]
            end
        
            #Regularization
            Gxx_reg = Gxx + A'*β*I*A
            Guu_reg = Guu + B'*β*I*B
            Gux_reg = Gux + B'*β*I*A
            C = cholesky(Symmetric([Gxx_reg Gux_reg'; Gux_reg Guu_reg]), check=false)
            if !issuccess(C)
                β = 2*β
                failed = true
                break
            end
        
            j[k] .= Guu_reg\gu
            K[k] .= Guu_reg\Gux_reg
        
            p[k] .= gx - K[k]'*gu + K[k]'*Guu*j[k] - Gux'*j[k]
            P[k] .= Gxx + K[k]'*Guu*K[k] - Gux'*K[k] - K[k]'*Gux
        
            ΔJ += gu'*j[k]
        end
        #display(j)

        if failed
            continue
        end

        # Forward rollout with line search
        Xbar[1] = X[1]
        α = 1.0

        for i = 1:ls_iters
            for k = 1:(Nt-1)
                t = prob.times[k]
                dt = prob.times[k+1] - prob.times[k]
                Ubar[k] = U[k] - α*j[k] - K[k]*(Xbar[k]-X[k])
                Xbar[k+1] = discrete_dynamics(RK4, model, Xbar[k], Ubar[k], t, dt) 
            end
            Jn = cost(obj, Xbar, Ubar)

            if Jn <= J - 1e-2*α*ΔJ
                break
            else
                α *= 0.5
            end
            if i == ls_iters
                @warn "Line Search failed"
                α = 0
            end
        end
        
        for k = 1:T-1
            X[k] = Xbar[k]
            U[k] = Ubar[k]
        end
        X[T] = Xbar[T]
        
        tol = maximum(norm.(j, Inf))
        β = max(0.9*β, 1e-6)

        if verbose > 0
            @printf("Iter: %3d, Cost: % 6.2f → % 6.2f (% 7.2e), res: % .2e, β= %.2e\n",
                iter, J, Jn, J-Jn, norm(j, Inf), β
            )
        end
        J = Jn

        if iter >= iters
            @warn "Reached max iterations"
            break
        end

    end
    println("Total Time: ", (time_ns() - t_start)*1e-6, " ms")
    return X,U 
end


# Cost functions
function stage_cost(x,u,k)
    return 0.5*(x-xref[:,k])'*Q*(x-xref[:,k]) + 0.5*(u-uhover)'*R*(u-uhover)
    #return 0.5*(x-xgoal)'*Q*(x-xgoal) + 0.5*(u-uhover)'*R*(u-uhover)
end

function terminal_cost(x)
    Nt = size(xref,2)
    return 0.5*(x-xref[:,Nt])'*Qn*(x-xref[:,Nt])
    #return 0.5*(x-xgoal)'*Qn*(x-xgoal)
end

function cost(xtraj,utraj)
    J = 0
    Nt = size(xtraj,2)
    for k = 1:(Nt-1)
        J += stage_cost(xtraj[:,k],utraj[:,k],k)
    end
    J += terminal_cost(xtraj[:,Nt])
    return J
end

function V(x,k)
    Δx = x-xtraj[:,k]
    return 0.5*Δx'*P[:,:,k]*Δx + p[:,k]'*Δx
end

function S(x,u,k)
    return stage_cost(x,u,k) + V(quad_dynamics_rk4(x,u),k+1)
end

# Dynamics
function quad_dynamics(x,u)
    g = 9.81 #m/s^2
    m = 1.0 #kg 
    ℓ = 0.3 #meters

    J = 0.2*m*ℓ*ℓ
    
    θ = x[3]
    
    ẍ = (1/m)*(u[1] + u[2])*sin(θ)
    ÿ = (1/m)*(u[1] + u[2])*cos(θ) - g
    θ̈ = (1/J)*(ℓ/2)*(u[2] - u[1])
    
    return [x[4:6]; ẍ; ÿ; θ̈]
end

function quad_dynamics_rk4(x,u)
    h = dt
    #RK4 integration with zero-order hold on u
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5*h*f1, u)
    f3 = quad_dynamics(x + 0.5*h*f2, u)
    f4 = quad_dynamics(x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

function dfdx(x,u)
    return ForwardDiff.jacobian(dx->quad_dynamics_rk4(dx,u),x)
end

function dfdu(x,u)
    return ForwardDiff.jacobian(du->quad_dynamics_rk4(x,du),u)
end

#Second derivatives of dynamics
function dAdx(x,u)
    return ForwardDiff.jacobian(dx->vec(dfdx(dx,u)),x)
end

function dBdx(x,u)
    return ForwardDiff.jacobian(dx->vec(dfdu(dx,u)),x)
end

function dAdu(x,u)
    return ForwardDiff.jacobian(du->vec(dfdx(x,du)),u)
end

function dBdu(x,u)
    return ForwardDiff.jacobian(du->vec(dfdu(x,du)),u)
end