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

function solve_ddp2(prob::Problem, xtraj, utraj;
        iters=100,
        reg_min=1e-6,
        verbose=0,
        eps=1e-5,
        eps_ddp=eps
    )
    t_start = time_ns()
    Nx,Nu,Nt = size(prob)

    p = zeros(Nx,Nt)
    P = zeros(Nx,Nx,Nt)
    j = ones(Nu,Nt-1)
    K = zeros(Nu,Nx,Nt-1)
    ΔJ = 0.0

    Gxx = zeros(Nx,Nx)
    Guu = zeros(Nu,Nu)
    Gxu = zeros(Nx,Nu)
    Gux = zeros(Nu,Nx)

    ∇f = RobotDynamics.DynamicsJacobian(prob.model) 
    ∇jac = zeros(n+m,n+m) 

    J = cost(xtraj, utraj)
    iter = 0
    tol = 1.0
    β = 1e-6
    while tol > eps 
        iter += 1
        
        p = zeros(Nx,Nt)
        P = zeros(Nx,Nx,Nt)
        j = zeros(Nu,Nt-1)
        K = zeros(Nu,Nx,Nt-1)
        ΔJ = 0.0

        p[:,T] = obj[end].Q*xtraj[:,T] + obj[end].q
        P[:,:,T] = obj[end].Q
        
        #Backward Pass
        failed = false
        for k = (Nt-1):-1:1
            #Calculate derivatives
            q = obj[k].Q*xtraj[:,k] + obj[k].q
            Q = obj[k].Q
        
            r = obj[k].R*utraj[:,k] + obj[k].r
            R = obj[k].R

            dt = prob.times[k+1] - prob.times[k]
            z = KnotPoint(SVector{n}(xtraj[:,k]), SVector{m}(utraj[:,k]), dt, prob.times[k])
            discrete_jacobian!(RK4, ∇f, model, z)

            A = ∇f.A
            B = ∇f.B
        
            gx = q + A'*p[:,k+1]
            gu = r + B'*p[:,k+1]
        
            Gxx = Q + A'*P[:,:,k+1]*A
            Guu = R + B'*P[:,:,k+1]*B
            Gux = B'*P[:,:,k+1]*A
            
            if tol < eps_ddp 
                # #Add full Newton terms
                RobotDynamics.∇discrete_jacobian!(RK4, ∇jac, model, z, p[:,k+1])
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
                display(β)
                break
            end
        
            j[:,k] .= Guu_reg\gu
            K[:,:,k] .= Guu_reg\Gux_reg
        
            p[:,k] .= gx - K[:,:,k]'*gu + K[:,:,k]'*Guu*j[:,k] - Gux'*j[:,k]
            P[:,:,k] .= Gxx + K[:,:,k]'*Guu*K[:,:,k] - Gux'*K[:,:,k] - K[:,:,k]'*Gux
        
            ΔJ += gu'*j[:,k]
        end
        #display(j)

        #Forward rollout with line search
        if failed
            continue
        end

        xn = zeros(Nx,Nt)
        un = zeros(Nu,Nt-1)
        xn[:,1] = xtraj[:,1]
        α = 1.0

        for k = 1:(Nt-1)
            un[:,k] .= utraj[:,k] - α*j[:,k] - K[:,:,k]*(xn[:,k]-xtraj[:,k])
            xn[:,k+1] .= quad_dynamics_rk4(xn[:,k],un[:,k])
        end
        Jn = cost(xn,un)
        
        while Jn > (J - 1e-2*α*ΔJ)
            α = 0.5*α
            for k = 1:(Nt-1)
                un[:,k] .= utraj[:,k] - α*j[:,k] - K[:,:,k]*(xn[:,k]-xtraj[:,k])
                xn[:,k+1] .= quad_dynamics_rk4(xn[:,k],un[:,k])
            end
            Jn = cost(xn,un)
        end
        #display(α)
        
        J = Jn
        xtraj .= xn
        utraj .= un
        
        tol = maximum(abs.(j[:]))
        β = max(0.9*β, 1e-6)

        if verbose > 0
            @printf("Iter: %3d, Cost: % 6.2f → % 6.2f (% 7.2e), res: % .2e, β= %.2e\n",
                iter, J, Jn, J-Jn, norm(j, Inf), β
            )
        end

    end
    println("Total Time: ", (time_ns() - t_start)*1e-6, " ms")
    return xtraj, utraj
end

function solve_ddp(prob::Problem{n,m}, xtraj, utraj;
        iters=100,
        ddp=true,
        reg_min=1e-6,
        verbose=0,
    ) where {n,m}

    t_start = time_ns()

    Nx,Nu,Nt = size(prob) 
    obj = prob.obj
    T = prob.T
    # x0 = [-2.0; 1.0; 0; 0; 0; 0]
    # xgoal = [2.0; 1.0; 0; 0; 0; 0]

    # xtraj = kron(ones(1,Nt), x0)
    # utraj = kron(ones(1,Nt-1), uhover)
    J = cost(xtraj,utraj)

    verbose = 1
    p = zeros(Nx,Nt)
    P = zeros(Nx,Nx,Nt)
    j = ones(Nu,Nt-1)
    K = zeros(Nu,Nx,Nt-1)
    ΔJ = 0.0

    Gxx = zeros(Nx,Nx)
    Guu = zeros(Nu,Nu)
    Gxu = zeros(Nx,Nu)
    Gux = zeros(Nu,Nx)
    
    ∇f = RobotDynamics.DynamicsJacobian(prob.model) 
    ∇jac = zeros(n+m,n+m) 

    β = 1e-6
    iter = 0
    tol = Inf 
    ρ = reg_min 
    while tol > 1e-8 
        iter += 1
        
        p = zeros(Nx,Nt)
        P = zeros(Nx,Nx,Nt)
        j = zeros(Nu,Nt-1)
        K = zeros(Nu,Nx,Nt-1)
        ΔJ = 0.0

        # p[:,Nt] = ForwardDiff.gradient(terminal_cost,xtraj[:,Nt])
        # P[:,:,Nt] = ForwardDiff.hessian(terminal_cost,xtraj[:,Nt])
        p[:,T] = obj[end].Q*xtraj[:,T] + obj[end].q
        P[:,:,T] = obj[end].Q
        
        #Backward Pass
        failed = false
        for k = (Nt-1):-1:1
            #Calculate derivatives
            # q = ForwardDiff.gradient(dx->stage_cost(dx,utraj[:,k],k),xtraj[:,k])
            # Q = ForwardDiff.hessian(dx->stage_cost(dx,utraj[:,k],k),xtraj[:,k])
            q = obj[k].Q*xtraj[:,k] + obj[k].q
            Q = obj[k].Q
        
            # r = ForwardDiff.gradient(du->stage_cost(xtraj[:,k],du,k),utraj[:,k])
            # R = ForwardDiff.hessian(du->stage_cost(xtraj[:,k],du,k),utraj[:,k])
            r = obj[k].R*utraj[:,k] + obj[k].r
            R = obj[k].R

            dt = prob.times[k+1] - prob.times[k]
            z = KnotPoint(SVector{n}(xtraj[:,k]), SVector{m}(utraj[:,k]), dt, prob.times[k])
            discrete_jacobian!(RK4, ∇f, model, z)

            # A = dfdx(xtraj[:,k],utraj[:,k])
            # B = dfdu(xtraj[:,k],utraj[:,k])
            A = ∇f.A
            B = ∇f.B
        
            gx = q + A'*p[:,k+1]
            gu = r + B'*p[:,k+1]
        
            Gxx = Q + A'*P[:,:,k+1]*A
            Guu = R + B'*P[:,:,k+1]*B
            Gxu = A'*P[:,:,k+1]*B
            Gux = B'*P[:,:,k+1]*A

            if ddp

                # Ax = dAdx(xtraj[:,k],utraj[:,k])
                # Bx = dBdx(xtraj[:,k],utraj[:,k])
                # Au = dAdu(xtraj[:,k],utraj[:,k])
                # Bu = dBdu(xtraj[:,k],utraj[:,k])

                RobotDynamics.∇discrete_jacobian!(RK4, ∇jac, model, z, p[:,k+1])
                Gxx .+= ∇jac[1:n, 1:n]
                Guu .+= ∇jac[n+1:end, n+1:end]
                Gux .+= ∇jac[n+1:end, 1:n]
                Gxu .+= ∇jac[1:n, n+1:end]
        
                # Gxx .+= kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Ax
                # Guu .+= kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bu
                # Gxu .+= kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Au
                # Gux .+= kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bx
            end
        
            #Regularization
            Gxx_reg = Gxx + A'*β*I*A
            Guu_reg = Guu + B'*β*I*B
            Gux_reg = Gux + B'*β*I*A
            C = cholesky(Symmetric([Gxx_reg Gux_reg'; Gux_reg Guu_reg]), check=false)
            if !issuccess(C)
                β = 2*β
                failed = true
                display(β)
                break
            end

            j[:,k] .= Guu_reg\gu
            K[:,:,k] .= Guu_reg\Gux_reg
        
            p[:,k] .= gx - K[:,:,k]'*gu + K[:,:,k]'*Guu*j[:,k] - Gxu*j[:,k]
            P[:,:,k] .= Gxx + K[:,:,k]'*Guu*K[:,:,k] - Gxu*K[:,:,k] - K[:,:,k]'*Gux
        
            ΔJ += gu'*j[:,k]
        end
        # display(j)

        #Forward rollout with line search
        # if failed
        #     continue
        # end

        xn = zeros(Nx,Nt)
        un = zeros(Nu,Nt-1)
        xn[:,1] = xtraj[:,1]
        α = 1.0

        for k = 1:(Nt-1)
            un[:,k] .= utraj[:,k] - α*j[:,k] - K[:,:,k]*(xn[:,k]-xtraj[:,k])
            xn[:,k+1] .= quad_dynamics_rk4(xn[:,k],un[:,k])
        end
        Jn = cost(xn,un)
        
        while Jn > (J - 1e-2*α*ΔJ)
            α = 0.5*α
            for k = 1:(Nt-1)
                un[:,k] .= utraj[:,k] - α*j[:,k] - K[:,:,k]*(xn[:,k]-xtraj[:,k])
                xn[:,k+1] .= quad_dynamics_rk4(xn[:,k],un[:,k])
            end
            Jn = cost(xn,un)
        end
        # display(α)

        if verbose > 0
            @printf("Iter: %3d, Cost: % 6.2f → % 6.2f (% 7.2e), res: % .2e\n",
                iter, J, Jn, J-Jn, norm(j, Inf)
            )
        end
        
        J = Jn
        xtraj .= xn
        utraj .= un

        tol = norm(j, Inf)
        β = max(0.9*β, 1e-6)

    end
    println("Total Time: ", (time_ns() - t_start)*1e-6, " ms")
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