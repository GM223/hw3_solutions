import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); 
using ForwardDiff
using Test
using RobotZoo: PlanarQuadrotor
using RobotDynamics
using LinearAlgebra
using StaticArrays
using SparseArrays
using BlockArrays
using Plots
using MatrixCalculus

include("quadrotor.jl")
include("quadratic_cost.jl")
include("ddp.jl")

## Build Problem
model = PlanarQuadrotor()
n,m = size(model)
dt = 0.025
tf = 1.5 
T = Int(tf/dt) + 1

# Initial & final condition
x0 = SA_F64[-2, 1, 0, 0, 0, 0]
xgoal = SA_F64[+2, 1, 0, 0, 0, 0]
uhover = @SVector fill(0.5*model.mass * model.g, m)

xtraj = kron(ones(1,T), x0)
utraj = kron(ones(1,T-1), uhover)

# Cost function
Q = Diagonal(SVector{6}([ones(3) ; fill(0.1, 3)]))
R = Diagonal(@SVector fill(1e-2, m))
Qn = Diagonal(@SVector fill(1e2, n))
Qf = Qn

# Reference Trajectory
x1ref = [LinRange(-3,0,20); zeros(20); LinRange(0,3,21)]
x2ref = [ones(20); LinRange(1,3,10); LinRange(3,1,10); ones(21)]
θref = [zeros(20); LinRange(0,-2*pi,20); -2*pi*ones(21)]
v1ref = [6.0*ones(20); zeros(20); 6.0*ones(21)]
v2ref = [zeros(20); 8.0*ones(10); -8.0*ones(10); zeros(21)]
ωref = [zeros(20); -4*pi*ones(20); zeros(21)]
xref = [x1ref'; x2ref'; θref'; v1ref'; v2ref'; ωref']
thist = Array(range(0,dt*(T-1), step=dt));

## Build Problem
Xref = [SVector{n}(x) for x in eachcol(xref)]
Uref = [SVector{m}(uhover) for k = 1:T-1]
obj = map(1:T-1) do k
    LQRCost(Q, R, Xref[k], Uref[k])
end
push!(obj, LQRCost(Qf, R, Xref[end]))
prob = Problem(model, obj, T, tf, MVector{n}(x0), MVector{n}(xgoal), thist)

##
xtraj = kron(ones(1,T), x0)
utraj = kron(ones(1,T-1), uhover)
X0 = [SVector{n}(x) for x in eachcol(xtraj)]
U0 = [SVector{m}(u) for u in eachcol(utraj)]
solve_ddp(prob, X0, U0, verbose=1, eps_ddp=1e-2)


## Problem

# xtraj = kron(ones(1,T), x0)
# utraj = kron(ones(1,T-1), uhover)
# solve_ddp(prob, xtraj, utraj, verbose=1)

# ## Cost Functions
# X = [SVector{n}(x) for x in eachcol(xtraj)]
# U = [SVector{m}(u) for u in eachcol(utraj)]


J = cost(xtraj, utraj)

##
k = 1
Ax = dAdx(xtraj[:,k],utraj[:,k])
Bx = dBdx(xtraj[:,k],utraj[:,k])
Au = dAdu(xtraj[:,k],utraj[:,k])
Bu = dBdu(xtraj[:,k],utraj[:,k])

Gxx = kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Ax
Guu = kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bu
Gxu = kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Au
Gux = kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bx

∇jac = zeros(n+m,n+m)
z = KnotPoint(xtraj[:,1], utraj[:,1], dt)
RobotDynamics.∇discrete_jacobian!(RK4, ∇jac, model, z, p[:,2])
∇jac[1:n,1:n] ≈ Gxx
∇jac[n+1:end, n+1:end] ≈ Guu
∇jac[1:n,n+1:end] ≈ Gxu
∇jac[n+1:end,1:n] ≈ Gux

## Run DDP
Nx,Nu,Nt = n,m,T
x0 = [-2.0; 1.0; 0; 0; 0; 0]
xgoal = [2.0; 1.0; 0; 0; 0; 0]
xtraj = kron(ones(1,Nt), x0)
utraj = kron(ones(1,Nt-1), uhover)
J = cost(xtraj,utraj)

xtraj = kron(ones(1,T), x0)
utraj = kron(ones(1,T-1), uhover)
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

iter = 0
while maximum(abs.(j[:])) > 1e-3
    iter += 1
    
    p = zeros(Nx,Nt)
    P = zeros(Nx,Nx,Nt)
    j = zeros(Nu,Nt-1)
    K = zeros(Nu,Nx,Nt-1)
    ΔJ = 0.0

    p[:,Nt] = ForwardDiff.gradient(terminal_cost,xtraj[:,Nt])
    P[:,:,Nt] = ForwardDiff.hessian(terminal_cost,xtraj[:,Nt])
    
    #Backward Pass
    for k = (Nt-1):-1:1
        #Calculate derivatives
        q = ForwardDiff.gradient(dx->stage_cost(dx,utraj[:,k],k),xtraj[:,k])
        Q = ForwardDiff.hessian(dx->stage_cost(dx,utraj[:,k],k),xtraj[:,k])
    
        r = ForwardDiff.gradient(du->stage_cost(xtraj[:,k],du,k),utraj[:,k])
        R = ForwardDiff.hessian(du->stage_cost(xtraj[:,k],du,k),utraj[:,k])
        
        A = dfdx(xtraj[:,k],utraj[:,k])
        B = dfdu(xtraj[:,k],utraj[:,k])
    
        Ax = dAdx(xtraj[:,k],utraj[:,k])
        Bx = dBdx(xtraj[:,k],utraj[:,k])
        Au = dAdu(xtraj[:,k],utraj[:,k])
        Bu = dBdu(xtraj[:,k],utraj[:,k])
    
        gx = q + A'*p[:,k+1]
        gu = r + B'*p[:,k+1]
    
        Gxx = Q + A'*P[:,:,k+1]*A + kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Ax
        Guu = R + B'*P[:,:,k+1]*B + kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bu
        Gxu = A'*P[:,:,k+1]*B + kron(p[:,k+1]',I(Nx))*comm(Nx,Nx)*Au
        Gux = B'*P[:,:,k+1]*A + kron(p[:,k+1]',I(Nu))*comm(Nx,Nu)*Bx
    
        α = 1e-5
        Guu += α*I
        Gxx += α*I
        C = cholesky(Symmetric([Gxx Gxu; Gux Guu]), check=false)
        while !issuccess(C)
            α = 10.0*α
            Guu += α*I
            Gxx += α*I
            C = cholesky(Symmetric([Gxx Gxu; Gux Guu]), check=false)
        end
        # display(α)
    
        j[:,k] .= Guu\gu
        K[:,:,k] .= Guu\Gux
    
        p[:,k] .= gx - K[:,:,k]'*gu + K[:,:,k]'*Guu*j[:,k] - Gxu*j[:,k]
        P[:,:,k] .= Gxx + K[:,:,k]'*Guu*K[:,:,k] - Gxu*K[:,:,k] - K[:,:,k]'*Gux
    
        ΔJ += gu'*j[:,k]
    end
    # display(j)

    #Forward rollout with line search
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
end
iter

##
cost(xtraj, utraj)
