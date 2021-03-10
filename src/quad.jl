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

