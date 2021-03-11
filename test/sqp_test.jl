import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); 
# using hw3
using ForwardDiff
using Test
using RobotZoo
using RobotDynamics
using LinearAlgebra
using StaticArrays
using SparseArrays
using BlockArrays

include("../src/quadratic_cost.jl")
include("../src/nlp.jl")
include("../src/qp.jl")
include("../src/sqp.jl")
# using hw3: LQRCost, NLP

## Set up problem
model = RobotZoo.Cartpole()
n,m = size(model)
Q = Diagonal(fill(1e-3,n))
R = Diagonal(fill(1e-1,m))
Qf = Diagonal(fill(1e3,n))

x0 = @SVector zeros(n)
xf = SA[0,pi,0,0]
T = 101
tf = 2.0
dt = tf / (T-1)

costfun = LQRCost(Q,R,xf)
costterm = LQRCost(Qf,R,xf)
obj = push!(fill(costfun,T-1), costterm)

## 
nlp = NLP(model, obj, tf, T, x0, xf)
Z = packZ(nlp, X, U)
λ = zeros(num_duals(nlp))

dZ1, dZ2 = solve_sqp!(nlp, Z, λ, verbose=1, iters=200, gn=true, eps_primal=1e-4)
norm(dZ1,Inf)
norm(dZ2,Inf)
dZ1
dZ2


## Visualizer
using TrajOptPlots
using MeshCat
vis = Visualizer()
open(vis)
TrajOptPlots.set_mesh!(vis, model)
Xsol = [Z[xi] for xi in nlp.xinds]
visualize!(vis, model, tf, Xsol)