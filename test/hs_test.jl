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
using BenchmarkTools

include("../src/quadratic_cost.jl")
include("../src/nlp.jl")
include("../src/qp.jl")
include("../src/sqp.jl")
include("../src/cartpole.jl")
include("../src/hermite_simpson.jl")

##
model = RobotZoo.Cartpole()
x1,u1 = rand(model)
x2,u2 = rand(model)
dt = 0.1
λ = @SVector rand(length(x1))

hermite_simpson(model, x1, u1, x2, u2, dt)
hs_jacobian(model, x1, u1, x2, u2, dt)
hs_∇jvp(model, x1, u1, x2, u2, λ, dt)

##
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

X = [x0 + (xf - x0)*t for t in range(0,1, length=T)]
U = [@SVector zeros(m) for k = 1:T];

nlp = NLP(model, obj, tf, T, x0, xf, HermiteSimpson)
Z = packZ(nlp, X, U)
λ = zeros(num_duals(nlp))
N,M = num_primals(nlp), num_duals(nlp)

@test N == T*(n+m)

qp = TOQP(nlp)
@test num_primals(qp) == N

## Test derivatives
λ = rand(M)
c = zero(λ)
eval_c!(nlp, c, Z)
hscon(x) = begin
    c = zeros(eltype(x), M)
    eval_c!(nlp, c, x)
    return c
end

jac = spzeros(M,N)
jac_c!(nlp, jac, Z)
jac0 = ForwardDiff.jacobian(hscon, Z)
@test jac0 ≈ jac

∇jac = spzeros(N,N)
∇jvp!(nlp, ∇jac, Z, λ)

cvp(x) = hscon(x)'λ 
∇jac0 = ForwardDiff.hessian(cvp, Z)
@test ∇jac0 ≈ ∇jac
   
lag(x) = eval_f(nlp, x) - λ'hscon(x)
@test lagrangian(nlp, Z, λ) ≈ eval_f(nlp,Z) - λ'c
@test lag(Z) ≈ eval_f(nlp, Z) - λ'c

grad = zero(Z)
grad_lagrangian!(nlp, grad, Z, λ)
@test ForwardDiff.gradient(lag, Z) ≈ grad

hess = spzeros(N,N)
hess_f!(nlp, hess, Z)
hess_L = zero(hess)
hess_lagrangian!(nlp, hess_L, Z, λ)
hess_L0 = ForwardDiff.hessian(lag, Z)
@test hess_L ≈ hess_L0

## Try SQP
Z = packZ(nlp, X, U)
λ = zeros(num_duals(nlp))
dZ1, dZ2, stats_gn = solve_sqp!(nlp, Z, λ, verbose=1, iters=80, gn=true, eps_dual=1e-2)
Z = packZ(nlp, X, U)
λ = zeros(num_duals(nlp))
dZ1, dZ2, stats_fn = solve_sqp!(nlp, Z, λ, verbose=1, iters=80, gn=false, eps_dual=1e-2)

using Plots
plot(stats_gn[:viol_primal], label="Gauss Newton",
    yscale=:log10, xlabel="iterations", ylabel="Constraint violation", legend=:topright
)
plot!(stats_fn[:viol_primal], label="Full Newton")

plot(stats_gn[:time], stats_gn[:viol_primal], label="Gauss Newton",
    yscale=:log10, xlabel="time (ms)", ylabel="Constraint violation", legend=:bottomleft
)
plot!(stats_fn[:time], stats_fn[:viol_primal], label="Full Newton")