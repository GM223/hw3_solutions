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

include("../src/nlp.jl")
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

X = [x0 + (xf - x0)*t for t in range(0,1, length=T)]
U = [@SVector zeros(m) for k = 1:T-1]

@test stagecost(costfun, X[1], U[1]) ≈ 0.5*(X[1]-xf)'Q*(X[1]-xf) + 0.5*U[1]'R*U[1]
@test termcost(costterm, X[T]) ≈ 0.0
@test termcost(costterm, X[1]) ≈ 0.5*(X[1]-xf)'Qf*(X[1]-xf)


## NLP
nlp = NLP(model, obj, tf, T, x0, xf)
Z = packZ(nlp, X, U)

# test cost
J0 = mapreduce(+,1:T) do k
    k < T ? stagecost(costfun, X[k], U[k]) : termcost(costterm, X[k])
end
@test eval_f(nlp, Z) ≈ J0

# test cost gradient
grad = zero(Z)
grad_f!(nlp, grad, Z)
@test ForwardDiff.gradient(x->eval_f(nlp, x), Z) ≈ grad

# test cost Hessian
hess = spzeros(num_primals(nlp), num_primals(nlp))
hess_f!(nlp, hess, Z)
@test ForwardDiff.hessian(x->eval_f(nlp, x), Z) ≈ hess 

# test constraint
c = zeros(num_duals(nlp))
eval_c!(nlp, c, Z)
@test c[1:n] ≈ zeros(n)
for k = 1:T-1
    @test c[k*n .+ (1:n)] ≈ discrete_dynamics(RK4, model, X[k], U[k], 0.0, dt) - X[k+1]
end
@test c[end-n+1:end] ≈ zeros(n)

# test constraint Jacobian
jac = spzeros(num_duals(nlp), num_primals(nlp))
jac_c!(nlp, jac, Z)
parts_primal = push!(repeat([n,m],T-1),n)
jac2 = PseudoBlockArray(jac, fill(n,T+1), parts_primal) 
∇f = map(1:T-1) do k
    F = RobotDynamics.DynamicsJacobian(model)
    discrete_jacobian!(RK4, F, model, KnotPoint(X[k], U[k], dt))
    F
end
@test jac2[Block(1,1)] ≈ I(n)
@test jac2[Block(2,1)] ≈ ∇f[1].A
@test jac2[Block(2,2)] ≈ ∇f[1].B
@test jac2[Block(2,3)] ≈ -I(n) 
@test jac2[Block(3,3)] ≈ ∇f[2].A
@test jac2[Block(T,2T-1)] ≈ -I(n)
@test jac2[Block(T+1,2T-1)] ≈ I(n)

jac0 = zero(jac)
@test ForwardDiff.jacobian!(jac0, (c,x)->eval_c!(nlp, c, x), c, Z) ≈ jac

# jvp
λ = rand(length(c))
jacvec = zero(Z)
jvp!(nlp, jacvec, Z, λ)
@test jacvec ≈ jac'λ

# ∇jvp
cvp(x) = begin
    c = zeros(eltype(x), length(λ))
    eval_c!(nlp, c, x)
    return λ'c
end
cvp(Z) ≈ c'λ
ForwardDiff.gradient(cvp, Z) ≈ jac'λ
∇jac0 = ForwardDiff.hessian(cvp, Z)

∇jac = zero(hess) 
∇jvp!(nlp, ∇jac, Z, λ)
@test ∇jac ≈ ∇jac0


# Lagrangian
@test lagrangian(nlp, Z, λ) ≈ J0 - λ'c
