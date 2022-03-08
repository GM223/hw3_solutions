import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); 
using ForwardDiff
using Test
using RobotZoo
import RobotDynamics
using LinearAlgebra
using StaticArrays
using SparseArrays
using Printf
using MeshCat
using Plots
using FiniteDiff
import MathOptInterface as MOI
using Ipopt
using JLD2

include("quadratic_cost.jl")
# include("../test/nlp_test.jl")
include("cartpole.jl")

# include("q2_types.jl")
include("q2_prob.jl")
include("q2_nlp.jl")
include("q2_dynamics.jl")
include("q2_cost_methods.jl")
include("q2_constraints.jl")
include("moi.jl")

prob = CartpoleProblem()
X,U = get_initial_trajectory(prob)
nlp = NLP(prob)
Z0 = packZ(nlp, X, U)
Zsol,solver = solve(Z0, nlp)

function test_costs()
    prob = CartpoleProblem()
    nlp = NLP(prob)
    X,U = get_initial_trajectory(prob) 
    Z = packZ(nlp, X, U)
     
    # Test the cost
    @test eval_f(nlp, Z) ≈ 0.22766546346850902 atol=1e-6

    # Test the cost gradient with FiniteDiff
    grad = zero(Z)
    grad_f!(nlp, grad, Z)
    grad_fd = FiniteDiff.finite_difference_gradient(x->eval_f(nlp, x), Z)
    @test norm(grad - grad_fd) < 1e-8
end
test_costs()

function test_constraints()
    prob = CartpoleProblem()
    nlp = NLP(prob)
    X,U = get_initial_trajectory(prob) 
    Z = packZ(nlp, X, U)

    resfile = joinpath(@__DIR__, "Q2.jld2")

    # Constraint function
    c = zeros(num_duals(nlp))
    devals = @dynamicsevals eval_c!(nlp, c, Z)
    @test devals <= 301
    @test devals <= 201
    @test norm(c - load(resfile, "c0")) < 1e-8

    # Calc constraint Jacobian and check Jacobian evals
    jac = zeros(num_duals(nlp), num_primals(nlp))
    jevals = @jacobianevals jac_c!(nlp, jac, Z)
    devals = @dynamicsevals jac_c!(nlp, jac, Z)
    @test devals <= 301
    @test devals <= 201
    @test devals <= 101
    @test devals == 0 
    @test jevals <= 301
    @test jevals <= 201
    @test jevals > 200  # this checks that they don't use ForwardDiff or FiniteDiff
    
    # Check constraint Jacobian with FiniteDiff
    jac_fd = zero(jac)
    FiniteDiff.finite_difference_jacobian!(jac_fd, (y,x)->eval_c!(nlp, y, x), Z)
    @test norm(jac - jac_fd) < 1e-6
end
c0 = test_constraints()

## Visualization
let model = prob.model
    global vis = Visualizer()
    set_mesh!(vis, model)
    render(vis)
end
let X = [Zsol[xi] for xi in nlp.xinds]
    visualize!(vis, prob.model, prob.tf, X)
end