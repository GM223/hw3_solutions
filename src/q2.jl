import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
using Random
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
using SparseArrays

include("quadratic_cost.jl")
include("q2_model.jl")
include("q2_tests.jl")
include("sparseblocks.jl")

## Part a: Cost Functions (15 pts)
include("q2_prob.jl")
include("q2_nlp.jl")
include("q2_cost_methods.jl")  # TODO: complete methods here
test_costs()

## Part b: Constraints (20 pts)
include("q2_constraints.jl")   # TODO: complete methods here
test_constraints()

## Part c: Solve (5 pts)
include("q2_moi.jl")
let
    prob = CartpoleProblem()
    X,U = get_initial_trajectory(prob)
    global nlp = NLP(prob)
    Z0 = packZ(nlp, X, U)
    global Zsol, solver
    Zsol,solver = solve(Z0, nlp)
end
test_solution(nlp, Zsol, solver)

# Visualization
let prob = CartpoleProblem() 
    model = prob.model
    global vis = Visualizer()
    set_mesh!(vis, model)
    render(vis)
end
let Z = Zsol 
    X, = unpackZ(nlp, Z)
    visualize!(vis, nlp.model, nlp.tf, X)
end

## Part (d): Track the solution (10 pts)
include("q2_controller.jl")
let Zref = copy(Zsol)
    ctrl = gen_controller(nlp, Zref)
    model2 = RobotZoo.Cartpole(1.1, 0.2, 0.5, 9.81)
    Xsim, Usim, tsim = simulate(model2, nlp.x0, ctrl, tf=5nlp.tf, dt=0.005)
    visualize!(vis, model2, tsim[end], Xsim)
end
test_tracking()

## Part (e): EXTRA CREDIT Leveraging sparsity
test_extracredit()
typeof(getrc(nlp.blocks))