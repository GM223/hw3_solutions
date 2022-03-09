import Pkg; Pkg.activate(joinpath(@__DIR__,"..")); Pkg.instantiate()
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

include("quadratic_cost.jl")   # defines the quadratic cost function type
include("q2_model.jl")         # sets up the dynamics
const isautograder = @isdefined autograder

#   Q1: Sequential Quadratic Programming (SQP) (50 pts)
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   In this problem you'll solve the canonical cartpole swing-up problem using
#   the classic direct collocation algorithm with Hermite-Simpson integration.
# 
#   Continuous Problem
#   ––––––––––––––––––––
# 
#   We will be solving a trajectory optimization problem of the form: :$
#   \begin{aligned} &\underset{x(t), u(t)}{\text{minimize}} && Jf(x(tf)) +
#   \int{t0}^{tf} J(x(t), u(t)) dt \ &\text{subject to} && \dot{x}(t) = f(x(t),
#   u(t), t) \ &&& x(t0) = x\text{init} \ &&& x(tf) = x_\text{goal}
#   \end{aligned} :$
# 
#   Hermite-Simpson Collocation
#   –––––––––––––––––––––––––––––
# 
#   Recall from lecture that direct collocation "transcribes" the
#   continuous-time optimal control problem into a finite-dimensional nonlinear
#   program (NLP). We will use Hermite-Simpson integration on both our dynamics
#   and our cost function. We will split our cost integral into N-1 segments of
#   length h seconds, and approximate the cost over this interval using a
#   Hermite spline:
# 
# :$
# 
#   \int{tk}^{t{k+1}} J\big(x(t),u(t)\big) dt \approx \frac{h}{6}\bigg(
#   J\big(x(tk), u(tk)\big) + 4 J\big(x(tk + h/2), u(tk + h/2)\big) + J\big(x(tk
#   + h), u(t_k + h)\big) \bigg) :$
# 
#   where we calculate the state at the midpoint with: :$ x(tk + h/2) = xm =
#   \frac{1}{2} \big(x1 + x2 \big) + \frac{h}{8}\big(f(xk, uk, tk) - f(x{k+1},
#   u{k+1}, t{k+1}) \big) :$
# 
#   and we use first-order-hold on the controls: :$ u(tk + h/2) = um =
#   \frac{1}{2} \big( u1 + u2 \big) :$
# 
#   For our dynamics constraint, we use implicit integration with the same
#   Hermite spline: :$ \frac{h}{6} \big(f(xk,uk,tk) + 4f(xm,um,tm) + f(x{k+1},
#   u{k+1}, t{k+1}) \big) + xk - x_{k+1} = 0 :$
# 
#   Discrete Problem
#   ––––––––––––––––––
# 
#   The resulting NLP has the following form: :$ \begin{aligned}
#   &\underset{x{1:N}, u{1:N}}{\text{minimize}} && Jf(xN) + \sum{k=1}^{N-1}
#   \frac{h}{6}(J(xk,uk) + 4J(xm,um) + J(x{k+1}, u{k+1})) \ &\text{subject to}
#   && \frac{h}{6} \big(f(xk,uk,tk) + 4f(xm,um,tm) + f(x{k+1}, u{k+1}, t{k+1})
#   \big) + xk - x{k+1} = 0 \ &&& x1 = x\text{init} \ &&& xN = x\text{goal}
#   \end{aligned} :$
# 
#   Note that the state midpoint is really a function of the states and controls
#   at the surrounding knot points: x_m(x_k, u_k, x_{k+1}, u_{k+1}, t_k, h) and
#   the control at the midpoint is a function of the previous and next and
#   control values: u_m(u_k, u_{k+1}). You will need differentiate through these
#   splines using the chain rule to generate the methods we need to solve our
#   NLP.
# 
#   Solving the Problem
#   –––––––––––––––––––––
# 
#   To make things easier, we'll use Ipopt to solve our NLP, but you'll still
#   need to define the functions we pass to Ipopt. Ipopt expects a problem of
#   the following form:
# 
#   
# \begin{aligned} 
# &\underset{x}{\text{minimize}} && f(x) \\
# &\text{subject to} && l \leq c(x) \leq u\\
# \end{aligned} 
# $$
# 
# Since our problem only has equality constraints, our upper and lower bounds $u$ and $l$ will both be zero. Ipopt requires that we specify analytical functions that evaluate $\nabla f$ and $\nabla c$. For best performance, the function evaluating the constraint Jacobian typically only evaluates the nonzero elements. To make things simple, we treat the Jacobian as dense. 
# 
# This homework problem will give you valuable experience in setting up the optimization problems in a way that can be passed to off-the-shelf NLP solvers like Ipopt.
# 
# ## The Problem
# You likely have already seen the cartpole swing-up problem previously.The system is comprised of a pendulum attached to a cart, where forces can only be applied to the cart. The goal is to balance the pendulum above the cart. The system dynamics can be written as:
# 
# $$ x = \begin{bmatrix} y \\ \theta \\ v \\ \omega \end{bmatrix}, \quad \dot{x} = \begin{bmatrix} \dot{q} \\ \ddot{q} \end{bmatrix}, \quad
# q = \begin{bmatrix} y \\ \theta \end{bmatrix}, \quad
# \ddot{q} = -H^{-1} (C \dot{q} + G - B u)
# 
#   where :$ H = \begin{bmatrix} mc + mp & mp l \cos{\theta} \ mp l \cos{\theta}
#   & mp l^2 \end{bmatrix}, \; C = \begin{bmatrix} 0 & -mp \omega l \sin{\theta}
#   \ 0 & 0 \end{bmatrix}, \; G = \begin{bmatrix} 0 \ m_p g l \sin{\theta}
#   \end{bmatrix}, \; B = \begin{bmatrix} 1 \ 0 \end{bmatrix} :$
# 
#   with the following parameters:
# 
#     •  m_p
#        : mass of the pole
# 
#     •  m_c
#        : mass of the cart
# 
#     •  g
#        : gravity
# 
#     •  l
#        : length of the rod
# 
#   Our goal is to move the cart in a way that we get the pendulum to swing from
#   the downward position ([0,0,0,0]) to an upright position ([0,pi,0,0]).
# 
#   We've encapsulated all of the problem information into a struct for
#   convenience (and to avoid polluting our global workspace with uncessary
#   global variables).
# 
#   Developing in External Editor
#   ===============================
# 
#   All of the methods you need to implement in this problem are in external
#   Julia files. Feel free to use a text editor / IDE of your choice (the Julia
#   VSCode extension is the IDE recommended by the Julia community) to write and
#   test these methods. You can use the q2.jl script to run the code, which
#   includes tests that are identical to those in this notebook. We will be
#   running the notebooks for for the autograder, so before you submit make sure
#   this notebook runs as expected and passes the tests (or run test/runtests.jl
#   which will run the autograder).

include("q2_prob.jl")  # Defines a struct containing all of the problem information

prob = CartpoleProblem();

let model = prob.model
    isautograder && return
    global vis = Visualizer()
    set_mesh!(vis, model)
    render(vis)
end

let X = get_initial_trajectory(prob)[1]
    isautograder || visualize!(vis, prob.model, prob.tf, X)
end

#   Part (a): Write Cost Functions (15 pts)
#   =========================================
# 
#   Our first task will be to write methods to evaluate our objective / cost
#   function. We first create a struct that will be responsible for evaluating
#   all the functions we need to pass to Ipopt.

include("sparseblocks.jl")  # SOLUTION
include("q2_nlp.jl")

#   Useful Examples
#   –––––––––––––––––
# 
#   You may find the following code snippets helpful as you complete the methods
#   for the NLP.

let
    # Create NLP
    nlp = NLP(prob)

    # Create a vector of all states and controls
    X,U = get_initial_trajectory(prob)
    Z = packZ(nlp, X, U)

    # Unpack into states and vectors
    X2, U2 = unpackZ(nlp, Z)

    # Get kth state, control
    k = 10
    x = Z[nlp.xinds[k]]
    u = Z[nlp.uinds[k]]

    # Dynamics
    t = nlp.times[k]
    dt = nlp.times[k+1] - nlp.times[k]
    dynamics(nlp.model, x, u, t)

    # Dynamics Jacobian
    A,B = dynamics_jacobians(nlp.model, x, u, t);
end;

#   Objective
#   ===========
# 
#   TASK Finish the following methods defined in this file included in the cell
#   below:
# 
#     •  eval_f (5 pts)
# 
#     •  grad_f! (10 pts)
# 
#   The docstrings for these function are printed below. You will be graded on
#   the number of function and Jacobian evaluations you use. You should avoid
#   unnecessary dynamics and dynamics Jacobian evaluations. You should only need
#   a maximum N + (N-1) evaluations each of the dynamics and dynamics Jacobians
#   for each function.
# 
#   TIP: You may find it helpful to define some helper function that evaluate
#   all of the terms you need upfront. We've provided some example starter code
#   in q2_dynamics.jl. Feel free to include that file and modify as needed. You
#   can also add fields to the NLP struct if you feel the need (the TA's
#   solution uses the provided fields).

# SOLUTION
include("q2_dynamics.jl")

include("q2_cost_methods.jl")

@testset "Q2a" begin                                               # POINTS = 15
    prob = CartpoleProblem()
    nlp = NLP(prob)
    X,U = get_initial_trajectory(prob) 
    Z = packZ(nlp, X, U)

    @testset "eval_f" begin                                        # POINTS = 5
        # Test the cost
        @test eval_f(nlp, Z) ≈ 0.22766546346850902 atol=1e-6       # POINTS = 3
        devals = @dynamicsevals eval_f(nlp, Z)
        @test devals <= 301                                        # POINTS = 0.5
        @test devals <= 201                                        # POINTS = 1
        @test devals >= 200                                        # POINTS = 0.5
    end

    @testset "grad_f" begin                                        # POINTS = 10
        # Test the cost gradient with FiniteDiff
        grad = zero(Z)
        grad_f!(nlp, grad, Z)
        devals = @dynamicsevals grad_f!(nlp, grad, Z)
        jevals = @jacobianevals grad_f!(nlp, grad, Z)
        @test devals <= 301                                        # POINTS = 0.5
        @test devals <= 201                                        # POINTS = 1
        @test devals >= 200                                        # POINTS = 0.5
        @test jevals <= 301                                        # POINTS = 0.5
        @test jevals <= 201                                        # POINTS = 1
        @test jevals >= 200                                        # POINTS = 0.5
        
        grad_fd = FiniteDiff.finite_difference_gradient(x->eval_f(nlp, x), Z)
        @test norm(grad - grad_fd) < 1e-8                          # POINTS = 6
    end    
end;

#   Part (b): Evaluate the Constraints (20 pts)
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   Next, we need to define functions to evaluate our constraints. We should
#   have n + (N-1) n + n constraints, since we have an initial and goal state,
#   and (N-1) dynamics constraints of size n, where n is the size of our state
#   vector (4). The vector should be stacked as follows:
# 
# :$
# 
#   \begin{bmatrix} x1 - x\text{init} \ \frac{h}{6}(f(x1, u1, t1) + 4 f(xm, um,
#   tm) + f(x2, u2, t2) + x1 - x2 \ \vdots \ \frac{h}{6}(f(x{N-1}, u{N-1},
#   t{N-1}) + 4 f(xm, um, tm) + f(xN, uN, tN) + x{N-1} - xN \ xN - x\text{goal}
#   \end{bmatrix} :$
# 
#   TASK: Complete the following functions defined in the file included in the
#   cell below:
# 
#     •  eval_c!(nlp, c, Z)
# 
#     •  jac_c!(nlp, jac, Z)
# 
#   As with the cost functions, you will be graded on how many dynamics function
#   evaluations you use. You should only need N + (N-1) dynamics evaluations for
#   the constraints and N + (N-1) dynamics Jacobian evaluations for the
#   constraint Jacobian.
# 
#   You are NOT allowed to use finite differencing or automatic differentiation
#   in this function. Not only should you be familiar with how to apply the
#   chain rule to get the pieces you need analytically, we already use
#   ForwardDiff to get the dynamics Jacobians, and nesting calls to ForwardDiff
#   usually results in poor performance.
# 
#   TIPS:
# 
#     •  Don't worry about the number of dynamics / Jacobian evaluations to
#        begin with. Do something that works, then worry about
#        "performance."
# 
#     •  Consider writing some helper functions to evaluate all the pieces
#        you need before the loop. These will probably be the same as the
#        ones you needed for the cost functions.
# 
#     •  Write out the derivatives you need by hand using the chain rule.
#        Cache the individual pieces of the chain rule you need and then
#        multiply them together to get the final
# 
#   Jacobians.
# 
#     •  Check intermediate Jacobians (e.g. the Jacobians for a single
#        dynamics constraint) with ForwardDiff or FiniteDiff to make sure
#        you've applied the chain rule correctly, then apply it in a loop.
# 
#   The docstrings for these functions are printed below.

include("q2_constraints.jl")

@testset "Q2b" begin                                              # POINTS = 20
    prob = CartpoleProblem()
    nlp = NLP(prob)
    X,U = get_initial_trajectory(prob) 
    Z = packZ(nlp, X, U)

    resfile = joinpath(@__DIR__, "Q2.jld2")

    @testset "eval_c" begin                                      # POINTS = 8
        # Constraint function
        c = zeros(num_duals(nlp))
        devals = @dynamicsevals eval_c!(nlp, c, Z)
        @test devals <= 301                                      # POINTS = 0.5
        @test devals <= 201                                      # POINTS = 1
        @test devals >= 200                                      # POINTS = 0.5
        @test norm(c - load(resfile, "c0")) < 1e-8               # POINTS = 6
    end
    

    @testset "jac_c" begin                                       # POINTS = 12 
        # Calc constraint Jacobian and check Jacobian evals
        jac = zeros(num_duals(nlp), num_primals(nlp))
        jevals = @jacobianevals jac_c!(nlp, jac, Z)
        devals = @dynamicsevals jac_c!(nlp, jac, Z)
        @test devals <= 301                                      # POINTS = 0.5
        @test devals <= 201                                      # POINTS = 1
        @test devals <= 101                                      # POINTS = 0.5
        @test devals == 0                                        # POINTS = 1
        @test jevals <= 301                                      # POINTS = 0.5
        @test jevals <= 201                                      # POINTS = 1
        @test jevals >= 200                                      # POINTS = 0.5

        # Check constraint Jacobian with FiniteDiff
        jac_fd = zero(jac)
        FiniteDiff.finite_difference_jacobian!(jac_fd, (y,x)->eval_c!(nlp, y, x), Z)
        @test norm(jac - jac_fd) < 1e-6                          # POINTS = 7
    end
end;

#   Part (c): Solving the NLP (5 pts)
#   ===================================
# 
#   Now that we have the methods we need to evaluate our NLP, we can solve it
#   with Ipopt. We use MathOptInterface.jl
#   (https://github.com/jump-dev/MathOptInterface.jl) to interface with the
#   Ipopt solver. Don't worry too much about this interface: we take care of all
#   of the boilerplate code in the file below.
# 
#   You don't need to do anything for this part: if you all of your methods
#   above are correct, your problem should converge in about 30 iterations. If
#   your problem isn't converging, go check your methods above. Remember, the
#   tests aren't perfect and won't catch all of your mistakes. Debugging these
#   types of solvers is a critical skill that takes practice.

include("q2_moi.jl")

prob = CartpoleProblem()
X,U = get_initial_trajectory(prob)
nlp = NLP(prob)
Z0 = packZ(nlp, X, U)
Zsol,solver = solve(Z0, nlp)

isautograder || render(vis)

let X = [Zsol[xi] for xi in nlp.xinds]
    isautograder || visualize!(vis, prob.model, prob.tf, X)
end

@testset "Q2c" begin                                     # POINTS = 5
    Z = copy(Zsol)
    λ = MOI.get(solver, MOI.NLPBlockDual()) # get the duals
    X,U = unpackZ(nlp, Zsol)
    @test norm(X[1] - prob.x0) < 1e-6                    # POINTS = 0.5
    @test norm(X[end] - prob.xf) < 1e-6                  # POINTS = 0.5
    grad = zeros(num_primals(nlp))
    grad_f!(nlp, grad, Z)
    c = zeros(num_duals(nlp))
    eval_c!(nlp, c, Z)
    jac = spzeros(num_duals(nlp), num_primals(nlp))
    jac_c!(nlp, jac, Z)
    @test norm(grad - jac'λ, Inf) < 1e-6                 # POINTS = 2
    @test norm(c, Inf) < 1e-6                            # POINTS = 2
end;

#   Part (d): Track the solution with model error (10 pts)
#   ========================================================
# 
#   Let's now use our trajectory and simulate it on a system with some model
#   mismatch.
# 
#   TASK:
# 
#     1. Generate controller that tracks your optimized trajectories.
# 
#     2. Run your controller on a simulated cartpole with a cart mass of
#        1.5 kg instead of 1 kg. Get it to successfully stabilize. The
#        final stabilized position doesn't have to to be at an x-position
#        of 0. Simulate for at least 10 seconds.
# 
#   TIPS:
# 
#     1. Feel free to use code from previous homeworks.
# 
#     2. It will stabilize with TVLQR
# 
#     3. If your cartpole gets it to the top but doesn't stabilize it for
#        the full 10 seconds, think about how you could design your
#        controller to stabilize it about the unstable equilibrium...

include("q2_controller.jl")

isautograder || render(vis)

# Simulate with a different model
let Zref = copy(Zsol)
    ctrl = gen_controller(nlp, Zref)
    model2 = RobotZoo.Cartpole(1.1, 0.2, 0.5, 9.81)
    Xsim, Usim, tsim = simulate(model2, nlp.x0, ctrl, tf=5nlp.tf, dt=0.005)
    isautograder || visualize!(vis, model2, tsim[end], Xsim)
end

using Random
@testset "Q2d" begin                                                # POINTS = 10
    Random.seed!(1)
    model2 = RobotZoo.Cartpole(1.1, 0.2, 0.5, 9.81)
    ctrl = gen_controller(nlp, Zsol)
    tsim = @elapsed Xsim, Usim, tsim = 
        simulate(model2, nlp.x0, ctrl, tf=5nlp.tf, dt=0.005)
    
    # Test real-time performance
    @test tsim < 5nlp.tf
    
    # Check that it gets to the goal
    @test abs(Xsim[end][1]) < 0.1
    @test abs(Xsim[end][2] - pi) < 1e-2
    @test abs(Xsim[end][3]) < 0.1
    @test abs(Xsim[end][4]) < 1e-2 
    @test norm(Usim[end-10:end], Inf) < 0.3
end;

#   Part (e): EXTRA CREDIT Leveraging sparsity (max 5 pts)
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   NLP solvers like Ipopt or SNOPT are designed to leverage sparsity in the
#   problem, especially in the constraint Jacobian. Right now we're ignoring the
#   sparsity structure in the constraint Jacobian, so the solver iterations are
#   fairly slow. Get up to 5 extra credit points by leveraging the sparsity
#   structure in the constraint Jacobian. You'll need to read up on how to
#   specify the sparsity pattern in the MathOptInterface.jl documentation
#   (http://jump.dev/MathOptInterface.jl/stable/reference/nonlinear/). You
#   should only leverage the sparsity when the use_sparse_jacobian flag in the
#   NLP struct is set to true. We use this flag to compare the solutions between
#   the normal (dense) version and your sparse version. You'll get points for
#   having matching Jacobians, the size of the nonzeros vector you're passing to
#   Ipopt (shoot for a sparsity of less than 2-5%). You'll also get up to 2
#   points for the speedup you get from the solver (the TA solution got a speed
#   up of about 100x).
# 
#   TIPS
# 
#     •  You'll need to modify the MOI.jacobian_structure method in
#        q2_moi.jl
# 
#     •  You'll need to modify the jac_c! method that takes a vector in
#        q2_constraints.jl
# 
#   We will run the following function to calculate your extra credit:

include("q2_tests.jl");

test_extracredit()