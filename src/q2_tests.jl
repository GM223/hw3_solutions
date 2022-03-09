
function test_costs()
    prob = CartpoleProblem()
    nlp = NLP(prob)
    X,U = get_initial_trajectory(prob) 
    Z = packZ(nlp, X, U)
     
    # Test the cost
    @test eval_f(nlp, Z) ≈ 0.22766546346850902 atol=1e-6
    devals = @dynamicsevals eval_f(nlp, Z)
    @test devals <= 301
    @test devals <= 201
    @test devals >= 200

    # Test the cost gradient with FiniteDiff
    grad = zero(Z)
    grad_f!(nlp, grad, Z)
    devals = @dynamicsevals grad_f!(nlp, grad, Z)
    jevals = @jacobianevals grad_f!(nlp, grad, Z)
    @test devals <= 301
    @test devals <= 201
    @test devals >= 200
    @test jevals <= 301
    @test jevals <= 201
    @test jevals >= 200

    grad_fd = FiniteDiff.finite_difference_gradient(x->eval_f(nlp, x), Z)
    @test norm(grad - grad_fd) < 1e-8
end

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

function test_tracking()
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
end