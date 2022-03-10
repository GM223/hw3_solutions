
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

function test_solution()
    prob = CartpoleProblem()
    X,U = get_initial_trajectory(prob)
    local nlp = NLP(prob)
    Z0 = packZ(nlp, X, U)
    local Zsol 
    Zsol,solver = solve(Z0, nlp)
    test_solution(nlp, Zsol, solver)
end
function test_solution(nlp, Zsol, solver)
    Z = copy(Zsol)
    λ = MOI.get(solver, MOI.NLPBlockDual()) # get the duals
    X,U = unpackZ(nlp, Zsol)
    @test norm(X[1] - nlp.x0) < 1e-6                    # POINTS = 0.5
    @test norm(X[end] - nlp.xf) < 1e-6                  # POINTS = 0.5
    grad = zeros(num_primals(nlp))
    grad_f!(nlp, grad, Z)
    c = zeros(num_duals(nlp))
    eval_c!(nlp, c, Z)
    jac = spzeros(num_duals(nlp), num_primals(nlp))
    jac_c!(nlp, jac, Z)
    @test norm(grad - jac'λ, Inf) < 1e-6                 # POINTS = 2
    @test norm(c, Inf) < 1e-6                            # POINTS = 2
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

function test_extracredit()
    prob = CartpoleProblem()
    X,U = get_initial_trajectory(prob)
    nlp_dense = NLP(prob, use_sparse_jacobian=false)
    nlp_sparse = NLP(prob, use_sparse_jacobian=true)
    Z0 = packZ(nlp_sparse, X, U)

    tdense = @elapsed Zdense, = solve(copy(Z0), nlp_dense)
    tsparse = @elapsed Zsparse,solver = solve(Z0, nlp_sparse)

    # Check solution is the same
    @test norm(Zsparse - Zdense, Inf) < 1e-5
    test_solution(nlp_sparse, Zsparse, solver)

    # Give up to 3 points for having a small nonzeros vector
    println("EXTRA CREDIT POINTS:")
    points = 0
    rc_sparse = MOI.jacobian_structure(nlp_sparse)
    rc_dense = MOI.jacobian_structure(nlp_dense)

    # Check that the Jacobian is the same
    r = [rc[1] for rc in rc_sparse]
    c = [rc[2] for rc in rc_sparse]
    nnzv = zeros(length(rc_sparse))
    MOI.eval_constraint_jacobian(nlp_sparse, nnzv, Z0)
    jac = zeros(num_duals(nlp_dense), num_primals(nlp_dense))
    MOI.eval_constraint_jacobian(nlp_dense, jac, Z0)
    jac_sparse = sparse(r, c, nnzv)
    jacobians_are_equal = jac_sparse ≈ jac
    if jacobians_are_equal
        println("  Got 1 point for having matching Jacobians")
        points += 1
    end

    # Check sparsity pecentage
    nnz_sparse = nnz(jac_sparse)
    nnz_dense = length(jac)
    percent_nonzeros = nnz_sparse / nnz_dense * 100
    if percent_nonzeros < 5
        println("  Got 1 point for having less 5% sparsity (actual = $(round(percent_nonzeros,digits=2))%)")
        points += 1
    end
    if percent_nonzeros < 2
        println("  Got 1 point for having less 2% sparsity (actual = $(round(percent_nonzeros,digits=2))%)")
        points += 1
    end

    # Check solve time
    speedup = tdense / tsparse
    if 5 < speedup < 20 
        println("  Got 1 point for a 5x speedup (actual = $(round(speedup,digits=2))x)")
        points += 1
    elseif 20 <= speedup 
        println("  Got 2 points for 20x speedup (actual = $(round(speedup,digits=2))x)")
        points += 2
    end
    println("TOTAL EXTRA CREDIT POINTS = ", points)
    return points
end