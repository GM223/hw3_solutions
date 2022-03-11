function MOI.eval_objective(nlp::NLP, x)
    eval_f(nlp, x)
end

function MOI.eval_objective_gradient(nlp::NLP, grad_f, x)
    grad_f!(nlp, grad_f, x)
    return nothing
end

function MOI.eval_constraint(nlp::NLP,g,x)
    eval_c!(nlp, g, x)
    return nothing
end

function MOI.eval_constraint_jacobian(nlp::NLP, vec, x)
    if nlp.use_sparse_jacobian
        # Calls the Jacobian with a Vector instead of a Matrix
        jac_c!(nlp, vec, x)
    else
        n_nlp, m_nlp = num_primals(nlp), num_duals(nlp)
        jac = reshape(vec, m_nlp, n_nlp)
        jac_c!(nlp, jac, x)
    end
    return nothing
end

function MOI.features_available(prob::NLP)
    return [:Grad, :Jac]
end

MOI.initialize(prob::NLP, features) = nothing

function MOI.jacobian_structure(nlp::NLP)
    if nlp.use_sparse_jacobian
        # EXTRA CREDIT: return the Jacobian sparsity structure (see MathOptInterface docs)
        rc = Tuple{Int,Int}[]
        return rc
    else
        return vec(Tuple.(CartesianIndices(zeros(num_duals(nlp), num_primals(nlp)))))
    end
end

"""
    solve(x0, nlp::NLP; tol, c_tol, max_iter)

Solve the NLP `nlp` using Ipopt via MathOptInterface, providing `x0` as an initial guess.

# Keyword Arguments
The following arguments are sent to Ipopt
* `tol`: overall optimality tolerance
* `c_tol`: constraint feasibility tolerance
* `max_iter`: maximum number of solver iterations
"""
function solve(x0,prob::NLP;
        tol=1.0e-6,c_tol=1.0e-6,max_iter=10000)
    n_nlp, m_nlp = num_primals(prob), num_duals(prob)
    x_l, x_u = fill(-Inf,n_nlp), fill(+Inf,n_nlp)
    c_l, c_u = zeros(m_nlp), zeros(m_nlp) 

    println("Creating NLP Block Data...")
    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    has_objective = true
    block_data = MOI.NLPBlockData(nlp_bounds, prob, has_objective)

    println("Creating Ipopt...")
    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    solver.options["max_cpu_time"] = 60.0

    x = MOI.add_variables(solver, n_nlp)

    println("Adding constraints...")
    for i = 1:n_nlp
        # xi = MOI.VariablePrimal(x[i])
        MOI.add_constraint(solver, x[i], MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, x[i], MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    println("starting Ipopt Solve...")
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res, solver
end