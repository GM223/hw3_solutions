function MOI.eval_objective(prob::NLP, x)
    eval_f(prob, x)
end

function MOI.eval_objective_gradient(prob::NLP, grad_f, x)
    grad_f!(prob, grad_f, x)
    return nothing
end

function MOI.eval_constraint(prob::NLP,g,x)
    eval_c!(prob, g, x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::NLP, vec, x)
    n_nlp, m_nlp = num_primals(prob), num_duals(prob)
    jac = reshape(vec, m_nlp, n_nlp)
    # ForwardDiff.jacobian!(reshape(jac,m_nlp,n_nlp), (c,x) -> eval_c!(nlp, c, x), zeros(m_nlp), x)
    jac_c!(nlp, jac, x)
    return nothing
end

function MOI.features_available(prob::NLP)
    return [:Grad, :Jac]
end

MOI.initialize(prob::NLP, features) = nothing
MOI.jacobian_structure(nlp::NLP) = vec(Tuple.(CartesianIndices(zeros(num_duals(nlp), num_primals(nlp)))))

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