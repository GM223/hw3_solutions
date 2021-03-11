using Printf

"""
    solve_sqp!(nlp, Z, λ; kwargs...)

Solve the trajectory optimization problem specified by `nlp` using Sequential Quadratic Programming, given the initial 
guess for the primal variables `Z` and `λ`.
"""
# TASK: Complete the SQP method
"""
    solve_sqp!(nlp, Z, λ; kwargs...)

Solve the trajectory optimization problem specified by `nlp` using Sequential Quadratic Programming, given the initial 
guess for the primal variables `Z` and `λ`.
"""
function solve_sqp!(nlp, Z0, λ0;
        iters=100,                   # max number of iterations
        verbose=0,                   # verbosity level
        eps_primal=1e-6,             # primal feasibility tolerance
        eps_dual=1e-4,               # dual feasibility tolerance
        eps_fn=sqrt(eps_primal),     # 
        gn::Bool=true,               # use Gauss-Newton approximation
        enable_soc::Bool=true,       # enable Second-Order-Corrections during the line search
        ls_iters=10,                 # max number of line search iterations
        reg_min=1e-6,                # minimum regularization
    )
    t_start = time_ns()

    # Initialize solution
    Z = deepcopy(Z0)
    λ = deepcopy(λ0)
    qp = TOQP(nlp)

    # Line Search parameters
    μ = 10.0
    ḡ = zero(Z)
    c̄ = zero(λ)
    Z̄ = zero(Z)
    dZ = zero(Z)
    reg = reg_min 
    
    stats = Dict(
        :cost => Float64[],
        :viol_primal => Float64[],  # constraint violation
        :viol_dual => Float64[],    # stationarity
        :time => Float64[]
    )

    for iter = 1:iters
        ## Check the residuals and cost
        res_p = primal_residual(nlp, Z, λ)
        res_d = dual_residual(nlp, Z, λ)
        J = eval_f(nlp, Z)
        push!(stats[:viol_primal], res_d)
        push!(stats[:viol_dual], res_p)
        verbose > 0 && @printf("Iteration %d: cost = %0.2f, res_p = %0.2e, res_d = %0.2e,", iter, J, res_p, res_d)

        # Termination conditions
        if res_p < eps_dual && res_d < eps_primal
            verbose > 0 && println()
            break
        end

        # Build QP
        build_qp!(qp, nlp, Z, λ, gn=gn)
        
        # Solve the QP
        dZ, dλ = solve_qp!(qp, reg)

        # Update penalty paramter
        μ_min = minimum_penalty(qp.Q, qp.q, qp.b, dZ)
        if μ < μ_min
            μ = μ_min*1.1
        end

        # Line Search
        α = 1.0
        J0 = eval_f(nlp, Z)
        grad0 = qp.q
        c0 = qp.b
        phi0 = J0 + μ * norm(c0, 1)            # merit function
        dphi0 = grad0'dZ - μ * norm(c0, 1)     # gradient of the merit function (Nocedal & Wright Theorem 18.2)
        phi = Inf
        
        push!(stats[:cost], J0)

        soc = false
        τ = 0.5
        η = 1e-2
        # verbose > 2 && @printf("\n   ϕ0: %0.2f, ϕ′: %0.2e, %0.2e\n", phi0, dphi0, dphi1)
        for i = 1:10
            # Calculate merit function at new step
            Z̄ .= Z .+ α .* dZ
            eval_c!(nlp, c̄, Z̄)
            phi = eval_f(nlp, Z̄) + μ * norm(c̄, 1)

            # Check Armijo
            if phi < phi0 + η*α*dphi0
                reg = max(reg /= 100, reg_min)
                break
            # Try second-order correction
            elseif α == 1.0 && enable_soc
                A = qp.A
                psoc = -A'*((A*A')\(c̄))
                Z̄ .= Z .+ dZ .+ psoc
                eval_c!(nlp, c̄, Z̄)
                phi = eval_f(nlp, Z̄) + μ * norm(c̄, 1)
                if phi < phi0 + η*α*dphi0
                    soc = true
                    reg = max(reg /= 100, reg_min)
                    break
                else
                    α *= τ
                end
            else
                α *= τ
            end

            # Line search failure
            if i == 10
                reg *= 10    # increase regularization
                α = 0        # don't take a step
                Z̄ .= Z 
                @warn "line search failed"
            end
        end
        # Apply step
        Z .= Z̄
        λ .= λ .- α .* dλ

        # Output
        verbose > 0 && @printf("   α = %0.2f, ΔJ: %0.2e, Δϕ: %0.2e, reg: %0.2e, pen: %d, soc: %d\n", 
            α, J - eval_f(nlp, Z), phi0 - phi, reg, μ, soc)
        push!(stats[:time], (time_ns() - t_start) / 1e6)  # ms
    end
    push!(stats[:time], (time_ns() - t_start) / 1e6)  # ms
    push!(stats[:cost], eval_f(nlp, Z))
    return Z, λ, stats 
end


"""
    minimum_penalty(Q,q,c, dZ; ρ)

Calculate the minimum penalty needed for the exact penalty function, where `Q` is the Hessian, `q` is the gradient, 
`c` is the constraint violation, and `dZ` is the search direction. (See Nocedal & Wright Eq. 18.36)
"""
function minimum_penalty(Q,q,c, dZ; ρ=0.5)
    a = dot(dZ, Q, dZ)
    σ = a > 0
    num = q'dZ + σ*0.5*a
    den = (1-ρ) * norm(c)
    return num/den
end
