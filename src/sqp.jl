using Printf

function solve_sqp!(nlp, Z, λ;
        iters=100,
        qp_solver=:osqp,
        adaptive_reg::Bool=false,
        verbose=0,
        eps_primal=1e-6,
        eps_dual=1e-6,
        eps_fn=sqrt(eps_primal),
        gn::Bool=true,
        enable_soc::Bool=true
    )

    # Initialize solution
    qp = TOQP(size(nlp)..., num_eq(nlp), 0)

    reg = 1e-6 
    dZ = zero(Z)

    # Line Search 
    μ = 10.0
    ḡ = zero(Z)
    c̄ = zero(λ)
    Z̄ = zero(Z)

    for iter = 1:iters
        ## Check the residuals and cost
        res_p = primal_residual(nlp, Z, λ)
        res_d = dual_residual(nlp, Z, λ)
        J = eval_f(nlp, Z)
        verbose > 0 && @printf("Iteration %d: cost = %0.2f, res_p = %0.2e, res_d = %0.2e,", iter, J, res_p, res_d)

        if res_p < eps_primal && res_d < eps_dual 
            verbose > 0 && println()
            break
        end

        # Build QP
        build_qp!(qp, nlp, Z, λ, gn=gn)
        
        # Solve the QP
        dZ, dλ = solve_qp!(qp, reg)
        qp_res_p = norm(qp.Q*dZ + qp.q + qp.A'dλ)
        qp_res_d = norm(qp.A*dZ - qp.b)
        verbose > 1 && @printf(" qp_res_p = %0.2e, qp_res_d = %0.2e, δλ = %0.2e", qp_res_p, qp_res_d, norm(dλ,Inf))

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
        phi0 = J0 + μ * norm(c0, 1)
        dphi0 = grad0'dZ - μ * norm(c0, 1)
        phi = Inf

        soc = false
        τ = 0.5
        η = 1e-2
        # dphi1 = gradient(ϕ, nlp, Z)'dZ
        verbose > 2 && @printf("\n   ϕ0: %0.2f, ϕ′: %0.2e, %0.2e\n", phi0, dphi0, dphi1)
        for i = 1:10
            Z̄ .= Z .+ α .* dZ
            # λbar = λ - α*dλ
            # phi = ϕ(nlp, Z̄)
            # res_d = dual_residual(nlp, Z̄, λbar)
            # res_p = primal_residual(nlp, Z̄, λbar)
            # verbose > 2 && @printf("   ls iter: %d, Δϕ: %0.2e, ϕ′: %0.2e, res_p: %0.2e, res_d: %0.2e\n", 
            #     i, phi-phi0, η*α*dphi0, res_p, res_d)
            eval_c!(nlp, c̄, Z̄)
            phi = eval_f(nlp, Z̄) + μ * norm(c̄, 1)

            if phi < phi0 + η*α*dphi0
                reg = 1e-6 
                break
            elseif α == 1.0 && enable_soc
                A = qp.A
                psoc = -A'*((A*A')\(c̄))
                Z̄ .= Z .+ dZ .+ psoc
                eval_c!(nlp, c̄, Z̄)
                phi = eval_f(nlp, Z̄) + μ * norm(c̄, 1)
                if phi < phi0 + η*α*dphi0
                    soc = true
                    reg = 1e-6
                    break
                else
                    α *= τ
                end
            else
                α *= τ
            end
            if i == 10
                reg *= 10
                α = 0
                Z̄ .= Z
            end
        end
        Z .= Z̄
        λ .= λ .- α .* dλ
        # A = qp.A
        # λ .= (A*A')\(A*qp.q)
        verbose > 0 && @printf("   α = %0.2f, ΔJ: %0.2e, Δϕ: %0.2e, reg: %0.2e, pen: %d, soc: %d\n", 
            α, J - eval_f(nlp, Z), phi0 - phi, reg, μ, soc)
        if α == 0.0
            @warn "line search failed"
            # build_qp!(qp, nlp, Z, λ, gn=false)
            # dZ_fn, = solve_qp!(qp, reg)
            # return dZ, dZ_fn
        end

    end 
    return Z, λ, qp 
end

function minimum_penalty(Q,q,c, dZ; ρ=0.5)
    a = dot(dZ, Q, dZ)
    σ = a > 0
    num = q'dZ + σ*0.5*a
    den = (1-ρ) * norm(c)
    return num/den
end

function meritgrad(μ, g, c, Z, dZ)
    return g'dZ - μ * norm(c, 1)
end

function solve_qp!(qp::TOQP, reg=0.0)
    N,M = num_primals(qp), num_duals(qp)
    K = [qp.Q + reg*I qp.A'; qp.A -reg*I]
    t = [-qp.q; qp.b]
    dY = K\t
    dZ = dY[1:N]
    dλ = dY[N+1:N+M]
    return dZ, dλ
end