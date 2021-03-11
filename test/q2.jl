using Test
using Statistics

function convergence_rate(v::AbstractVector{T}) where T
    n = length(v)
    r = zeros(n-1)
    for i = 1:n-1
        if v[i] > one(T)
            @inbounds a,b = v[i], v[i+1]
        else
            @inbounds a,b = v[i+1], v[i]
        end
        @inbounds r[i] = abs(log10(a) / log10(b))
    end
    return r
end

@testset "Q2" begin
qp = TOQP(nlp)
Zrand = randn(num_primals(nlp))
λrand = randn(num_duals(nlp))
Xrand,Urand = unpackZ(nlp, Zrand)

@testset "Build QP" begin
# test build_qp
Ftest = RobotDynamics.DynamicsJacobian(model) 
discrete_jacobian!(RK4, Ftest, model, Xrand[1], Urand[1], 0.0, dt)

build_qp!(qp, nlp, Zrand, λrand)
@test isdiag(qp.Q)
@test qp.Q[end,end] ≈ Qf[end,end]
@test qp.Q[1,1] ≈ Q[1,1]
@test qp.Q[5,5] ≈ R[1,1]

@test qp.q[1:n] ≈ (Q*(Xrand[1] - xf) - λrand[1:n] - Ftest.A'λrand[n .+ (1:n)])

@test qp.A[1:n,1:n] ≈ I(n)
Ftest = RobotDynamics.DynamicsJacobian(model) 
discrete_jacobian!(RK4, Ftest, model, Xrand[1], Urand[1], 0.0, dt)
@test qp.A[n .+ (1:n), 1:n] ≈ Ftest.A
@test qp.A[n .+ (1:n), n .+ (1:m)] ≈ Ftest.B
@test qp.A[n .+ (1:n), (n+m) .+ (1:n)] ≈ -I(n)

@test qp.b[1:n] ≈ -(Xrand[1] - x0)
@test qp.b[n .+ (1:n)] ≈ -(discrete_dynamics(RK4, model, Xrand[1], Urand[1], 0.0, dt) - Xrand[2])
@test qp.b[end-n+1:end] ≈ -(Xrand[T] - xf)
end

@testset "QP Solve" begin
# test qp solve
dZ_, dλ_ = solve_qp!(qp)
@test qp.A*dZ_ ≈ qp.b atol=1e-8
@test qp.Q*dZ_ + qp.q + qp.A'dλ_ ≈ zero(Zrand) atol=1e-8
end


@testset "SQP Solve" begin
# solving the problem 
Ztest, λtest, stats_test = solve_sqp(nlp, Z, λ)
@test primal_residual(nlp, Ztest, λtest) < 1e-3
@test dual_residual(nlp, Ztest, λ) < 1e-6
@test Ztest[1:n] ≈ zeros(n) atol=1e-5
@test Ztest[end-n+1:end] ≈ xf atol=1e-5

@test length(stats_test[:cost]) > 1
@test length(stats_test[:cost]) < 400
end

@testset "Solver Convergence" begin
# test convergence
@test length(stats_fn[:cost]) < length(stats_gn[:cost])
fn_iters = length(stats_fn[:viol_primal])
fn_cr = mean(convergence_rate(stats_fn[:viol_primal])[fn_iters ÷ 2:end])
gn_cr = mean(convergence_rate(stats_gn[:viol_primal])[fn_iters ÷ 2:end])
@test gn_cr < fn_cr
@test 1.2 < fn_cr < 2
end

@testset "TVLQR tracking" begin
# TVLQR 
@test tsim[end] >= 2*tf
@test Xsim[end][2] ≈ pi atol=1e-1
@test Xsim[end][3:4] ≈ zeros(2) atol=1e-1
end

end