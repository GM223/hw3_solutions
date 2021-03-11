using Test
using Statistics

@testset "Q1" begin
p = [zeros(n,n) for k = 1:T]
p = [zeros(n) for k = 1:T]
K = [zeros(m,n) for k = 1:T-1]
d = [zeros(m) for k = 1:T-1]
Xtest = deepcopy(X0)
Utest = deepcopy(U0)

@testset "Line Problem" begin
# Problem 
@test prob.obj[1].Q ≈ Q
@test prob.obj[end].Q ≈ Qf
@test prob.obj[1].q ≈ -Q*xgoal
@test prob.obj[end].q ≈ -Qf*xgoal
@test prob.x0 ≈ x0
end

@testset "Line Problem solution" begin
# Line trajectory solve
@test rand(X0) ≈ x0
@test rand(U0) ≈ uhover
Xtest, Utest, Ktest, Ptest = solve_ilqr(prob, X0, U0)
J = cost(obj,Xtest, Utest)
@test cost(obj, X0, U0) > J 
@test J < 500 

ΔJ_ = backwardpass!(prob, P, p, K, d, Xtest, Utest)
@test ΔJ_ < 1e-6 
@test norm(K - Ktest) < 1e-2
@test maximum(norm.(d,Inf)) < 1e-3

Jn_, α_ = forwardpass!(prob, Xtest, Utest, K, d, ΔJ_, cost(obj, Xtest, Utest))
@test (J - Jn_) < 1e-6

@test norm(Xtest[end] - xgoal) < 1
@test rad2deg(norm([x[3] for x in Xtest], Inf)) < 90
end

@testset "Flip reference" begin
# Reference trajectory
Xref_ = flip_reference()
Xref_ = hcat(Vector.(Xref)...)
@test mean(diff(Xref_[1,1:21])) ≈ 3/20 atol=1e-1
@test mean(Xref_[4,1:20]) ≈ 6.0 atol=1e-1 
@test mean(Xref_[1,21:40]) ≈ 0 atol=1e-1
@test mean(diff(Xref_[1,41:end])) ≈ 3/20 atol=1e-1
@test mean(Xref_[4,41:end]) ≈ 6.0 atol=1e-1 

@test Xref_[3,1] ≈ 0 atol=1e-6
@test Xref_[3,end] ≈ -2pi atol = 1e-6
@test std(diff(Xref_[6,21:40])) < 0.1
@test std(diff(Xref_[4,21:40])) < 0.1
end

@testset "Flip objective" begin
Xref_ = flip_reference()

# Flip objective
for k = 1:T-1
    @test obj_flip[1].Q ≈ obj_flip[k].Q
    @test obj_flip[k].q ≈ -Q*Xref_[k]
end
@test obj_flip[T].Q ≈ Qf
end

@testset "Flip solution" begin
# Flip solve
prob_test = Problem(model, obj_flip, tf, x0)
Xtest, Utest, Ktest, Ptest = solve_ilqr(prob_test, X0, U0)

ΔJ_ = backwardpass!(prob_test, P, p, K, d, Xtest, Utest)
@test ΔJ_ < 1e-6 
@test norm(K - Ktest) < 1e-2
@test maximum(norm.(d,Inf)) < 1e-3
@test rad2deg(abs(Xtest[end][3] + 2pi)) < 5
@test maximum(norm(Ktest - Klqr, Inf)) < 1e-3

end

@testset "Tracking" begin
Xref_ = flip_reference()

xgoal2 = Xref_[end]
@test norm((Xlqr[end] - xgoal2)[1:3]) < 0.2
@test norm((Xopen[end] - xgoal2)[1:3]) > 1.0
end

end