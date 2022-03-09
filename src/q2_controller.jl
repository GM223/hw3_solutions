function tvlqr(A,B,Q,R,Qf)
    # Extract some variables
    T = length(A)+1
    n,m = size(B[1])
    P = [zeros(n,n) for k = 1:T]
    K = [zeros(m,n) for k = 1:T-1]
    
    P[end] .= Qf
    for k = reverse(1:T-1) 
        K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
        P[k] .= Q + A[k]'P[k+1]*A[k] - A[k]'P[k+1]*B[k]*K[k]
    end
    
    return K,P
end

function lqr(A,B,Q,R; max_iters=200, tol=1e-6)
    P = copy(Q)
    n,m = size(B)
    K = zeros(m,n)
    K_prev = copy(K)
    
    P = copy(Q)
    for k = 1:max_iters
        K .= (R + B'P*B) \ (B'P*A)
        P .= Q + A'P*A - A'P*B*K
        if norm(K-K_prev,Inf) < tol
            println("Converged in $k iterations")
            return K
        end
        K_prev .= K
    end
    return K * NaN
end

"""
    LQRController

A TVLQR controller that tracks the trajectory specified by `Xref` and `Uref`
using the linear feedback gains `K`.
"""
struct LQRController
    K::Vector{Matrix{Float64}}
    Xref::Vector{Vector{Float64}}
    Uref::Vector{Vector{Float64}}
    times::Vector{Float64}
end
get_k(controller, t) = searchsortedlast(controller.times, t)

function get_control(ctrl::LQRController, x, t)
    k = get_k(ctrl, t)
    K = ctrl.K[k]
    return ctrl.Uref[k] - K*(x - ctrl.Xref[k])
end


"""
    gen_controller(nlp, Zref)

Create a controller that tracks the output of the NLP solver, `Zref`.
The `ctrl` object you output should support a function with the following signature:

    get_control(ctrl, x, t)

that returns the control `u` given the state vector `x` and time `t`.

You are free to implement any controller that satisfies this signature 
(LQR, TVLQR, MPC, a learned policy, etc.) The only requirements are that it achieves the 
swing-up for a cartpole mass of 1.5 kg (the reference was designed with a mass of 1.0 kg),
and runs in faster than real time.

Before you try anything crazy, consider the simplest controller that can probably achieve 
this goal.
"""
function gen_controller(nlp, Zref)
    # TASK: Build a controller that tracks `Zref`
    ctrl = NullController(nlp.model)

    # SOLUTION:
    # Generate A,B matrices
    N = nlp.N
    n,m = state_dim(nlp.model), control_dim(nlp.model)
    Xref,Uref = unpackZ(nlp, Zref)
    A = [zeros(n,n) for k = 1:N-1]
    B = [zeros(n,m) for k = 1:N-1]
    for k = 1:N-1
        t = nlp.times[k]
        dt = nlp.times[k+1] - nlp.times[k]
        Ak,Bk = discrete_jacobian(nlp.model, Xref[k], Uref[k], t, dt)
        A[k] .= Ak
        B[k] .= Bk
    end

    Jstage = nlp.obj[1]
    Jterm = nlp.obj[end]
    Q = Jstage.Q
    R = Jstage.R * 1e-2
    Qf = Jterm.Q

    # Get the infinite-horizon gain to stabilize it once it gets to the top
    Kinf = lqr(A[end], B[end], Matrix(Qf), Matrix(R), max_iters=2000)

    # Solve for the TVLQR gains
    K, = tvlqr(A,B,Q,R,Qf)

    # Add the infinite horizon gain
    push!(K, Kinf)
    Uref[end] = [0]

    # Build the controller
    ctrl = LQRController(K, Xref, Uref, nlp.times);

    # Return the controller
    # must support `get_control(ctrl, x, t)`
    return ctrl
end