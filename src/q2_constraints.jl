
# TASK: Complete the following methods
#       eval_c!
#       jac_c!
"""
    eval_c!(nlp, c, Z)

Evaluate the equality constraints at `Z`, storing the result in `c`.
The constraints should be ordered as follows: 
1. Initial condition ``x_1 = x_\\text{init}``
2. Hermite Simpson Dynamics: ``\\frac{h}{6} (f(x_k, u_k) + 4 f(x_m, u_m) + f(x_{k+1}, u_{k+1})) + x_k - x_{k+1} = 0``
3. Terminal constraint ``x_N = x_\\text{goal}``

Consider leveraging the caches in `nlp` to evaluate the dynamics and the midpoints 
before the main loop, so that you can making redundant calls to the dynamics.

Remember, you will loose points if you make more dynamics calls than necessary. 
Start with something that works, then think about how to eliminate any redundant 
dynamics calls.
"""
function eval_c!(nlp::NLP{n,m}, c, Z) where {n,m}
    N = nlp.N
    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]

    # TODO: initial condition

    # Dynamics
    for k = 1:N-1
        idx = idx .+ n
        x1,x2 = Z[xi[k]], Z[xi[k+1]]
        u1,u2 = Z[ui[k]], Z[ui[k+1]]
        t = nlp.times[k]
        h = nlp.times[k+1] - nlp.times[k]
        
        # TASK: Dynamics constraint
        c[idx] .= 0
        
    end

    # TODO: terminal constraint
    idx = idx .+ n
    c[idx] .= 0
    
    return c
end

"""
    jac_c!(nlp, jac, Z)

Evaluate the constraint Jacobian, storing the result in the matrix `jac`.
You will need to apply the chain rule to calculate the derivative of the dynamics
constraints with respect to the states and controls at the current and next time 
steps.

### Use of automated differentiation tools
You are not allowed to use automatic differentiation methods for this function. 
You are only allowed to call `dynamics_jacobians` (which under the hood does use
ForwardDiff). You are allowed to check your answer with these tools, but your final 
solution should not use them.
"""
function jac_c!(nlp::NLP{n,m}, jac, Z) where {n,m}
    # TODO: Initial condition

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]

    for k = 1:nlp.N-1
        idx = idx .+ n
        x1,x2 = Z[xi[k]], Z[xi[k+1]]
        u1,u2 = Z[ui[k]], Z[ui[k+1]]
        t = nlp.times[k]
        h = nlp.times[k+1] - nlp.times[k]
        
        jac_x1 = view(jac, idx, xi[k])
        jac_u1 = view(jac, idx, ui[k])
        jac_x2 = view(jac, idx, xi[k+1])
        jac_u2 = view(jac, idx, ui[k+1])
        
    end
    idx = idx .+ n 
    
    # TODO: Terminal constraint
end


# EXTRA CREDIT: Specify the sparsity directly in the nonzeros vector
#               Read the MathOptInterface documentation!
function jac_c!(nlp::NLP{n,m}, jacvec::AbstractVector, Z) where {n,m}
end
