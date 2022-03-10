
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
    # SOLTUION
    c[idx] = Z[xi[1]] - nlp.x0
    # END SOLUTION

    # Dynamics
    # SOLUTION
    evaluate_dynamics!(nlp, Z)
    evaluate_midpoints!(nlp, Z)
    # END SOLUTION
    for k = 1:N-1
        idx = idx .+ n
        x1,x2 = Z[xi[k]], Z[xi[k+1]]
        u1,u2 = Z[ui[k]], Z[ui[k+1]]
        t = nlp.times[k]
        h = nlp.times[k+1] - nlp.times[k]
        
        # TASK: Dynamics constraint
        c[idx] .= 0
        
        # SOLUTION
        f1 = nlp.f[k]
        f2 = nlp.f[k+1]
        fm = nlp.fm[k]
        xm,um = nlp.xm[k], nlp.um[k]
        xm = (x1 + x2)/2 + h/8 * (f1 - f2)
        um = (u1 + u2)/2
        c[idx] = h * (f1 + 4fm + f2) / 6 + x1 - x2
        # END SOLUTION
    end

    # TODO: terminal constraint
    idx = idx .+ n
    c[idx] .= 0
    
    # SOLUTION
    c[idx] = Z[xi[N]] - nlp.xf
    # END SOLUTION
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
    # SOLUTION
    for i = 1:n
        jac[i,i] = 1
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]

    # SOLUTION
    evaluate_dynamics_jacobians!(nlp, Z)
    evaluate_midpoint_jacobians!(nlp, Z)
    # END SOLUTION
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
        
        # TODO: Dynamics constraint
        jac_x1 .= 0 
        jac_u1 .= 0 
        jac_x2 .= 0 
        jac_u2 .= 0 

        # SOLUTION
        A1,B1 = nlp.A[k], nlp.B[k]
        A2,B2 = nlp.A[k+1], nlp.B[k+1]
        Am,Bm = nlp.Am[k], nlp.Bm[k]
        Am1, Am2 = nlp.Am[k,2], nlp.Am[k,3]
        Bm1, Bm2 = nlp.Bm[k,2], nlp.Bm[k,3]
        dx1 = h/6 * (A1 + 4*Am*Am1) + I
        du1 = h/6 * (B1 + 4*(Am*Bm1 + Bm/2))
        dx2 = h/6 * (A2 + 4*Am*Am2) - I
        du2 = h/6 * (B2 + 4*(Am*Bm2 + Bm/2))
        jac_x1 .= dx1
        jac_u1 .= du1
        jac_x2 .= dx2
        jac_u2 .= du2
        # END SOLUION
    end
    idx = idx .+ n 
    
    # TODO: Terminal constraint
    # SOLUTION
    for i = 1:n
        jac[idx[i], xi[end][i]] = 1
    end
end


# EXTRA CREDIT: Specify the sparsity directly in the nonzeros vector
#               Read the MathOptInterface documentation!
function jac_c!(nlp::NLP{n,m}, jacvec::AbstractVector, Z) where {n,m}
    # SOLUTION
    jac = NonzerosVector(jacvec, nlp.blocks)

    for i = 1:n
        jac[i,i] = 1
    end

    model = nlp.model
    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    evaluate_dynamics_jacobians!(nlp, Z)
    evaluate_midpoint_jacobians!(nlp, Z)
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
        
        A1,B1 = nlp.A[k], nlp.B[k]
        A2,B2 = nlp.A[k+1], nlp.B[k+1]
        f1,f2 = nlp.f[k], nlp.f[k+1]
        xm,um = nlp.xm[k], nlp.um[k]
        Am,Bm = nlp.Am[k], nlp.Bm[k]
        Am1, Am2 = nlp.Am[k,2], nlp.Am[k,3]
        Bm1, Bm2 = nlp.Bm[k,2], nlp.Bm[k,3]
        dx1 = h/6 * (A1 + 4*Am*Am1) + I
        du1 = h/6 * (B1 + 4*(Am*Bm1 + Bm/2))
        dx2 = h/6 * (A2 + 4*Am*Am2) - I
        du2 = h/6 * (B2 + 4*(Am*Bm2 + Bm/2))
        jac_x1 .= dx1
        jac_u1 .= du1
        jac_x2 .= dx2
        jac_u2 .= du2
    end
    idx = idx .+ n 
    
    for i = 1:n
        jac[idx[i], xi[end][i]] = 1
    end
end

# SOLUTION
function initialize_sparsity!(nlp)
    n,m = size(nlp) 
    blocks = nlp.blocks
    for i = 1:n
        setblock!(blocks, i, i)
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    for k = 1:nlp.N-1
        idx = idx .+ n
        
        setblock!(blocks, idx, xi[k])
        setblock!(blocks, idx, ui[k])
        setblock!(blocks, idx, xi[k+1])
        setblock!(blocks, idx, ui[k+1])
    end
    idx = idx .+ n 
    
    for i = 1:n
        setblock!(blocks, idx[i], xi[end][i])
    end
end