
# TASK: Finish the following methods
#       eval_f
#       grad_f!
#       hess_f!

"""
    eval_f(nlp, Z)

Evaluate the objective, returning a scalar. The continuous time objective is of the form:

```math
\\int_{t0}^{tf} \\ell(x(t), u(t)) dt 
```

You need to approximate this with an integral of the form:
```math
\\sum_{k=1}^{N-1} \\frac{h}{6}(\\ell(x_k,u_k) + 4\\ell(x_m, u_m) + \\ell(x_{k+1}, u_{k+1}))
```

where
```math
x_m = \\frac{1}{2} (x_1 + x_2) + \\frac{h}{8}(f(x_1, u_1, t) - f(x_2, u_2, t + h))
```
and
```math
u_m = \\frac{1}{2} (u_1 + u_2)
```
"""
function eval_f(nlp::NLP, Z) 
    # TASK: compute the objective value (cost)
    J = NaN
    
    return J
end

"""
    grad_f!(nlp, grad, Z)

Evaluate the gradient of the objective at `Z`, storing the result in `grad`.
"""
function grad_f!(nlp::NLP{n,m}, grad, Z) where {n,m}
    ix,iu = nlp.xinds, nlp.uinds

    Jstage = nlp.stagecost
    Jterm = nlp.termcost
    for k = 1:nlp.N-1
        x1,x2 = Z[ix[k]], Z[ix[k+1]]
        u1,u2 = Z[iu[k]], Z[iu[k+1]]
        xm = nlp.xm[k]
        um = nlp.um[k]
        t = nlp.times[k]
        h = nlp.times[k+1] - nlp.times[k]
        
        # TASK: Compute the cost gradient
        grad[ix[k]] .= 0
        grad[iu[k]] .= 0
        
    end
    return nothing
end
