
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
    J = 0.0
    
    # SOLUTION
    ix,iu = nlp.xinds, nlp.uinds
    evaluate_dynamics!(nlp, Z)
    evaluate_midpoints!(nlp, Z)
    J1 = 0.0
    let k = 1
        x1,u1 = Z[ix[k]], Z[iu[k]]
        J1 = stagecost(nlp.obj[k], x1, u1)
    end
    for k = 1:nlp.N-1
        t = nlp.times[k]
        h = nlp.times[k+1] - nlp.times[k]
        x2,u2 = Z[ix[k+1]], Z[iu[k+1]]
        J2 = stagecost(nlp.obj[k+1], x2, u2)
        
        xm = nlp.xm[k]
        um = nlp.um[k]
        Jm = stagecost(nlp.obj[k], xm, um)
        J += h/6 * (J1 + 4Jm + J2)
        
        J1 = J2
    end
    return J
end

"""
    grad_f!(nlp, grad, Z)

Evaluate the gradient of the objective at `Z`, storing the result in `grad`.
"""
function grad_f!(nlp::NLP{n,m}, grad, Z) where {n,m}
    ix,iu = nlp.xinds, nlp.uinds
    obj = nlp.obj
    evaluate_dynamics!(nlp, Z)
    evaluate_midpoints!(nlp, Z)
    evaluate_dynamics_jacobians!(nlp, Z)
    evaluate_midpoint_jacobians!(nlp, Z)

    grad .= 0
    
    for k = 1:nlp.N-1
        x1,x2 = Z[ix[k]], Z[ix[k+1]]
        u1,u2 = Z[iu[k]], Z[iu[k+1]]
        xm = nlp.xm[k]
        um = nlp.um[k]
        t = nlp.times[k]
        h = nlp.times[k+1] - nlp.times[k]
        
        # TASK: Compute the cost gradient
#         grad[ix[k]] .= 0
#         grad[iu[k]] .= 0
        
        # SOLUTION
        dxmx1, dxmx2 = nlp.Am[k,2], nlp.Am[k,3]
        dxmu1, dxmu2 = nlp.Bm[k,2], nlp.Bm[k,3]
        
        dx1 = obj[k].Q*x1 + obj[k].q
        dx2 = obj[k+1].Q*x2 + obj[k+1].q
        du1 = obj[k].R*u1 + obj[k].r
        du2 = obj[k+1].R*u2 + obj[k+1].r
        dxm = obj[k].Q*xm + obj[k].q
        dum = obj[k].R*um + obj[k].r
        
        grad[ix[k]] += h/6 * (dx1 + 4dxmx1'dxm)
        grad[ix[k+1]] += h/6 * (dx2 + 4dxmx2'dxm)
        grad[iu[k]] += h/6 * (du1 + 4dxmu1'dxm + dum/2)
        grad[iu[k+1]] += h/6 * (du2 + 4dxmu2'dxm + dum/2)
    end
    return nothing
end

"""
    hess_f!(nlp, hess, Z)

Evaluate the Hessian of the objective at `Z`, storing the result in `hess`.
"""
function hess_f!(nlp::NLP{n,m}, hess, Z) where {n,m}
    # TASK: Compute the objective hessian
    ix,iu = nlp.xinds, nlp.uinds
    obj = nlp.obj
    hess .= 0
    for k = 1:nlp.N-1
        ix1, ix2 = ix[k], ix[k+1]
        iu1, iu2 = iu[k], iu[k+1]
        h = nlp.times[k+1] - nlp.times[k]
        dxmx1, dxmx2 = nlp.Am[k,2], nlp.Am[k,3]
        dxmu1, dxmu2 = nlp.Bm[k,2], nlp.Bm[k,3]
        
        ddx1 = obj[k].Q
        ddu1 = obj[k].R
        ddx2 = obj[k+1].Q
        ddu2 = obj[k+1].R
        ddxm = obj[k].Q
        ddum = obj[k].R
        
        hess[ix1,ix1] .+= h/6 * (ddx1 + 4dxmx1'ddxm*dxmx1)
        hess[ix1,iu1] .+= h/6 * (4dxmx1'ddxm*dxmu1)
        hess[ix1,ix2] .+= h/6 * (4dxmx1'ddxm*dxmx2)
        hess[ix1,iu2] .+= h/6 * (4dxmx1'ddxm*dxmu2)
        
        hess[iu1,ix1] .+= h/6 * (4dxmu1'ddxm*dxmx1)
        hess[iu1,iu1] .+= h/6 * (4dxmu1'ddxm*dxmu1 + ddum / 4 + ddu1)
        hess[iu1,ix2] .+= h/6 * (4dxmu1'ddxm*dxmx2)
        hess[iu1,iu2] .+= h/6 * (4dxmu1'ddxm*dxmu2 + ddum / 4)
        
        hess[ix2,ix1] .+= h/6 * (4dxmx2'ddxm*dxmx1)
        hess[ix2,iu1] .+= h/6 * (4dxmx2'ddxm*dxmu1)
        hess[ix2,ix2] .+= h/6 * (4dxmx2'ddxm*dxmx2 + ddx2)
        hess[ix2,iu2] .+= h/6 * (4dxmx2'ddxm*dxmu2)
        
        hess[iu2,ix1] .+= h/6 * (4dxmu2'ddxm*dxmx1)
        hess[iu2,iu1] .+= h/6 * (4dxmu2'ddxm*dxmu1 + ddum / 4)
        hess[iu2,ix2] .+= h/6 * (4dxmu2'ddxm*dxmx2)
        hess[iu2,iu2] .+= h/6 * (4dxmu2'ddxm*dxmu2 + ddum / 4 + ddu2)
    end
end