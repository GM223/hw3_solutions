
"""
    evaluate_dynamics!(nlp, Z)

Evaluate the dynamics at the knot points, caching the result in `nlp`.
"""
function evaluate_dynamics!(nlp::NLP, Z)
    ix,iu = nlp.xinds, nlp.uinds
    for k = 1:nlp.N
        t = nlp.times[k]
        x,u = Z[ix[k]], Z[iu[k]]
        nlp.f[k] = dynamics(nlp.model, x, u, t)
    end
end

"""
    evaluate_dynamics_jacobians!(nlp, Z)

Evaluate the dynamics Jacobians at the knot points, caching the result in `nlp`.
"""
function evaluate_dynamics_jacobians!(nlp::NLP, Z)
    ix,iu = nlp.xinds, nlp.uinds
    for k = 1:nlp.N
        t = nlp.times[k]
        x,u = Z[ix[k]], Z[iu[k]]
        nlp.A[k], nlp.B[k] = dynamics_jacobians(nlp.model, x, u, t)
    end
end

"""
    evaluate_midpoints!(nlp, Z)

Evaluate the midpoint of the Hermite Simpson splines. The state midpoint is given 
by 
```math
x_m = \\frac{1}{2} (x_1 + x_2) + \\frac{h}{8}(f(x_1, u_1, t) - f(x_2, u_2, t + h))
```
and the control midpoint is
```math
u_m = \\frac{1}{2} (u_1 + u_2)
```

Also use these to evaluate the dynamics at the midpoint. Cache the results in `nlp`.
"""
function evaluate_midpoints!(nlp::NLP, Z)
    ix,iu = nlp.xinds, nlp.uinds
    for k = 1:nlp.N-1
        # Extract data
        h = nlp.times[k+1] - nlp.times[k]
        t = nlp.times[k]
        x1,x2 = Z[ix[k]], Z[ix[k+1]]
        u1,u2 = Z[iu[k]], Z[iu[k+1]]
        f1 = nlp.f[k]
        f2 = nlp.f[k+1]
        A1,A2 = nlp.A[k], nlp.A[k+1]
        B1,B2 = nlp.B[k], nlp.B[k+1]
        
        # Calculate midpoint
        xm = (x1 + x2) / 2 + h/8 * (f1 - f2)
        um = (u1 + u2) / 2
        
        # Evaluate dynamics at the midpoint
        fm = dynamics(nlp.model, xm, um, t + h/2)
        
        # Cache the results
        nlp.fm[k] = fm
        nlp.xm[k] = xm
        nlp.um[k] = um
    end    
end

"""
    evaluate_midpoint_jacobians!(nlp, Z)

Use the chain rule to evaluate the Jacobians at the midpoint.
Feel free to cache whatever pieces you need to calculate the derivatives of the 
cost and constraints. The caches `nlp.Am` and `nlp.Bm` are matrices, so you can 
store as many temporary matrices you want by adding columns to the cache 
(it currently has 3 columns).
"""
function evaluate_midpoint_jacobians!(nlp::NLP, Z)
    ix,iu = nlp.xinds, nlp.uinds
    for k = 1:nlp.N-1
        # Extract data
        h = nlp.times[k+1] - nlp.times[k]
        t = nlp.times[k]
        x1,x2 = Z[ix[k]], Z[ix[k+1]]
        u1,u2 = Z[iu[k]], Z[iu[k+1]]
        f1 = nlp.f[k]
        f2 = nlp.f[k+1]
        A1,A2 = nlp.A[k], nlp.A[k+1]
        B1,B2 = nlp.B[k], nlp.B[k+1]
        
        # Get midpoints
        xm, um = nlp.xm[k], nlp.um[k]
        
        # Dynamics Jacobians at midpoint
        Am,Bm = dynamics_jacobians(nlp.model, xm, um, t + h/2)
        
        # Derivatives of xm, um,
        dxmx1 = (I/2 + h/8 * A1)    # (n,n)
        dxmu1 = h/8 * B1            # (n,m)
        dxmx2 = (I/2 - h/8 * A2)    # (n,n)
        dxmu2 = -h/8 * B2           # (n,m)
        dumu1 = I/2                 # (m,m)
        dumu2 = I/2                 # (m,m)
        
        # Cache the results
        nlp.Am[k,1], nlp.Am[k,2], nlp.Am[k,3] = Am, dxmx1, dxmx2
        nlp.Bm[k,1], nlp.Bm[k,2], nlp.Bm[k,3] = Bm, dxmu1, dxmu2
    end    
end