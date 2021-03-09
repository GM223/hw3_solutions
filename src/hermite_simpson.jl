function hermite_simpson(model, x1, u1, x2, u2, dt)
    f₁ = dynamics(model, x1,u1) 
    f₂ = dynamics(model, x2,u2)
    umid = 0.5*(u1 + u2)
    xmid = 0.5*(x1 + x2) + dt * (f₁ - f₂) / 8
    fmid = dynamics(model, xmid, umid) 
    return x1 + dt * (f₁ + 4fmid + f₂) / 6 - x2
end

function hs_jacobian(model, x1::StaticVector{n}, u1::StaticVector{m}, x2, u2, dt) where {n,m}
    ix1, iu1 = SVector{n}(1:n), SVector{m}(n .+ (1:m))
    ix2, iu2 = ix1 .+ (n+m), iu1 .+ (n+m)
    faug(zz) = hermite_simpson(model, zz[ix1], zz[iu1], zz[ix2], zz[iu2], dt)
    zz = [x1; u1; x2; u2]
    ForwardDiff.jacobian(faug, zz)
end

function hs_jvp(model, x1::StaticVector{n}, u1::StaticVector{m}, x2, u2, λ, dt) where {n,m}
    ix1, iu1 = SVector{n}(1:n), SVector{m}(n .+ (1:m))
    ix2, iu2 = ix1 .+ (n+m), iu1 .+ (n+m)
    cvp(zz) = dot(λ, hermite_simpson(model, zz[ix1], zz[iu1], zz[ix2], zz[iu2], dt))
    zz = [x1; u1; x2; u2]
    ForwardDiff.gradient(cvp, zz)
end

function hs_∇jvp(model, x1::StaticVector{n}, u1::StaticVector{m}, x2, u2, λ, dt) where {n,m}
    ix1, iu1 = SVector{n}(1:n), SVector{m}(n .+ (1:m))
    ix2, iu2 = ix1 .+ (n+m), iu1 .+ (n+m)
    cvp(zz) = dot(λ, hermite_simpson(model, zz[ix1], zz[iu1], zz[ix2], zz[iu2], dt))
    zz = [x1; u1; x2; u2]
    ForwardDiff.hessian(cvp, zz)
end
