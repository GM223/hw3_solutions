using MeshCat, GeometryBasics, Colors, CoordinateTransformations, Rotations
using RobotDynamics: state_dim, control_dim


###############################################
# Visualization
###############################################
function defcolor(c1, c2, c1def, c2def)
    if !isnothing(c1) && isnothing(c2)
        c2 = c1
    else
        c1 = isnothing(c1) ? c1def : c1
        c2 = isnothing(c2) ? c2def : c2
    end
    c1,c2
end

function set_mesh!(vis0, model::RobotZoo.Cartpole; 
        color=nothing, color2=nothing)
    vis = vis0["robot"]
    dim = Vec(0.1, 0.3, 0.1)
    rod = Cylinder(Point3f0(0,-10,0), Point3f0(0,10,0), 0.01f0)
    cart = Rect3D(-dim/2, dim)
    hinge = Cylinder(Point3f0(-dim[1]/2,0,dim[3]/2), Point3f0(dim[1],0,dim[3]/2), 0.03f0)
    c1,c2 = defcolor(color,color2, colorant"blue", colorant"red")

    pole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l),0.01f0)
    mass = HyperSphere(Point3f0(0,0,model.l), 0.05f0)
    setobject!(vis["rod"], rod, MeshPhongMaterial(color=colorant"grey"))
    setobject!(vis["cart","box"],   cart, MeshPhongMaterial(color=isnothing(color) ? colorant"green" : color))
    setobject!(vis["cart","hinge"], hinge, MeshPhongMaterial(color=colorant"black"))
    setobject!(vis["cart","pole","geom","cyl"], pole, MeshPhongMaterial(color=c1))
    setobject!(vis["cart","pole","geom","mass"], mass, MeshPhongMaterial(color=c2))
    settransform!(vis["cart","pole"], Translation(0.75*dim[1],0,dim[3]/2))
end

function visualize!(vis, model::RobotZoo.Cartpole, x::StaticVector)
    y = x[1]
    θ = x[2]
    q = expm((pi-θ) * @SVector [1,0,0])
    settransform!(vis["robot","cart"], Translation(0,-y,0))
    settransform!(vis["robot","cart","pole","geom"], LinearMap(UnitQuaternion(q)))
end

function visualize!(vis, model::RobotDynamics.AbstractModel, tf::Real, X)
    fps = Int(round((length(X)-1)/tf))
    anim = MeshCat.Animation(fps)
    n = state_dim(model)
    for (k,x) in enumerate(X)
        atframe(anim, k) do
            x = X[k]
            visualize!(vis, model, SVector{n}(x)) 
        end
    end
    setanimation!(vis, anim)
end

###############################################
# Dynamics
###############################################
mutable struct DynamicsEvals 
    dynamics::Int
    jacobian::Int
end
const DYNAMICS_EVALS = DynamicsEvals(0,0)

macro dynamicsevals(expr)
    quote
        _evals = DYNAMICS_EVALS.dynamics 
        $(esc(expr))
        DYNAMICS_EVALS.dynamics - _evals
    end
end

macro jacobianevals(expr)
    quote
        _evals = DYNAMICS_EVALS.jacobian
        $(esc(expr))
        DYNAMICS_EVALS.jacobian - _evals
    end
end

function dynamics(model::RobotZoo.Cartpole, x, u, t)
    DYNAMICS_EVALS.dynamics += 1
    RobotDynamics.dynamics(model, x, u)
end

const CARTPOLE_JACOBIAN_CACHE = zeros(4, 5)
function dynamics_jacobians(model::RobotZoo.Cartpole, x, u, t)
    DYNAMICS_EVALS.jacobian += 1
    z = RobotDynamics.StaticKnotPoint(x, u, NaN, t)
    RobotDynamics.jacobian!(CARTPOLE_JACOBIAN_CACHE, model, z)
    ix = SA[1,2,3,4]
    iu = SA[5]
    A = CARTPOLE_JACOBIAN_CACHE[ix,ix]
    B = CARTPOLE_JACOBIAN_CACHE[ix,iu]
    return A,B
end

function discrete_dynamics(model::RobotDynamics.AbstractModel, x, u, t, dt)
    z = RobotDynamics.StaticKnotPoint(x, u, dt, t)
    RobotDynamics.discrete_dynamics(RobotDynamics.RK4, model, z)
end

function discrete_jacobian(model::RobotDynamics.AbstractModel, x, u, t, dt)
    z = RobotDynamics.StaticKnotPoint(x, u, dt, t)
    RobotDynamics.discrete_jacobian!(RobotDynamics.RK4, CARTPOLE_JACOBIAN_CACHE, model, z)
    ix = SA[1,2,3,4]
    iu = SA[5]
    A = CARTPOLE_JACOBIAN_CACHE[ix,ix]
    B = CARTPOLE_JACOBIAN_CACHE[ix,iu]
    return A,B
end

function simulate(model::RobotDynamics.AbstractModel, x0, ctrl; tf=2.0, dt=0.025, w=0.1)
    n,m = size(model)
    times = range(0, tf, step=dt)
    N = length(times)
    X = [@SVector zeros(n) for k = 1:N] 
    U = [@SVector zeros(m) for k = 1:N-1]
    X[1] = x0

    tstart = time_ns()

    for k = 1:N-1
        U[k] = get_control(ctrl, X[k], times[k]) + w*@SVector randn(m)
        # X[k+1] = discrete_dynamics(RK4, model, X[k], U[k], times[k], dt)
        X[k+1] = discrete_dynamics(model, X[k], U[k], times[k], dt)
    end
    tend = time_ns()
    rate = N / (tend - tstart) * 1e9
    println("Controller ran at $rate Hz")
    return X,U,times
end

###############################################
# Null Controller
###############################################

struct NullController{m} 
    NullController(m::Integer) = new{Int(m)}()
end
NullController(model::RobotDynamics.AbstractModel) = NullController(control_dim(model))
get_control(ctrl::NullController{m}, x, t) where m = @SVector zeros(m)

# function run_tests()
#     include(joinpath(@__DIR__,"..","test","q2.jl"))
# end