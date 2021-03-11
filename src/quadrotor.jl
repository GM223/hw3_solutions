using MeshCat
using RobotZoo: Quadrotor, PlanarQuadrotor
using CoordinateTransformations, Rotations, Colors, StaticArrays, RobotDynamics

function set_mesh!(vis, model::L;
        scaling=1.0, color=colorant"black"
    ) where {L <: Union{Quadrotor, PlanarQuadrotor}} 
    # urdf_folder = joinpath(@__DIR__, "..", "data", "meshes")
    urdf_folder = @__DIR__
    # if scaling != 1.0
    #     quad_scaling = 0.085 * scaling
    obj = joinpath(urdf_folder, "quadrotor_scaled.obj")
    if scaling != 1.0
        error("Scaling not implemented after switching to MeshCat 0.12")
    end
    robot_obj = MeshFileGeometry(obj)
    mat = MeshPhongMaterial(color=color)
    setobject!(vis["robot"]["geom"], robot_obj, mat)
    if hasfield(L, :ned)
        model.ned && settransform!(vis["robot"]["geom"], LinearMap(RotX(pi)))
    end
end

function visualize!(vis, model::PlanarQuadrotor, x::StaticVector)
    py,pz = x[1], x[2]
    θ = x[3]
    settransform!(vis["robot"], compose(Translation(0,py,pz), LinearMap(RotX(-θ))))
end

function visualize!(vis, model, tf::Real, X)
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

function line_reference()
    x1ref = Array(LinRange(-2,2,101))
    x2ref = Array(ones(101))
    θref = Array(zeros(101))
    v1ref = Array((4.0/3.0)*ones(101))
    v2ref = Array(zeros(101))
    ωref = Array(zeros(101))
    xref = [x1ref'; x2ref'; θref'; v1ref'; v2ref'; ωref'] 
    return [SVector{6}(x) for x in eachcol(xref)]
end

function flip_reference()
    x1ref = [LinRange(-3,0,20); zeros(20); LinRange(0,3,21)]
    x2ref = [ones(20); LinRange(1,3,10); LinRange(3,1,10); ones(21)]
    θref = [zeros(20); LinRange(0,-2*pi,20); -2*pi*ones(21)]
    v1ref = [6.0*ones(20); zeros(20); 6.0*ones(21)]
    v2ref = [zeros(20); 8.0*ones(10); -8.0*ones(10); zeros(21)]
    ωref = [zeros(20); -4*pi*ones(20); zeros(21)]
    xref = [x1ref'; x2ref'; θref'; v1ref'; v2ref'; ωref']
    return [SVector{6}(x) for x in eachcol(xref)]
end

function RobotDynamics.discrete_jacobian!(::Type{Q}, ∇f, model::AbstractModel,
        x, u, t, dt) where {Q<:RobotDynamics.Explicit}
    z = KnotPoint(x, u, dt, t)
    RobotDynamics.discrete_jacobian!(Q, ∇f, model, z)
end

struct WindyQuad <: AbstractModel
    quad::PlanarQuadrotor
    dir::MVector{2,Float64}   # wind direction
    wd::Float64               # std on wind angle
    wm::Float64               # std on wind magnitude
end
function WindyQuad(quad::PlanarQuadrotor;
        wind = [1,1]*1.0,
        wd = deg2rad(10),
        wm = 0.01,
    )
    WindyQuad(quad, SA_F64[wind[1], wind[2]], Float64(wd), Float64(wm)) 
end
RobotDynamics.state_dim(model::WindyQuad) = state_dim(model.quad)
RobotDynamics.control_dim(model::WindyQuad) = control_dim(model.quad)
function RobotDynamics.dynamics(model::WindyQuad, x, u)
    ẋ = dynamics(model.quad, x, u)
    mass = model.quad.mass
    wind_mag = randn()*model.wm
    wind_dir = Angle2d(randn()*model.wd) * model.dir
    Fwind =  Angle2d(randn()*model.wd) * model.dir 
    ẋ2 = SA[ẋ[1], ẋ[2], ẋ[3], ẋ[4] + Fwind[1]/mass, ẋ[5] + Fwind[2]/mass, ẋ[6]]
    return ẋ2
end

function simulate(quad::PlanarQuadrotor, x0, ctrl; tf=1.5, dt=0.025, kwargs...)
    model = WindyQuad(quad; kwargs...)

    n,m = size(model)
    times = range(0, tf, step=dt)
    N = length(times)
    X = [@SVector zeros(n) for k = 1:N] 
    U = [@SVector zeros(m) for k = 1:N-1]
    X[1] = x0

    tstart = time_ns()

    for k = 1:N-1
        U[k] = get_control(ctrl, X[k], times[k])
        X[k+1] = discrete_dynamics(RK4, model, X[k], U[k], times[k], dt)
    end
    tend = time_ns()
    rate = N / (tend - tstart) * 1e9
    println("Controller ran at $rate Hz")
    return X,U,times
end

function run_tests()
    include(joinpath(@__DIR__,"..","test","q1.jl"))
end