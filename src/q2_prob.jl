Base.@kwdef struct CartpoleProblem{T}
    model::RobotZoo.Cartpole{T} = RobotZoo.Cartpole()
    n::Int = 4                               # num states
    m::Int = 1                               # num controls
    N::Int = 101                             # horizon
    tf::T = 2.0                              # final time (sec)
    x0::SVector{4,T} = @SVector zeros(4)     # initial state
    xf::SVector{4,T} = SA[0,pi,0,0.]         # goal state
    Q::Diagonal{T, SVector{4,T}} = Diagonal(@SVector fill(1e-2, 4))
    R::Diagonal{T, SVector{1,T}} = Diagonal(@SVector fill(5e-0, 1))
    Qf::Diagonal{T, SVector{4,T}} = Diagonal(@SVector fill(1e1, 4))
end

# function get_objective(prob::CartpoleProblem)
#     costfun = LQRCost(prob.Q, prob.R, prob.xf)
#     costterm = LQRCost(prob.Qf, prob.R, prob.xf)
#     obj = push!(fill(costfun, prob.N-1), costterm)
#     return obj
# end

"""
    get_initial_trajectory(prob)

Return the initial state and control trajectories for the cartpole. 
The state trajectory rotates the pendulum from 0 to pi with constant 
velocity, and the cart remains still.
"""
function get_initial_trajectory(prob::CartpoleProblem)
    tf = prob.tf
    x0 = prob.x0
    xf = prob.xf
    times = range(0,tf,length=prob.N)
    X = map(times) do t
        alpha = t / tf
        SA[
            sin(alpha * 2pi)*0,
            -pi * alpha,
            cos(alpha * 2pi)*0,
            -alpha*0
        ]
    end
    X[end] = xf
    # X = [x0 + (xf - x0)*t for t in range(0,1, length=T)]
    U = [@SVector zeros(prob.m) for k = 1:prob.N];
    return X, U
end