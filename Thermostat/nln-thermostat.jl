import Pkg
Pkg.activate("."); 
Pkg.instantiate();

using ReactiveMP, GraphPPL, Rocket, Plots
import ReactiveMP: getinterface, messageout

struct NewControlPosteriorFromModel
    posterior
end

struct NewObservationFromEnvironment
    observation
end

mutable struct Environment <: Actor{Any}
    
    # Attributes:
    real_position :: Float64
    real_temperature :: Float64
    temperature_at_origin :: Float64
    
    events # :: Subject{Any} <- event emitter
end


# States
zₜ = NormalMeanVariance(0.0, 1.0)

# Time horizon
T = 2

# Transition matrix
A = 1.0

# Control matrix
B = 1.0

# Observation matrix
C = -1.0

# Process noise covariance matrix
Q = 1e-2

# Measurement noise covariance matrix
R = 1e-2

# Goal prior
goals = [ 100.0 for t in 1:T ]  # note from bvdmitri: the goal is to find heat source

# Control prior
θᵤ = NormalMeanVariance(0, 1e-2)

# Nonlinear observation function
g(state) = 100.0/(state^2 + 1)
# g_inv(temp) = (1/temp - 1)^0.5 


@model function controller_reactive(T, A, B, Q, R, C, θᵤ; T0=100)

    cA = constvar(A) # Transition matrix
    cQ = constvar(Q) # Transition noise
    cB = constvar(B) # Observation matrix
    cR = constvar(R) # Observation noise
    cC = constvar(C) # Control matrix

    # Vector of reference to factor nodes of latent states. Useful for later
    znodes = Vector{FactorNode}(undef, T)
    
    z = randomvar(T) # Hidden states
    u = randomvar(T) # Control states
    r = randomvar(T) # Temp states
    x = datavar(Float64, T) # Goal observations

    # current state
    z_current_mean = datavar(Float64)  # Prior state
    z_current_var  = datavar(Float64)

    z_prior ~ NormalMeanVariance(z_current_mean, z_current_var)
    z_prev = z_prior
    
    # Current observation for filtering
    x_0 = datavar(Float64)
    # x_0 ~ MvNormalMeanCovariance(cB * z_prior, cR)
    r_0 ~ DeltaNode(z_prev) where { meta = DeltaNodeMeta(g, nothing, Unscented()) }
    x_0 ~ NormalMeanPrecision(r_0, 0.01)

    # Extend the model T steps into the future
    for t in 1:T

        # Future control
        u[t] ~ NormalMeanVariance(mean(θᵤ), var(θᵤ))

        # Future states
        znodes[t], z[t] ~ NormalMeanVariance(cA*z_prev + cC*u[t], cQ)
    
        # Extend the model T steps into the future
        # x[t] ~ MvNormalMeanCovariance(cB * z[t], cR)
        r[t] ~ DeltaNode(z[t]) where { meta = DeltaNodeMeta(g, nothing, Unscented()) }
        x[t] ~ NormalMeanPrecision(r[t], 0.01)

        # Update previous state variable
        z_prev = z[t]
    end

    return x, x_0, z, znodes, u, z_current_mean, z_current_var
end


function Environment(; T0=100) # T0 is temperature at origin

    # Initial position
    real_position = 3.0 # Initial position is '4'
    real_temperature = T0 / (real_position^2 + 1)

    # BehaviorSubject allows us to set the initial state
    events = BehaviorSubject(NewObservationFromEnvironment(real_temperature + sqrt(0.01)*randn()))
    
    return Environment(real_position, real_temperature, T0, events)
end

function Rocket.on_next!(environment::Environment, event::NewControlPosteriorFromModel)
    # println("Event to our env: ", mean(first(event.posterior)))
    
    # Update position with the mean of the first planned action.
    next_real_position = environment.real_position + mean(first(event.posterior))
    
    # Set the position variable in the environment 
    environment.real_position = next_real_position
    
    # Update current temperature
    environment.real_temperature = environment.temperature_at_origin / (environment.real_position^2 + 1.0)

    # Update environment and emit a new observation
    temperature_observation = environment.real_temperature + sqrt(0.01)*randn()
    println("---------")
    println("Action = " * string(mean(first(event.posterior))))
    println("Temperature = " * string(temperature_observation))
    println("Position = " * string(next_real_position))
    println("---------")

    next!(environment.events, NewObservationFromEnvironment(temperature_observation))
end

struct PlanningModel <: Actor{Any}
    model
    model_variables
    prediction_subscription
    control_subscription
    events :: Subject{Any}
end

function PlanningModel()
    model, (x, x_0, z, znodes, u, zcm, zcc) = controller_reactive(T, A, B, Q, R, C, θᵤ);
    
    # Set initial state priors
    update!(zcm, 3.0)
    update!(zcc, 1.0)
    
    # Subject allows the agent to emit actions that can be picked up by the environment
    events = Subject(Any)
    
    # Update the prior by performing a single filtering step. note that we get the _message_
    # instead of the _marginal_ here to prevent erroneous information from the future
    prediction_subscription = subscribe!(
        messageout(getinterface(znodes[1], :out)), (prediction_z_1) -> begin # TODO HERE ;)
        m, c = mean_cov(as_message(prediction_z_1))
        update!(zcm, m)
        update!(zcc, c)
    end)
    
    # Emit an action whenever the control marginals are updated
    control_subscription = subscribe!(getmarginals(u), (posterior_u) -> next!(events, NewControlPosteriorFromModel(posterior_u)))
    
    return PlanningModel(model, (x, x_0, z, znodes, u, zcm, zcc), prediction_subscription, control_subscription, events)
end

function Rocket.on_next!(model::PlanningModel, event::NewObservationFromEnvironment)
    # println("Event to our model: ", event)
    (x, x_0, z, znodes, u, zcm, zcc) = model.model_variables
    update!(x_0, event.observation)
end

# function Rocket.on_next!(model::PlanningModel, event::FatalErrorEvent)
#     unsubscribe!(model.control_subscription)
# end

environment = Environment();
model = PlanningModel();

debug_events = []

env_to_model_subscription = subscribe!(environment.events |> tap((e) -> push!(debug_events, e)), model)
model_to_env_subscription = subscribe!(model.events, environment)
;

t = from(1:100) # Update once every second

# Set goal priors every time the timer ticks. This fills in the remaining datavars and triggers a round
# of message passing
timer_subscription = subscribe!(t, (_) -> begin 
    let 
        (x, x_0, z, znodes, u, zcm, zcc) = model.model_variables
        update!(x, goals)
        # println("Static timer")
    end
end)

;

# unsubscribe!(timer_subscription)

# goals_timer = timer(0, 500)
# goals_subscription = subscribe!(goals_timer, (index) -> begin
#     x = sin(index / π)
#     y = cos(index / π)
#     new_goal = [ x, y ]
#     # println("We have a new goal: ", new_goal,"\n")
#     for i in 1:T
#         copyto!(goals[i], new_goal)
#     end
# end)

# function emit_signal(env::Environment)
#     # y ∼ N(T(zₜ), θ)
#     #y = [env.z[2] / (env.z[1]^2 + 1.0) + sqrt(env.z[3])*randn()] # Report noisy temperature at current position
#     # Report noisy temperature at current position
#     y = env.real_temperature / (env.real_position^2 + 1.0) + sqrt(0.01)*randn()
#     #y = [y]
#     return y
# end
