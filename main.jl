include("./BenoitModel.jl")
include("./RafalModel.jl")

using CSV, DataFrames, NeuralDynamics, Plots
using .RafalModel: create_rafal_model, simulate_rafal_model
using .BenoitModel: create_benoit_model, simulate_benoit_model

function create_oscill_input(A, f, base, phase, range_t)
    Lt = length(range_t)
    R = zeros(length(range_t))
    for i in 1:Lt
        R[i] = A * sin(f * 2 * pi * range_t[i] + phase) + base
    end
    return R
end

function plot_act_time(df)
    plot(df.t, [df.rE, df.rI], label=["E" "I"], xlabel="t", ylabel="Activity")
    savefig("plots/myplot.png")
end

function plot_oscill_time(df)
    p1 = plot(df.t, [df.theta_E, df.theta_I], xlabel="t", ylabel="Input")
    p2 = plot(df.t, [df.rE, df.rI], xlabel="t", ylabel="Activity")
    plot(p1, p2, layout=(2,1))
    savefig("plots/myplot.png")
end

function plot_max_min(df)
    plot(df.theta, [df.rE_max, df.rE_min], label=["max" "min"], xlabel="theta_I"*df.input_pop[1], ylabel="E amplitude")
    savefig("plots/myplot.png")
end


function run_max_min(m, simulate, range_t, dt, range_theta_input, theta_const, input_pop)
    Lt = length(range_t)
    Lte = length(range_theta_input)
    rE_max = zeros(Lte)
    rE_min = zeros(Lte)

    window = [0.05, 0.1]

    for i in 1:Lte
        theta_E = input_pop == "E" ? fill(range_theta_input[i], Lt) : fill(theta_const, Lt)
        theta_I = input_pop == "I" ? fill(range_theta_input[i], Lt) : fill(theta_const, Lt)

        rE, _ = simulate(m, range_t, dt, theta_E, theta_I)
        rE_max[i], _ = findmax(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
        rE_min[i], _ = findmin(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
    end
   return DataFrame(theta=range_theta_input, rE_max=rE_max, rE_min=rE_min, input_pop=input_pop)
end

function run_act_time(m, simulate, range_t, dt, theta_E, theta_I)
    theta_E_t = fill(theta_E, length(range_t))
    theta_I_t = fill(theta_I, length(range_t))

    rE, rI = simulate(m, range_t, dt, theta_E_t, theta_I_t)
    return DataFrame(t=range_t, rE=rE, rI=rI)
end

function run_act_oscill_time(m, simulate, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)
    theta_E = create_oscill_input(E_A, E_f, E_base, E_phase, range_t)
    theta_I = create_oscill_input(I_A, I_f, I_base, I_phase, range_t)
    rE, rI = simulate(m, range_t, dt, theta_E, theta_I)

    return DataFrame(t=range_t, rE=rE, rI=rI, theta_E=theta_E, theta_I=theta_I)
end

function main()
    # Parameters (time in s)
    tau_E = Float32(0.0032)
    tau_I = Float32(0.0032)
    w_EE = Float32(2.4)
    w_EI = Float32(2.0)
    w_IE = Float32(2.0)
    beta = Float32(4.0)

    model = create_benoit_model(tau_E, tau_I, w_EE, w_EI, w_IE, beta)
    
    T = 1.0
    dt = 0.001
    range_t = 0.0:dt:T

    E_A = 0.45
    E_f = 4
    E_base = 0.45
    E_phase = 0.0
    I_A = 0.5
    I_f = 4
    I_base = 0.5
    I_phase = -(pi / 3)
    
    df = run_act_oscill_time(model, simulate_benoit_model, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)
    plot_oscill_time(df)

end

main()

