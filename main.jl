include("./BenoitModel.jl")
include("./ByrneModel.jl")
include("./RafalModel.jl")

using CSV, DataFrames, FFTW, NeuralDynamics, Plots, Statistics
using .RafalModel: create_rafal_model, simulate_rafal_model
using .BenoitModel: create_benoit_model, simulate_benoit_model
using .ByrneModel: create_byrne_pop, simulate_byrne_pop

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

function plot_oscill_time(df, sampling_rate)
    freqs = fftshift(fftfreq(length(df.t), sampling_rate))
    F_E = fftshift(fft(df.rE .- mean(df.rE)))
    F_I = fftshift(fft(df.rI .- mean(df.rI)))

    p1 = plot(df.t, [df.theta_E, df.theta_I], xlabel="t", ylabel="Input")
    p2 = plot(df.t, [df.rE, df.rI], xlabel="t", ylabel="Activity")
    p3 = plot(freqs, [abs.(F_E), abs.(F_I)], xlabel="f", xlim=(0, +100), xticks=0:20:100) 
    plot(p1, p2, p3, layout=(3,1))
    savefig("plots/myplot.png")
end

function plot_max_min(df)
    plot(df.theta, [df.rE_max, df.rE_min], label=["max" "min"], xlabel="theta_I"*df.input_pop[1], ylabel="E amplitude")
    savefig("plots/myplot.png")
end

function plot_byrne_single(df)
    p1 = plot(df.t, df.rR, xlabel="t", ylabel="R")
    p2 = plot(df.t, df.rV, xlabel="t", ylabel="V")
    p3 = plot(df.t, df.rZ, xlabel="t", ylabel="|Z|")
    plot(p1, p2, p3, layout=(3,1))
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
    R = simulate(m, range_t, dt, theta_E, theta_I)

    rE = R[1].rE
    rI = R[1].rI

    return DataFrame(t=range_t, rE=rE, rI=rI, theta_E=theta_E, theta_I=theta_I)
end

function run_byrne_single(p, simulate, range_t, dt)
    rR, rV, rZ = simulate(p, range_t, dt)
    return DataFrame(t=range_t, rR=rR, rV=rV, rZ=rZ)
end

function main_raf()
    # Parameters (time in s)
    N=1
    W=Float32[0.0]
    tau_E = Float32(0.0032)
    tau_I = Float32(0.0032)
    w_EE = Float32(2.4)
    w_EI = Float32(2.0)
    w_IE = Float32(2.0)
    beta = Float32(4.0)

    model = create_rafal_model(N, W, tau_E, tau_I, w_EE, w_EI, w_IE, beta)
    
    T = 1.0
    dt = 0.001
    range_t = 0.0:dt:T
    sampling_rate = T / dt

    E_A = 0.4
    E_f = 4
    E_base = 0.4
    E_phase = 0.0
    I_A = 0.0
    I_f = 4
    I_base = 0.0
    I_phase = -(pi / 3)
    
    df = run_act_oscill_time(model, simulate_rafal_model, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)
    plot_oscill_time(df, sampling_rate)

end

function main_byrne()
    # Parameters (time in ms)
    ex = Float32(2.0)
    ks = Float32(1.0)
    kv = Float32(1.0)
    gamma = Float32(0.5)
    tau = Float32(16.0)
    alpha = Float32(0.5)

    p = create_byrne_pop(ex, ks, kv, gamma, tau, alpha)
    
    T = 1000.0
    dt = 0.001
    range_t = 0.0:dt:T
    
    df = run_byrne_single(p, simulate_byrne_pop, range_t, dt)
    plot_byrne_single(df)

end

main_raf()

