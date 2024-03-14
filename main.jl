include("./BenoitModel.jl")
include("./ByrneModel.jl")
include("./RafalModel.jl")
include("./Stimulation.jl")

using ControlSystems, CSV, DataFrames, DSP, FFTW, KernelDensity, LsqFit, NeuralDynamics, Plots, Statistics, StatsBase
using .RafalModel: create_rafal_model, simulate_rafal_model
using .BenoitModel: create_benoit_model, simulate_benoit_model
using .ByrneModel: create_byrne_pop, create_byrne_pop_EI, create_byrne_network, create_if_pop, simulate_if_pop, simulate_byrne_EI_network
using .Stimulation: create_stimulus, create_stim_response, yousif_transfer

function create_oscill_input(A, f, base, phase, range_t)
    Lt = length(range_t)
    R = zeros(length(range_t))
    for i in 1:Lt
        R[i] = A * sin(f * 2 * pi * range_t[i] + phase) + base
    end
    return R
end

function plot_act_time(df, N)
    plots = []
    for i in 1:N
        push!(plots, plot(df.T[i], [df.theta_E[i], df.theta_I[i]], label=["E" "I"], xlabel="t", ylabel="Input"))
        push!(plots, plot(df.T[i], [df.R[i].rE, df.R[i].rI], label=["E" "I"], xlabel="t", ylabel="Activity"))
    end
    plot(plots..., layout=(2*N, 1), size=(500, 400*N))
    savefig("plots/act.png")
end

function plot_spec(df, N, sampling_rate)
    plots = []
    for i in 1:N
        freqs = fftshift(fftfreq(length(df.T[i]), sampling_rate))
        F_E = fftshift(fft(df.R[i].rE .- mean(df.R[i].rE)))
        F_I = fftshift(fft(df.R[i].rI .- mean(df.R[i].rI)))
        push!(plots, plot(freqs, [abs.(F_E), abs.(F_I)], xlabel="f", xlim=(0, +10), xticks=0:2:10) )
    end
    plot(plots..., layout=(1, 2*N), size=(600*N, 300))
    savefig("plots/spec.png")
end

function plot_oscill_time(df, sampling_rate, spec=false)
    freqs = fftshift(fftfreq(length(df.t), sampling_rate))
    F_E = fftshift(fft(df.rE .- mean(df.rE)))
    F_I = fftshift(fft(df.rI .- mean(df.rI)))

    p1 = plot(df.t, [df.theta_E, df.theta_I], xlabel="t", ylabel="Input")
    p2 = plot(df.t, [df.rE, df.rI], xlabel="t", ylabel="Activity")

    if spec
        p3 = plot(freqs, [abs.(F_E), abs.(F_I)], xlabel="f", xlim=(0, +100), xticks=0:2:10) 
        plot(p1, p2, p3, layout=(3,1))
    else
        plot(p1, p2, layout=(2,1))
    end

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

function plot_avg_if_activity(df)
    plot(df.t, df.rVu, xlabel="t", ylabel="V")
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

function run_act_time(m, simulate, range_t, dt, theta_E, theta_I, stim_response)
    theta_E_t = [fill(i, length(range_t)) for i in theta_E]
    theta_I_t = [fill(i, length(range_t)) for i in theta_I]

    for i in 1:length(stim_response)
        theta_E_t[1][i] = theta_E_t[1][i] .+ stim_response[i]
    end

    R = simulate(m, range_t, dt, theta_E_t, theta_I_t)
    lR = length(R)
    for i in 1:lR
        urE = mean(R[i].rE)
        srE = std(R[i].rE)
        urI = mean(R[i].rI)
        srI = std(R[i].rI)

        for j in 1:length(R[i].rE)
            R[i].rE[j] = (R[i].rE[j] - urE) / srE
            R[i].rI[j] = (R[i].rI[j] - urI) / srI
        end
    end

    T = [range_t for i in 1:length(theta_E)]
    return DataFrame(T=T, R=R, theta_E=theta_E_t, theta_I=theta_I_t)
end

function run_act_oscill_time(m, simulate, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)
    theta_E_t = [create_oscill_input(E_A, E_f, E_base, E_phase, range_t) for i in 1:2]
    theta_I_t = [create_oscill_input(I_A, I_f, I_base, I_phase, range_t) for i in 1:2]

    R = simulate(m, range_t, dt, theta_E_t, theta_I_t)
    T = [range_t for i in 1:2]
    return DataFrame(T=T, R=R, theta_E=theta_E_t, theta_I=theta_I_t)
end

function run_byrne_single(p, simulate, range_t, dt)
    rR, rV, rZ = simulate(p, range_t, dt)
    return DataFrame(t=range_t, rR=rR, rV=rV, rZ=rZ)
end

function run_byrne_net(N, simulate, range_t, dt)
    R = simulate(N, range_t, dt)
    return DataFrame(t=range_t, rR=R[1].rR_E, rV=R[1].rV_E, rZ=R[1].rZ_E)
end

function run_byrne_if(p, simulate, range_t, dt)
    _, rVu = simulate(p, range_t, dt)
    return DataFrame(t=range_t, rVu=rVu)
end

function hilbert_amplitude_pdf(signal::Array{Float32, 1}; bandwidth=0.1)
    hilbert_transform = hilbert(signal)
    hilbert_amp = abs.(hilbert_transform)
    
    # Estimate PDF using kernel density estimation
    U = kde(hilbert_amp, bandwidth=bandwidth)
    
    return U.x, U.density, hilbert_amp
end

function plot_hilbert_amplitude_pdf(signal::Array{Float32, 1},T, sampling_rate, bandwidth=0.1)
    x, y, ha = hilbert_amplitude_pdf(signal, bandwidth=bandwidth)
    plot(x, y, xlabel="Amplitude", ylabel="Density")
    savefig("plots/hilbert_amp_pdf.png")
    plot(T, ha, xlabel="Amplitude", ylabel="Density")
    savefig("plots/hilbert_amp.png")

    freqs = fftshift(fftfreq(length(T), sampling_rate))
    F_A = fftshift(fft(ha))
    plot(freqs, abs.(F_A), xlabel="f", xlim=(0.1, +10), xticks=0.1:2:10)
    savefig("plots/hilbert_psd.png")

end


function main_raf()
    # Parameters (time in s)
    #N=2
    #W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    #etta=Float32(1.0)
    #tau_E = Float32(0.0176)
    #tau_I = Float32(0.0176)
    #w_EE = Float32(2.4)
    #w_EI = Float32(2.0)
    #w_IE = Float32(2.0)
    #beta = Float32(4.0)

    N=1
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(1.0)
    tau_E = Float32(0.0758)
    tau_I = Float32(0.0758)
    w_EE = Float32(6.7541)
    w_EI = Float32(9.6306)
    w_IE = Float32(9.4014)
    beta = Float32(1.1853)

    model = create_benoit_model(N, W, etta, tau_E, tau_I, w_EE, w_EI, w_IE, beta)
    
    T = 100.0
    dt = 0.001
    range_t = 0.0:dt:T
    sampling_rate = 1.0 / dt

    #E_A = 0.1
    #E_f = 4
    #E_base = 0.6
    #E_phase = 0.0
    #I_A = 0.0
    #I_f = 4
    #I_base = 0.0
    #I_phase = -(pi / 3)
    #df = run_act_oscill_time(model, simulate_benoit_model, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)

    A=2*100*1e-3
    f=4

    #stim=create_stimulus(A, f, range_t)
    #response=create_stim_response(stim, range_t)
    response = fill(0.0, length(range_t)) #yousif_transfer(A, f, range_t)
    for i in 1:6:T-6
        #Start pulse
        for j in 0:24
            for k in 0:2:10
                response[Int64(trunc(i*1000+j*200+k*(1000/130)))] = 0.1684
            end
        end
    end
    theta_E = [1.4240]
    theta_I = [-3.2345]
    stim = response
    df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

    #filter = digitalfilter(Bandpass(3.0,7.0),Butterworth(2))
    #df.R[1].rE = filtfilt(filter, df.R[1].rE)

    plot_act_time(df, N)
    plot_spec(df, N, sampling_rate)
    plot_hilbert_amplitude_pdf(df.R[1].rE, df.T[1], sampling_rate)
end

function main_byrne()
    # Parameters (time in ms)
    N=1
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(1.0)
    ex = Float32(2.0)
    ks = Float32(0.5)
    kv = Float32(0.5)
    gamma = Float32(0.5)
    tau = Float32(16.0)
    alpha = Float32(0.5)

    vth = 1.000
    vr = -1.000

    #p = create_byrne_pop(ex, ks, kv, gamma, tau, alpha)
    #p = create_if_pop(1000, ex, ks, kv, gamma, tau, alpha, vth, vr)
    E = create_byrne_pop_EI(ex, gamma, tau)
    I = create_byrne_pop_EI(ex, gamma, tau)
    N = create_byrne_network(N, W, etta, E, I, ks, kv, alpha)
    
    T = 15.0
    dt = 0.001
    range_t = 0.0:dt:T
    
    #df = run_byrne_single(p, simulate_byrne_pop, range_t, dt)
    #df = run_byrne_if(p, simulate_if_pop, range_t, dt)
    df= run_byrne_net(N, simulate_byrne_EI_network, range_t, dt)

    plot_byrne_single(df)
    #plot_avg_if_activity(df)

end

function main_stim()
    A=100*1e-6
    f=10
    T = 1.0
    dt = 0.001
    range_t = 0.0:dt:T

    stim=create_stimulus(A, f, range_t)
    response=create_stim_response(stim, range_t)
    plot(range_t, response, xlabel="t", ylabel="Activity")
    savefig("plot2.png")
end

main_raf()

