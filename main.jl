include("./analysis.jl")

include("./BenoitModel.jl")
include("./ByrneModel.jl")
include("./RafalModel.jl")
include("./Stimulation.jl")
include("./Signal.jl")
include("./Optim.jl")

using ControlSystems, CSV, CurveFit, DataFrames, DSP, FFTW, KernelDensity, LsqFit, NeuralDynamics, Plots, Statistics, StatsBase
using .RafalModel: create_rafal_model, simulate_rafal_model
using .BenoitModel: create_benoit_model, simulate_benoit_model
using .ByrneModel: create_byrne_pop, create_byrne_pop_EI, create_byrne_network, create_if_pop, simulate_if_pop, simulate_byrne_EI_network
using .Stimulation: create_stimulus, create_stim_response, yousif_transfer

using .Signal: get_pow_spec, get_hilbert_amplitude_pdf, get_beta_data

using .analysis: run_spec, run_hilbert_pdf, run_beta_burst, run_plv


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

function gaussian(x, A, μ, σ)
    return A * exp.(-(x - μ).^2 / (2 * σ^2))
end

function plot_spec(df, N, sampling_rate)
    plots = []
    for i in 1:N
        freqs = fftshift(fftfreq(length(df.T[i]), sampling_rate))
        F_E = fftshift(fft(df.R[i].rE .- mean(df.R[i].rE)))
        #F_I = fftshift(fft(df.R[i].rI .- mean(df.R[i].rI)))
        #push!(plots, plot(freqs, [abs.(F_E), abs.(F_I)], xlabel="f", xlim=(0, +10), xticks=0:2:10) )

        plot(freqs, abs.(F_E ./ (1.5*10e3)), xlabel="frequency (Hz)", xlim=(0, +50), xticks=0:10:50, yticks=0:0.5:1.6, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        csv_df = DataFrame(Frequency = freqs, PSD = abs.(F_E))
        #CSV.write("data/psd-"*string(i)*".csv", csv_df)
    end
    #plot(plots..., layout=(1, N), size=(700*N,750))
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

function plot_data_model_features()
    # Load data
    df_psd_data = CSV.read("data/psd.csv", DataFrame)   
    df_psd_model = CSV.read("data/psd-m.csv", DataFrame)
    df_beta_amp_pdf_data = CSV.read("data/beta-hpdf.csv", DataFrame)
    df_beta_amp_pdf_model = CSV.read("data/beta-hpdf-m.csv", DataFrame)
    df_beta_dur_pdf_data = CSV.read("data/beta-dur-hpdf.csv", DataFrame)
    df_beta_dur_pdf_model = CSV.read("data/beta-dur-pdf-m.csv", DataFrame)
    df_plvs_data = CSV.read("data/plvs.csv", DataFrame)
    df_plvs_model = CSV.read("data/plvs-m.csv", DataFrame)

    plot(
        [df_psd_data[!, 1], df_psd_model[!, 1]],
        [df_psd_data[!, 2], df_psd_model[!, 2]],
        xlabel="frequency (Hz)",
        size=(500,500),
        linewidth=3,
        xtickfont=16,
        ytickfont=16,
        label=["data" "model"],
        titlefont=16,
        guidefont=16,
        tickfont=16,
    )
    savefig("plots/optim/comb/psd.png")


    plot(
        df_beta_amp_pdf_data[!, 1],
        [df_beta_amp_pdf_data[!, 2], df_beta_amp_pdf_model[!, 2]],
        xlabel="amplitude",
        size=(500,500),
        linewidth=3, 
        xtickfont=16,
        ytickfont=16,
        label=["data" "model"],
        titlefont=16, 
        guidefont=16,
        tickfont=16,
    )
    savefig("plots/optim/comb/beta-amp-pdf.png")

    plot(
        df_beta_dur_pdf_data[!, 1],
        [df_beta_dur_pdf_data[!, 2], df_beta_dur_pdf_model[!, 2]],
        xlabel="duration (s)",
        size=(500,500),
        linewidth=3, 
        xtickfont=16,
        ytickfont=16,
        label=["data" "model"],
        titlefont=16, 
        guidefont=16,
        tickfont=16,
    )
    savefig("plots/optim/comb/beta-dur-pdf.png")

    plot(
        df_plvs_data[!, 1],
        [df_plvs_data[!, 2], df_plvs_model[!, 2]],
        xlabel="frequency (Hz)",
        size=(500,500),
        linewidth=3,
        xtickfont=16,
        ytickfont=16,
        label=["data" "model"],
        titlefont=16,
        guidefont=16,
        tickfont=16
    )
    savefig("plots/optim/comb/plv.png")
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

function run_byrne_net(N, simulate, range_t, T, dt, theta_E, theta_I, stim)
    theta_E_t = [fill(i, length(range_t)) for i in theta_E]
    theta_I_t = [fill(i, length(range_t)) for i in theta_I]

    R = simulate(N, range_t, dt, theta_E_t, theta_I_t, stim)
    T = [0.0(dt/1000.0):(T/1000.0) for i in 1:length(theta_E)]   # Convert to s
    return DataFrame(T=T, R=R, theta_E=theta_E_t, theta_I=theta_I_t)
end

function run_byrne_if(p, simulate, range_t, dt)
    _, rVu = simulate(p, range_t, dt)
    return DataFrame(t=range_t, rVu=rVu)
end

function plot_hilbert_amplitude_pdf(signal::Array{Float32, 1},T, sampling_rate, bandwidth=0.1)
    x, y, ha = get_hilbert_amplitude_pdf(signal, bandwidth=bandwidth)
    plot(x, y, xlabel="amplitude", ylim=(0.0, 1.0), xlim=(0, 6), xticks=0:2:6, yticks=0:0.5:1.0, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    savefig("plots/hilbert_amp_pdf.png")
    plot(T, ha, xlabel="Amplitude", ylabel="Amplitude")
    savefig("plots/hilbert_amp.png")

    freqs = fftshift(fftfreq(length(T), sampling_rate))
    F_A = fftshift(fft(ha))
    plot(freqs, abs.(F_A ./ (1.5*10e3)), xlabel="envelope frequency (Hz)", xlim=(0, +10), ylim=(0.0, 1.0), linewidth=3, xticks=0:5:10, yticks=0:0.5:1.5, size=(500,500), xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    savefig("plots/hilbert_psd.png")

    csv_df = DataFrame(x=x,y=y)
    #CSV.write("data/hilber-amp-pdf.csv", csv_df)
    csv_df = DataFrame(x=T,y=ha)
    #CSV.write("data/hilber-amp.csv", csv_df)
    csv_df = DataFrame(x=freqs,y=abs.(F_A))
    #CSV.write("data/hilber-psd.csv", csv_df)

end

function main_raf()
    # Parameters (time in s)
    N=2
    #[0.0147128, 19.451, 19.0874, 24.8283, 10.9761, -26.5766, 15.5643]
    #[0.0135525, 10.7344, 28.9111, 21.09, 9.30416, -24.7052, 1.16424]
    #[0.000490727, 29.6002, 29.9636, 18.7491, 22.1327, -0.58056, 28.771]
    #[0.0208265, 2.0853, 27.2393, 11.0874, 0.349061, -9.00186, -6.28161]
    #[0.0371499, 8.14398, 28.6796, 23.8907, 0.487599, -1.42008, -3.58852]
    #[0.0140624, 9.09084, 24.6724, 23.7999, 0.0892136, -24.2169, 9.29577]
    #[0.0210295, 1.11603, 9.62243, 26.7759, 1.81981, 0.952654, 11.3475]
    #p = [0.0292394, 8.70738, 19.7852, 26.7454, 19.4352, 8.48873, -18.6168]
    #[0.0266424, 10.3601, 16.5145, 20.1698, 0.168476, -16.4866, -8.21806]
    #p = [0.0218115, 0.551917, 3.11244, 13.9572, 15.124, -9.31442, -27.2121]
    #p = [0.0293204, 7.8674, 19.6515, 26.1374, 19.2606, 8.92379, -18.5011]

    #p = [0.01668432168662548, 2.4, 2.0, 2.0, 4.0, 0.7, 0.0]
    # Best param 252
    #p = [0.01639675162732601,1.5492684841156006,4.827946186065674,8.769134521484375,7.870232105255127,8.341343879699707,-1.2525522708892822]
    #p = [0.0160626, 1.6523, 6.5157, 6.27421, 7.2687, 0.473071, -8.26678] # Opt to just hpdf
    #p = [0.01639675162732601, 1.6523, 6.5157, 6.27421, 7.2687, 10.5473071, 0.0]

    #p = [0.016329117119312286, 3.9730966091156006, 2.6395351886749268, 6.608642578125, 3.699204444885254, 0.35007891058921814, -0.09347619861364365]
    #p = [0.0163153, 8.4994, 22.1685, 17.0323, 27.1569, 7.88831, -20.8976]
    #p = [0.016624921932816505, 4.1515889167785645, 5.530158519744873, 9.802279472351074, 3.491934299468994, 0.16449561715126038, -0.7124000191688538]
    #BB p = [0.0165082, 4.79867, 7.75704, 9.93353, 2.27035, -0.115528, -1.50204]
    #p = [0.016075320541858673,7.580315589904785,5.22007942199707,7.1634345054626465,0.7875996828079224,0.8672178983688354,-1.7587604522705078, 2.445979595184326,0.2242824137210846]
    #p = [0.0165082, 4.79867, 7.75704, 9.93353, 2.27035, -0.115528, -1.50204]

    #P20
    p = [0.016686, 1.30585, 4.18644, 7.88385, 5.09226, 0.496913, -0.904573]

    #E3
    p = [0.0165725, 2.4, 2.0, 2.0, 20.4199, -5.25897, 3.94641]

    #P7
    #p = [0.016783476, 7.9688745, 0.50424606, 8.942622, 5.3960485, 0.473071, -8.26678]


    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)
    tau_E = Float32(p[1])
    tau_I = Float32(p[1])
    w_EE = Float32(p[2])
    w_EI = Float32(p[3])
    w_IE = Float32(p[4])
    beta = Float32(p[5])
    thE_A = Float32(p[6])
    thI_A = Float32(p[7])
    thE_B = Float32(p[6])
    thI_B = Float32(p[7])

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

    stim=create_stimulus(A, f, range_t)
    response=create_stim_response(stim, range_t)
    #plot(range_t, response, xlabel="time (ms)", ylabel="V", size=(500,500), xlim=(0, 0.1), xticks=0:0.02:0.1, linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    #savefig("jul-test.png")

    stim = zeros(length(range_t))
    #response = fill(0.0, length(range_t)) #yousif_transfer(A, f, range_t)
    #for i in 1:6:T-6
    #    #Start pulse
    #    for j in 0:24
    #        for k in 0:2:10
    #            response[Int64(trunc(i*1000+j*200+k*(1000/130)))] = 0.001684
    #        end
    #    end
    #end
    theta_E = [thE_A, thE_B]
    theta_I = [thI_A, thI_B]
    #stim = response
    df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

    #filter = digitalfilter(Bandpass(3.0,7.0),Butterworth(2))
    #df.R[1].rE = filtfilt(filter, df.R[1].rE)

    #plot_act_time(df, N)
    #plot_spec(df, N, sampling_rate)
    #plot_hilbert_amplitude_pdf(df.R[1].rE, df.T[1], sampling_rate)

    #zscore
    cut_model_signal = df.R[1].rE[100:end]
    cut_model_alt_signal = df.R[2].rE[100:end]
    raw_model_signal = (cut_model_signal .- mean(cut_model_signal)) ./ std(cut_model_signal)
    raw_model_alt_signal = (cut_model_alt_signal .- mean(cut_model_alt_signal)) ./ std(cut_model_alt_signal)
    model_flt_beta = get_beta_data(cut_model_signal)
    model_flt_beta = (model_flt_beta .- mean(model_flt_beta)) ./ std(model_flt_beta)

    df_psd_data = CSV.read("data/psd.csv", DataFrame)   

    run_spec(raw_model_signal, true)
    run_hilbert_pdf(raw_model_signal, true)

    run_beta_burst(model_flt_beta, true)
    run_plv(raw_model_signal, raw_model_alt_signal, true)

    plot(1:length(raw_model_signal), raw_model_signal, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    savefig("plots/optim/model/raw.png")

    plot(1:length(model_flt_beta), model_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    savefig("plots/optim/model/flt_beta.png")
end

function main_byrne()
    # Parameters (time in ms)
    p = [23.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(1.0)

    tau = Float32(p[1])
    ex = Float32(p[2])
    ks = Float32(p[3])
    kv = Float32(p[4])
    gamma = Float32(p[5])
    alpha = Float32(p[6])

    thE_A = Float32(p[7])
    thI_A = Float32(p[8])

    vth = 1.000
    vr = -1.000

    #p = create_byrne_pop(ex, ks, kv, gamma, tau, alpha)
    #p = create_if_pop(1000, ex, ks, kv, gamma, tau, alpha, vth, vr)
    E = create_byrne_pop_EI(ex, gamma, tau)
    I = create_byrne_pop_EI(ex, gamma, tau)
    N = create_byrne_network(N, W, etta, E, I, ks, kv, alpha)
    
    #timescale now ms
    T = 100000.0
    dt = 1.0  
    range_t = 0.0:dt:T
    
    theta_E = [thE_A, thE_A]
    theta_I = [thI_A, thI_A]
    #df = run_byrne_single(p, simulate_byrne_pop, range_t, dt)
    #df = run_byrne_if(p, simulate_if_pop, range_t, dt)
    df= run_byrne_net(N, simulate_byrne_EI_network, range_t, T, dt, theta_E, theta_I, [])

    #timescale now s

    #plot_byrne_single(df)
    
    #zscore
    cut_model_signal = df.R[1].rV_E[100:end]
    cut_model_alt_signal = df.R[2].rV_E[100:end]
    raw_model_signal = (cut_model_signal .- mean(cut_model_signal)) ./ std(cut_model_signal)
    raw_model_alt_signal = (cut_model_alt_signal .- mean(cut_model_alt_signal)) ./ std(cut_model_alt_signal)
    model_flt_beta = get_beta_data(cut_model_signal)
    model_flt_beta = (model_flt_beta .- mean(model_flt_beta)) ./ std(model_flt_beta)  
 
    run_spec(raw_model_signal, true)
    run_hilbert_pdf(raw_model_signal, true)
 
    run_beta_burst(model_flt_beta, true)
    run_plv(raw_model_signal, raw_model_alt_signal, true)
 
    plot(1:length(raw_model_signal), raw_model_signal, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    savefig("plots/optim/model/raw.png")
 
    plot(1:length(model_flt_beta), model_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    savefig("plots/optim/model/flt_beta.png")

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
plot_data_model_features()
