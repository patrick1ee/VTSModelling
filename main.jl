include("./analysis.jl")

include("./BenoitModel.jl")
include("./ByrneModel.jl")
include("./RafalModel.jl")
include("./Stimulation.jl")
include("./Signal.jl")
include("./Optim.jl")
include("./Oscilltrack.jl")

using ControlSystems, CSV, CurveFit, DataFrames, DSP, FFTW, KernelDensity, LsqFit, Measures, NeuralDynamics, Plots, Statistics, StatsBase
using .RafalModel: create_rafal_model, simulate_rafal_model
using .BenoitModel: create_benoit_model, simulate_benoit_model
using .ByrneModel: create_byrne_pop, create_byrne_pop_EI, create_byrne_node, create_byrne_network, create_if_pop, simulate_if_pop, simulate_byrne_EI_network
using .Oscilltrack: Oscilltracker
using .Stimulation: create_stim_response, yousif_transfer, create_stim_block

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

        plot(freqs, abs.(F_E ./ (1.5*10e3)), xlabel="frequency (Hz)", xlim=(0, +50), xticks=0:10:50, yticks=0:0.5:1.6, size=(500,500), linewidth=5, xtickfont=22, ytickfont=22, legend=false, titlefont=22, guidefont=22, tickfont=22, legendfont=22)
        csv_df = DataFrame(Frequency = freqs, PSD = abs.(F_E))
        #CSV.write("data/psd-"*string(i)*".csv", csv_df)
    end
    #plot(plots..., layout=(1, N), size=(700*N,750))
    savefig("plots/spec.png")
end

function plot_two_spec()
    # Load data
    df_psd_1= CSV.read("data/model/psd-16.csv", DataFrame)   
    df_psd_2 = CSV.read("data/model/psd.csv", DataFrame)

    plot(
        df_psd_1[!, 1],
        df_psd_1[!, 2],
        title="Power Spectral Density",
        xlabel="Frequency (Hz)",
        xlim=(6, 40),
        xticks=10:10:40,
        size=(500,500),
        linewidth=4,
        xtickfont=12,
        ytickfont=12,
        label="τ=0.016",
        titlefont=12,
        guidefont=12,
        tickfont=12,
        c=3
    )
    plot!(
        df_psd_2[!, 1],
        df_psd_2[!, 2],
        title="Power Spectral Density",
        xlabel="Frequency (Hz)",
        xlim=(6, 40),
        xticks=10:10:40,
        size=(500,500),
        linewidth=2,
        xtickfont=12,
        ytickfont=12,
        label="τ=0.008",
        titlefont=12,
        guidefont=12,
        tickfont=12,
        c=2,
        linestyle=:dot
    )
    savefig("plots/optim/model/psd-pair.png")

end

function plot_md_spec()
    df_psd_1= CSV.read("data/model/psd-16.csv", DataFrame)   
    df_psd_2 = CSV.read("data/P14/12_02_2024_P14_Ch14_FRQ=10Hz_FULL_CL_phase=0_OL11Hz_STIM_EC_v2/psd.csv", DataFrame)

    plot(
        df_psd_1[!, 1],
        df_psd_1[!, 2],
        title="Power Spectral Density",
        xlabel="Frequency (Hz)",
        xlim=(6, 40),
        xticks=10:10:40,
        size=(500,500),
        linewidth=4,
        xtickfont=12,
        ytickfont=12,
        label="τ=0.016",
        titlefont=12,
        guidefont=12,
        tickfont=12,
        c=3
    )
    plot!(
        df_psd_2[!, 1],
        df_psd_2[!, 2],
        title="Power Spectral Density",
        xlabel="Frequency (Hz)",
        xlim=(6, 40),
        xticks=10:10:40,
        size=(500,500),
        linewidth=2,
        xtickfont=12,
        ytickfont=12,
        label="τ=0.008",
        titlefont=12,
        guidefont=12,
        tickfont=12,
        c=2,
        linestyle=:dot
    )
    savefig("plots/optim/model/psd-pair.png")
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
    plot(
        df.theta,
        [df.rE_max, df.rE_min], 
        legend=false,
        xlabel=df.input_pop[1]*" input",
        ylabel="E amplitude", 
        c=2,
        size=(500, 500),
        xticks=0:0.5:1.5,
        xlim=(0, 1.5),
        linewidth=5,
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        margin=2.5mm
        )
    savefig("plots/diss/wc-oscill-1-bif.png")
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

function plot_data_model_features(csv_data_path)
    # Load data
    df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
    df_psd_model = CSV.read("data/model/psd.csv", DataFrame)
    df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
    df_beta_amp_pdf_model = CSV.read("data/model/bapdf.csv", DataFrame)
    df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
    df_beta_dur_pdf_model = CSV.read("data/model/bdpdf.csv", DataFrame)
    df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)
    df_plvs_model = CSV.read("data/model/plvs.csv", DataFrame)

    plot(
        [df_psd_data[!, 1], df_psd_model[!, 1]],
        [df_psd_data[!, 2], df_psd_model[!, 2]],
        xlabel="Frequency (Hz)",
        title="Power Spectral Density",
        size=(500,500),
        linewidth=3,
        xtickfont=12,
        ytickfont=12,
        label=["data" "model"],
        titlefont=12,
        guidefont=12,
        tickfont=12,
    )
    savefig("plots/optim/comb/psd.png")


    plot(
        df_beta_amp_pdf_data[!, 1],
        [df_beta_amp_pdf_data[!, 2], df_beta_amp_pdf_model[!, 2]],
        xlabel="Amplitude",
        title="Beta Burst Amplitude PDFs",
        size=(500,500),
        linewidth=3, 
        xtickfont=12,
        ytickfont=12,
        label=["data" "model"],
        titlefont=12, 
        guidefont=12,
        tickfont=12,
    )
    savefig("plots/optim/comb/beta-amp-pdf.png")

    plot(
        df_beta_dur_pdf_data[!, 1] .* 1000.0,
        [df_beta_dur_pdf_data[!, 2], df_beta_dur_pdf_model[!, 2]],
        xlabel="Dduration (ms)",
        title="Beta Burst Amplitude PDFs",
        size=(500,500),
        linewidth=3, 
        xtickfont=12,
        ytickfont=12,
        label=["data" "model"],
        titlefont=12, 
        guidefont=12,
        tickfont=12,
    )
    savefig("plots/optim/comb/beta-dur-pdf.png")

    plot(
        df_plvs_data[!, 1],
        [df_plvs_data[!, 2], df_plvs_model[!, 2]],
        xlabel="Frequency (Hz)",
        title="Inter-hemisphere Phase-locking Value",
        size=(500,500),
        linewidth=3,
        xtickfont=12,
        ytickfont=12,
        label=["data" "model"],
        titlefont=12,
        guidefont=12,
        tickfont=12
    )
    savefig("plots/optim/comb/plv.png")
end

function plot_data_model_features_stim(csv_data_path)
    # Load data
    df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
    df_psd_model = CSV.read("data/model/psd.csv", DataFrame)
    df_psd_model_s = CSV.read("data/model-stim/psd.csv", DataFrame)

    df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
    df_beta_amp_pdf_model = CSV.read("data/model/bapdf.csv", DataFrame)
    df_beta_amp_pdf_model_s = CSV.read("data/model-stim/bapdf.csv", DataFrame)

    df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
    df_beta_dur_pdf_model = CSV.read("data/model/bdpdf.csv", DataFrame)
    df_beta_dur_pdf_model_s = CSV.read("data/model-stim/bdpdf.csv", DataFrame)

    df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)
    df_plvs_model = CSV.read("data/model/plvs.csv", DataFrame)
    df_plvs_model_s = CSV.read("data/model-stim/plvs.csv", DataFrame)

    plot(
        [df_psd_data[!, 1], df_psd_model[!, 1], df_psd_model_s[!, 1]],
        [df_psd_data[!, 2], df_psd_model[!, 2], df_psd_model_s[!, 2]],
        xlabel="Frequency (Hz)",
        title="Power Spectral Density",
        size=(500,500),
        linewidth=3,
        xtickfont=12,
        ytickfont=12,
        label=["data" "model" "model-stim"],
        titlefont=12,
        guidefont=12,
        tickfont=12,
    )
    savefig("plots/optim/comb/psd.png")


    plot(
        df_beta_amp_pdf_data[!, 1],
        [df_beta_amp_pdf_data[!, 2], df_beta_amp_pdf_model[!, 2], df_beta_amp_pdf_model_s[!, 2]],
        xlabel="Amplitude",
        title="Beta Burst Amplitude PDFs",
        size=(500,500),
        linewidth=3, 
        xtickfont=12,
        ytickfont=12,
        label=["data" "model" "model-stim"],
        titlefont=12, 
        guidefont=12,
        tickfont=12,
    )
    savefig("plots/optim/comb/beta-amp-pdf.png")

    plot(
        df_beta_dur_pdf_data[!, 1] .* 1000.0,
        [df_beta_dur_pdf_data[!, 2], df_beta_dur_pdf_model[!, 2], df_beta_dur_pdf_model_s[!, 2]],
        xlabel="Dduration (ms)",
        title="Beta Burst Amplitude PDFs",
        size=(500,500),
        linewidth=3, 
        xtickfont=12,
        ytickfont=12,
        label=["data" "model" "model-stim"],
        titlefont=12, 
        guidefont=12,
        tickfont=12,
    )
    savefig("plots/optim/comb/beta-dur-pdf.png")

    plot(
        df_plvs_data[!, 1],
        [df_plvs_data[!, 2], df_plvs_model[!, 2], df_plvs_model_s[!, 2]],
        xlabel="Frequency (Hz)",
        title="Inter-hemisphere Phase-locking Value",
        size=(500,500),
        linewidth=3,
        xtickfont=12,
        ytickfont=12,
        label=["data" "model" "model-stim"],
        titlefont=12,
        guidefont=12,
        tickfont=12
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

function run_max_min_wc_net(model, simulate, range_t, dt, range_theta_input, theta_const, input_pop)
    Lt = length(range_t)
    Lte = length(range_theta_input)
    rE_max = zeros(Lte)
    rE_min = zeros(Lte)

    window = [0.5, 1.0]

    for i in 1:Lte
        thE = input_pop == "E" ? fill(range_theta_input[i], Lt) : fill(theta_const, Lt)
        thI = input_pop == "I" ? fill(range_theta_input[i], Lt) : fill(theta_const, Lt)

        theta_E = [thE, thE]
        theta_I = [thI, thI]
        R = simulate(model, range_t, dt, theta_E, theta_I)

        rE = R[1].rE
        rE_max[i], _ = findmax(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
        rE_min[i], _ = findmin(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
    end
   return DataFrame(theta=range_theta_input, rE_max=rE_max, rE_min=rE_min, input_pop=input_pop)
end

function run_act_time(m, simulate, range_t, dt, theta_E, theta_I, oscilltracker, stimblock)
    theta_E_t = [fill(i, length(range_t)) for i in theta_E]
    theta_I_t = [fill(i, length(range_t)) for i in theta_I]

    R, sd = simulate(m, range_t, dt, theta_E_t, theta_I_t, oscilltracker, stimblock)
    T = [range_t for i in 1:length(theta_E)]
    SD = [sd for i in 1:length(theta_E)]
    return DataFrame(T=T, R=R, sd=SD, theta_E=theta_E_t, theta_I=theta_I_t)
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
    T = [0.0:dt:T for i in 1:length(theta_E)]   # Convert to s
    return DataFrame(T=T, R=R, theta_E=theta_E_t, theta_I=theta_I_t)
end

function run_byrne_if(p, simulate, range_t, dt)
    _, rVu = simulate(p, range_t, dt)
    return DataFrame(t=range_t, rVu=rVu)
end

function plot_hilbert_amplitude_pdf(signal::Array{Float32, 1},T, sampling_rate, bandwidth=0.1)
    x, y, ha = get_hilbert_amplitude_pdf(signal, bandwidth=bandwidth)
    plot(x, y, xlabel="amplitude", ylim=(0.0, 1.0), xlim=(0, 6), xticks=0:2:6, yticks=0:0.5:1.0, size=(500,500), linewidth=5, xtickfont=22, ytickfont=22, legend=false, titlefont=22, guidefont=22, tickfont=22, legendfont=22)
    savefig("plots/hilbert_amp_pdf.png")
    plot(T, ha, xlabel="Amplitude", ylabel="Amplitude")
    savefig("plots/hilbert_amp.png")

    freqs = fftshift(fftfreq(length(T), sampling_rate))
    F_A = fftshift(fft(ha))
    plot(freqs, abs.(F_A ./ (1.5*10e3)), xlabel="envelope frequency (Hz)", xlim=(0, +10), ylim=(0.0, 1.0), linewidth=5, xticks=0:5:10, yticks=0:0.5:1.5, size=(500,500), xtickfont=22, ytickfont=22, legend=false, titlefont=22, guidefont=22, tickfont=22, legendfont=22)
    savefig("plots/hilbert_psd.png")

    csv_df = DataFrame(x=x,y=y)
    #CSV.write("data/hilber-amp-pdf.csv", csv_df)
    csv_df = DataFrame(x=T,y=ha)
    #CSV.write("data/hilber-amp.csv", csv_df)
    csv_df = DataFrame(x=freqs,y=abs.(F_A))
    #CSV.write("data/hilber-psd.csv", csv_df)

end

function main_raf(p; csv_path = "data/model")
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
    #p = [0.016686, 1.30585, 4.18644, 7.88385, 5.09226, 0.04, 0.496913, -0.904573]
    #good_guess = [0.01684509590268135, 2.808759927749634, 2.9388251304626465, 5.182344913482666, 8.326308250427246, 0.595751166343689, 0.3267395496368408]
    #p = good_guess

    #P20 fixedWB
    #p = [0.0160928, 1.30585, 4.18644, 7.88385, 5.09226, 0.53322, -0.693958]

    #P7
    #p = [0.016783476, 7.9688745, 0.50424606, 8.942622, 5.3960485, 0.473071, -8.26678]

    #P20 noise - data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1
    #p = [0.0167907, 1.84502, 8.10264, 4.90234, 3.76054, 0.0673752, 0.27402, 0.275149, -2.27837]
    #p = [0.0163928, 1.30585, 4.18644, 7.88385, 5.09226, 0.0204127, 0.0, 0.53322, -0.693958]

    #P20 - 15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1
    #p = [0.016717, 1.53592, 8.4121, 3.81443, 2.88771, 0.204127, 0.00539509, 0.28448, -4.23767]
    #p = [0.0162904, 0.434542, 6.39134, 6.7426, 5.9051, 0.264186, 0.27402, 0.933472, -5.18417]

    #P9 - 07_02_2024_P9_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v2
    #p = [0.01630263589322567, 1.1694562435150146, 4.138545036315918, 2.86141300201416, 3.389385938644409, 0.1073860228061676, 0.0, 3.007591962814331, 0.5085129141807556]
    #p = [0.0166656, 0.118731, 8.28523, 4.19118, 8.46533, 0.172144, 0.0, 0.960148, -4.25145]

    #P4 - 05_02_2024_P4_Ch14_FRQ=11Hz_FULL_CL_phase=0_REST_EC_v1
    #p = [0.01658759079873562,1.9765267372131348,7.805781841278076,8.708059310913086,8.914440155029297,0.16979466378688812,0.0, 7.87988805770874,-5.185361862182617]
    #p = [0.04485432058572769,2.053753137588501,8.179677963256836,9.587191581726074,4.211299419403076,0.15265828371047974,0.0, 8.231314659118652,-0.49894988536834717]
    #p = [0.0128556, 0.382607, 2.21226, 6.21208, 9.48622, 0.271685, 0.0, 6.98884, 0.745291]
    #p = [0.0166703, 0.634375, 9.93469, 7.61141, 5.06469, 0.127048, 0.0, 0.794265, -6.08712]
   # p = [0.0103056, 0.238631, 7.27788, 4.41397, 8.84963, 0.154915, 0.0, 4.92246, 0.102291]
   # p = [0.0167882, 0.47009, 9.81269, 8.75759, 4.55063, 0.113229, 0.0, 0.749659, -5.39723]
    #Alpha oscillations
    #p = [0.016, 2.4, 2.0, 2.0, 4.0, 0.75, 0.5, 0.0]

    #p = [0.0167907, 1.84502, 8.10264, 4.90234, 3.76054, 0.0673752, 0.0, 0.275149, -2.27837]

    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)
    tau_E = Float32(p[1])
    tau_I = Float32(p[1])
    w_EE = Float32(p[2])
    w_EI = Float32(p[3])
    w_IE = Float32(p[4])
    beta = Float32(p[5])
    noise_dev = Float32(p[6])
    stim_mag = Float32(p[7])
    thE_A = Float32(p[8])
    thI_A = Float32(p[9])
    thE_B = Float32(p[8])
    thI_B = Float32(p[9])

    model = create_benoit_model(N, W, etta, tau_E, tau_I, w_EE, w_EI, w_IE, beta, noise_dev, stim_mag)
    
    T = 100.0
    dt = 0.001
    range_t = 0.0:dt:T
    sampling_rate = 1.0 / dt

    stimBlock = create_stim_block(100.0, 25, 25, 0.0:dt:100.0, 1)
    #plot(stimBlock)
    #savefig("./myplot.png")
    SR = 1000.0
    gamma_param = 0.1 # or 0.05
    OT_suppress = 0.3
    target_phase = pi / 2.0
    target_freq = 10.0
    oscilltracker = Oscilltracker(target_freq, target_phase, SR, OT_suppress, gamma_param)

    #df = run_max_min_wc_net(model, simulate_benoit_model, range_t, dt, 0.0:0.01:2.0, 0.0, "E")
    #plot_max_min(df)
    #return

    #E_A = 0.1
    #E_f = 4
    #E_base = 0.6
    #E_phase = 0.0
    #I_A = 0.0
    #I_f = 4
    #I_base = 0.0
    #I_phase = -(pi / 3)
    #df = run_act_oscill_time(model, simulate_benoit_model, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)

    theta_E = [thE_A, thE_B]
    theta_I = [thI_A, thI_B]

    df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, oscilltracker, stimBlock)

    #filter = digitalfilter(Bandpass(3.0,7.0),Butterworth(2))
    #df.R[1].rE = filtfilt(filter, df.R[1].rE)

    #plot_act_time(df, N)
    #plot_spec(df, N, sampling_rate)
    #plot_hilbert_amplitude_pdf(df.R[1].rE, df.T[1], sampling_rate)

    #zscore
    cut_model_signal = df.R[1].rE[1:end]
    cut_model_alt_signal = df.R[2].rE[1:end]
    raw_model_signal = (cut_model_signal .- mean(cut_model_signal)) ./ std(cut_model_signal)
    raw_model_alt_signal = (cut_model_alt_signal .- mean(cut_model_alt_signal)) ./ std(cut_model_alt_signal)

    #plot(1:length(raw_model_signal), raw_model_signal, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=5, xtickfont=22, ytickfont=22, legend=false, titlefont=22, guidefont=22, tickfont=22, legendfont=22)
    #savefig("plots/optim/model/raw.png")

    model_flt_beta = get_beta_data(cut_model_signal)
    model_flt_beta = (model_flt_beta .- mean(model_flt_beta)) ./ std(model_flt_beta)  

    plot_path = "plots/optim/model"
 
    run_spec(raw_model_signal, plot_path, csv_path)
    run_hilbert_pdf(raw_model_signal, true)
 
    run_beta_burst(model_flt_beta, plot_path, csv_path)
    run_plv(raw_model_signal, raw_model_alt_signal, plot_path, csv_path)
 
    plot(df.sd[1])
    savefig("plots/optim/model/stim_delivered.png")

    # E-I plots
    #=raw_model_signal_I = (df.R[1].rI .- mean(df.R[1].rI)) ./ std(df.R[1].rI)
    raw_model_alt_signal_I = (df.R[2].rI .- mean(df.R[2].rI)) ./ std(df.R[2].rI)
    p1 = plot(
        range_t[1:1000],
        raw_model_signal[1:1000], 
        xlabel="Time (s)", 
        title="Activity of Node 1",
        xticks=0:200.0:1000.0,
        yticks=-2:2:2,
        ylim=(-3, 3),
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        color=2,
        legend=:topright,
        label="Excitatory",
        legendfont=22,
        )
    plot!(
        p1,
        range_t[1:1000],
        raw_model_signal_I[1:1000],
        xlabel="Time (s)",
        title="Activity of Node 1",
        xticks=0:200.0:1000.0,
        yticks=-2:2:2,
        ylim=(-3, 3),
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        color=1,
        legend=:topright,
        label="Inhibitory",
        legendfont=22,
        )
    p2 = plot(
        range_t[1:1000],
        raw_model_alt_signal[1:1000], 
        xlabel="Time (s)", 
        title="Activity of Node 2",
        xticks=0:200.0:1000.0,
        yticks=-2:2:2,
        ylim=(-3, 3),
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        color=2,
        legend=false,
        )
    plot!(
        p2,
        range_t[1:1000],
        raw_model_alt_signal_I[1:1000],
        xlabel="Time (s)",
        title="Activity of Node 2",
        xticks=0:200.0:1000.0,
        yticks=-2:2:2,
        ylim=(-3, 3),
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        legend=false,
        color=1,
        )
    plot(p1, p2, layout=(2,1))
    savefig("plots/diss/wc-oscill-alpha-noise.png")=#
end

function main_byrne()
    # Parameters (time in ms)
    #p = [23.1, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    #p = [16.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0]

    #F1 - data/P7/06_02_2024_P7_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1
    #p = [23.0953, 0.762638, 0.657167, 0.347283, 0.0464327, 0.5, 0.0, 3.92163, 1.51794]

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.0)

    tau = Float32(1000.0)
    ex_E = Float32(5.0)
    ex_I = Float32(-3.0)
    kv_E = Float32(0.5)
    kv_I = Float32(0.5)
    
    ks_EE = Float32(15.0)
    ks_EI = Float32(-15.0)
    ks_IE = Float32(25.0)
    ks_II = Float32(-15.0)
    kv_EI = Float32(0.0)
    alpha_EE = Float32(0.2)
    alpha_EI = Float32(0.07)
    alpha_IE = Float32(0.1)
    alpha_II = Float32(0.06)
    gamma = Float32(0.5)
    noise_dev = Float32(0.0)

    thE_A = Float32(0.0)
    thI_A = Float32(0.0)

    vth = 1.000
    vr = -1.000

    #p = create_byrne_pop(ex, ks, kv, gamma, tau, alpha)
    #p = create_if_pop(1000, ex, ks, kv, gamma, tau, alpha, vth, vr)
    E = create_byrne_pop_EI(tau, ex_E, kv_E, gamma)
    I = create_byrne_pop_EI(tau, ex_I, kv_I, gamma)
    N1 = create_byrne_node(E, I, ks_EE, ks_EI, ks_IE, ks_II, kv_EI, alpha_EE, alpha_EI, alpha_IE, alpha_II, noise_dev)
    N2 = create_byrne_node(E, I, ks_EE, ks_EI, ks_IE, ks_II, kv_EI, alpha_EE, alpha_EI, alpha_IE, alpha_II, noise_dev)
    model = create_byrne_network([N1, N2], W, etta, noise_dev)
    
    #timescale now ms
    T = 100000.0
    dt = 1.0 
    range_t = 0.0:dt:T
    
    theta_E = [thE_A, thE_A]
    theta_I = [thI_A, thI_A]
    #df = run_byrne_single(p, simulate_byrne_pop, range_t, dt)
    #df = run_byrne_if(p, simulate_if_pop, range_t, dt)
    df= run_byrne_net(model, simulate_byrne_EI_network, range_t, T, dt, theta_E, theta_I, [])

    #timescale now ms

    #plot_byrne_single(df)
    
    #zscore
    cut_model_signal = df.R[1].rV_E[1:end]
    cut_model_alt_signal = df.R[2].rV_E[1:end]
    raw_model_signal = (cut_model_signal .- mean(cut_model_signal)) ./ std(cut_model_signal)
    raw_model_alt_signal = (cut_model_alt_signal .- mean(cut_model_alt_signal)) ./ std(cut_model_alt_signal)

    #plot(1:length(raw_model_signal), raw_model_signal, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=5, xtickfont=22, ytickfont=22, legend=false, titlefont=22, guidefont=22, tickfont=22, legendfont=22)
    #savefig("plots/optim/model/raw.png")

    #model_flt_beta = get_beta_data(cut_model_signal)
    #model_flt_beta = (model_flt_beta .- mean(model_flt_beta)) ./ std(model_flt_beta)  

    plot_path = "plots/optim/model"
    csv_path = "data/model"

    df_csv = DataFrame(t=range_t, raw=raw_model_alt_signal)
    CSV.write(csv_path*"/raw.csv", df_csv)
 
    run_spec(raw_model_signal, plot_path, csv_path)
    #run_hilbert_pdf(raw_model_signal, true)
 
    #run_beta_burst(model_flt_beta, plot_path, csv_path)
    #run_plv(raw_model_signal, raw_model_alt_signal, plot_path, csv_path)
 
    #plot(1:length(model_flt_beta), model_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=5, xtickfont=22, ytickfont=22, legend=false, titlefont=22, guidefont=22, tickfont=22, legendfont=22)
    #savefig("plots/optim/model/flt_beta.png")

    # E-I plots
    raw_model_signal_I = (df.R[1].rV_I .- mean(df.R[1].rV_I)) ./ std(df.R[1].rV_I)
    raw_model_alt_signal_I = (df.R[2].rV_I .- mean(df.R[2].rV_I)) ./ std(df.R[2].rV_I)
    p1 = plot(
        range_t[1:10000],
        raw_model_signal[1:10000], 
        xlabel="Time (s)", 
        title="Activity of Node 1",
        xticks=0:200.0:10000.0,
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        color=2,
        legend=:topright,
        label="Excitatory",
        legendfont=22,
        )
    plot!(
        p1,
        range_t[1:10000],
        raw_model_signal_I[1:10000],
        xlabel="Time (s)",
        title="Activity of Node 1",
        xticks=0:2000.0:10000.0,
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        color=1,
        legend=:topright,
        label="Inhibitory",
        legendfont=22,
        )
    p2 = plot(
        range_t[1:10000],
        raw_model_alt_signal[1:10000], 
        xlabel="Time (s)", 
        title="Activity of Node 2",
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        color=2,
        legend=false,
        )
    plot!(
        p2,
        range_t[1:10000],
        raw_model_alt_signal_I[1:10000],
        xlabel="Time (s)",
        title="Activity of Node 2",
        xticks=0:2000.0:10000.0,
        size=(1000, 1000),
        margin=10mm,
        linewidth=5,
        xtickfont=22,
        ytickfont=22,
        titlefont=22,
        guidefont=22,
        tickfont=22,
        legend=false,
        color=1,
        )
    plot(p1, p2, layout=(2,1))
    savefig("plots/diss/bc-oscill-alpha.png")

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

#main_byrne()
#plot_data_model_features("data/P7/06_02_2024_P7_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1")

p = [0.0167907, 1.84502, 8.10264, 4.90234, 3.76054, 0.0673752, 0.27402, 0.275149, -2.27837]
ps = [0.0162904, 0.434542, 6.39134, 6.7426, 5.9051, 0.264186, 0.27402, 0.933472, -5.18417]
main_raf(p)
main_raf(ps, csv_path = "data/model-stim")
#plot_data_model_features("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1")
plot_data_model_features_stim("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1")
#plot_data_model_features("data/P9/07_02_2024_P9_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v2")
#plot_data_model_features("data/P4/05_02_2024_P4_Ch14_FRQ=11Hz_FULL_CL_phase=0_REST_EC_v1")

#plot_md_spec()