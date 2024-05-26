module analysis

    include("./Signal.jl")

    using ApproxFun, Base.Filesystem, CSV, DataFrames, DSP, FFTW, HDF5, Interpolations, KernelDensity, KissSmoothing, LPVSpectral, LsqFit, Measures, NaNStatistics, Plots, StatsBase, StatsPlots, Statistics

    using .Signal: get_bandpassed_signal, get_beta_data, get_pow_spec, get_hilbert_amplitude_pdf, get_burst_profiles, get_signal_phase

    const SR = 1000  # recording sampling rate in Hz, do not change this

    export run_spec, run_hilbert_pdf, run_beta_burst, run_plv, get_raw

    function interpolate_nan(arr)
        Larr = length(arr)
        for i in 1:Larr
            if isnan(arr[i])
                j = i
                while isnan(arr[j])
                    j += 1
                end
                arr[i:j-1] .= arr[i-1] .+ (1:j-i) .* (arr[j] .- arr[i-1]) ./ (j-i)
            end
        end
        
        return arr
    end

    function model(x,p) 
        f = Fun(Chebyshev(Interval(0,50)),p)
        f.(x)
    end

    function get_chan_idx(chan_order, chan_name)
        chan_idx = findall(x -> x == chan_name, chan_order.chan_name)
        return chan_idx
    end

    function return_ref_chan(EEG_LOC, chan_order)
        EEG_chan = 0
        REF_chan = 0
        if EEG_LOC == "C3_POz"
            EEG_chan = get_chan_idx(chan_order, "C3")
            REF_chan = get_chan_idx(chan_order, "POz")
        elseif EEG_LOC == "C4_POz"
            EEG_chan = get_chan_idx(chan_order, "C4")
            REF_chan = get_chan_idx(chan_order, "POz")
        elseif EEG_LOC == "C3_local"
            EEG_chan = get_chan_idx(chan_order, "C3")
            REF_chan = [
                get_chan_idx(chan_order, "FC5"),
                get_chan_idx(chan_order, "FC1"),
                get_chan_idx(chan_order, "CP5"),
                get_chan_idx(chan_order, "CP1")
            ]
        elseif EEG_LOC == "C4_local"
            EEG_chan = get_chan_idx(chan_order, "C4")
            REF_chan = [
                get_chan_idx(chan_order, "FC6"),
                get_chan_idx(chan_order, "FC2"),
                get_chan_idx(chan_order, "CP6"),
                get_chan_idx(chan_order, "CP2")
            ]
        elseif EEG_LOC == "CP1_local"
            EEG_chan = get_chan_idx(chan_order, "CP1")
            REF_chan = [
                get_chan_idx(chan_order, "Cz"),
                get_chan_idx(chan_order, "C3"),
                get_chan_idx(chan_order, "Pz"),
                get_chan_idx(chan_order, "P3")
            ]
        elseif EEG_LOC == "CP2_local"
            EEG_chan = get_chan_idx(chan_order, "CP2")
            REF_chan = [
                get_chan_idx(chan_order, "Cz"),
                get_chan_idx(chan_order, "C4"),
                get_chan_idx(chan_order, "Pz"),
                get_chan_idx(chan_order, "P4")
            ]
        elseif EEG_LOC == "CP5_local"
            EEG_chan = get_chan_idx(chan_order, "CP5")
            REF_chan = [
                get_chan_idx(chan_order, "T7"),
                get_chan_idx(chan_order, "C3"),
                get_chan_idx(chan_order, "P3"),
                get_chan_idx(chan_order, "P7")
            ]
        elseif EEG_LOC == "CP6_local"
            EEG_chan = get_chan_idx(chan_order, "CP6")
            REF_chan = [
                get_chan_idx(chan_order, "C4"),
                get_chan_idx(chan_order, "T8"),
                get_chan_idx(chan_order, "P4"),
                get_chan_idx(chan_order, "P8")
            ]
        elseif EEG_LOC == "P3_local"
            EEG_chan = get_chan_idx(chan_order, "P3")
            REF_chan = [
                get_chan_idx(chan_order, "CP5"),
                get_chan_idx(chan_order, "CP1"),
                get_chan_idx(chan_order, "Pz"),
                get_chan_idx(chan_order, "P7")
            ]
        elseif EEG_LOC == "P4_local"
            EEG_chan = get_chan_idx(chan_order, "P4")
            REF_chan = [
                get_chan_idx(chan_order, "CP2"),
                get_chan_idx(chan_order, "CP6"),
                get_chan_idx(chan_order, "Pz"),
                get_chan_idx(chan_order, "P8")
            ]
        elseif EEG_LOC == "FC1_local"
            EEG_chan = get_chan_idx(chan_order, "FC1")
            REF_chan = [
                get_chan_idx(chan_order, "F3"),
                get_chan_idx(chan_order, "Fz"),
                get_chan_idx(chan_order, "C3"),
                get_chan_idx(chan_order, "Cz")
            ]
        elseif EEG_LOC == "FC2_local"
            EEG_chan = get_chan_idx(chan_order, "FC2")
            REF_chan = [
                get_chan_idx(chan_order, "Fz"),
                get_chan_idx(chan_order, "F4"),
                get_chan_idx(chan_order, "Cz"),
                get_chan_idx(chan_order, "C4")
            ]
        elseif EEG_LOC == "F3_local"
            EEG_chan = get_chan_idx(chan_order, "F3")
            REF_chan = [
                get_chan_idx(chan_order, "F7"),
                get_chan_idx(chan_order, "FC5"),
                get_chan_idx(chan_order, "FC1"),
                get_chan_idx(chan_order, "Fz")
            ]
        elseif EEG_LOC == "F4_local"
            EEG_chan = get_chan_idx(chan_order, "F4")
            REF_chan = [
                get_chan_idx(chan_order, "Fz"),
                get_chan_idx(chan_order, "FC2"),
                get_chan_idx(chan_order, "FC6"),
                get_chan_idx(chan_order, "F8")
            ]
        elseif EEG_LOC == "POz_local"
            EEG_chan = get_chan_idx(chan_order, "POz")
            REF_chan = [
                get_chan_idx(chan_order, "O1"),
                get_chan_idx(chan_order, "O2"),
                get_chan_idx(chan_order, "P3"),
                get_chan_idx(chan_order, "P4")
            ]
        end
        return EEG_chan, REF_chan
    end

    function run_beta_data(sig)
        data_flt_beta = get_beta_data(sig)

        freq, power = get_pow_spec(data_flt_beta)
        plot(freq, power, xlabel="frequency (Hz)", xlim=(6, 55), xticks=6:4:55, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig("plots/optim/data/beta-psd.png")
        return data_flt_beta
    end

    function run_spec(sig, plot_path, csv_path; freqs=Nothing, sampling_rate=1000)
        freq, power = get_pow_spec(sig, freqs, sampling_rate)

        csv_df = DataFrame(Frequency = freq, PSD = abs.(power))
        CSV.write(csv_path*"/psd.csv", csv_df)

        plot(
            freq,
            power, 
            legend=false,
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            c=2,
            size=(500, 500),
            xlim=(6, 30),
            xticks=10:10:30,
            linewidth=5,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            margin=2.5mm
        )
        savefig(plot_path*"/psd.png")
    end

    function run_hilbert_pdf(signal, model, bandwidth=0.1)
        x, y, ha = get_hilbert_amplitude_pdf(signal, bandwidth=bandwidth)

        plot(x, y, xlabel="amplitude", ylim=(0.0, 2.0), xlim=(0, 6), xticks=0:2:6, yticks=0:0.5:2.0, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        csv_df = DataFrame(x=x,y=y)
        
        if model
            savefig("plots/optim/model/hpdf.png")
            CSV.write("data/hpdf-m.csv", csv_df)
        else 
            savefig("plots/optim/data/hpdf.png")
            CSV.write("data/hpdf.csv", csv_df)
        end

    end

    function run_beta_burst(signal, plot_path, csv_path, bandwidth=0.1)
        x, y, ha = get_hilbert_amplitude_pdf(signal, bandwidth=bandwidth)
        S, N = denoise(convert(AbstractArray{Float64}, ha))

        csv_df = DataFrame(x=1:length(signal),y=S)
        CSV.write(csv_path*"/bamp.csv", csv_df)
        plot(1:length(signal), S, xlabel="time", size=(500,500), xlim=(0, 10000), xticks=0:2000:10000, linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig(plot_path*"/bamp.png")

        ax, ay, dx, dy, burst_amps, burst_durations = get_burst_profiles(S)

        csv_df = DataFrame(x=ax,y=ay)
        CSV.write(csv_path*"/bapdf.csv", csv_df)
        plot(ax, ay, xlabel="amplitude", title="Beta Burst Amplitude PDF", c=2, ylim=(0.0, 2.0), xlim=(0, 6), xticks=0:2:6, yticks=0:0.5:2.0, size=(500,500), linewidth=3, xtickfont=12, ytickfont=12, legend=false, titlefont=12, guidefont=12, tickfont=12, legendfont=12)
        savefig(plot_path*"/bapdf.png")

        csv_df = DataFrame(x=dx,y=dy)
        CSV.write(csv_path*"/bdpdf.csv", csv_df)
        plot(dx, dy, xlabel="duration", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig(plot_path*"/bdpdf.png")

    end

    function run_plv(s1, s2, plot_path, csv_path)
        plvs = []
        freqs = 6:29
        for f in freqs
            s1f = get_bandpassed_signal(s1, f-0.5, f+0.5)
            s2f = get_bandpassed_signal(s2, f-0.5, f+0.5)
            t= time()
            p1f = get_signal_phase(s1f, Float64(f))
            p2f = get_signal_phase(s2f, Float64(f))
            plv = abs(mean(exp.(1im*(p1f .- p2f))))
            push!(plvs, plv)
        end

        p1 = angle.(hilbert(s1))
        p2 = angle.(hilbert(s2))

        csv_df = DataFrame(x=1:length(s1),y=p1)
        CSV.write(csv_path*"/p1.csv", csv_df)
        
        plot(1:length(s1), p1)
        savefig(plot_path*"/p1.png")

        csv_df = DataFrame(x=1:length(s2),y=p2)
        CSV.write(csv_path*"/p2.csv", csv_df)
        
        plot(1:length(s2), p2)
        savefig(plot_path*"/p2.png")

        S, N = denoise(convert(AbstractArray{Float64}, plvs))
        csv_df = DataFrame(Frequency = freqs, PLV = S)
        CSV.write(csv_path*"/plvs.csv", csv_df)

        plot(freqs, plvs, c=2, size=(500,500), linewidth=3, xtickfont=12, ytickfont=12, ylabel="PLV", title="Phase-locking Value", xlabel="Frequency (Hz)", legend=false, titlefont=12, guidefont=12, tickfont=12, legendfont=12)
        savefig(plot_path*"/plvs.png")

        #println("Max PLV: "*string(maximum(plvs))*", Min PLV: "*string(minimum(plvs)))
    end

    function parse_eeg_data(data, CONST_REF_CHAN)
        chan_order = CSV.read("matlab_analyses/EEG_channel_order.csv", DataFrame)
        WIN_START = 5  # sec
        WIN_END = 75  # sec

        EEG_chan, REF_chan = return_ref_chan(CONST_REF_CHAN, chan_order)
        EEG = data[EEG_chan, :] .- mean([data[i] for i in REF_chan])
        EEG[1] = EEG[2]  # replace the first 0 with the second value
        EEG_preSubtract = EEG[WIN_START * SR: WIN_END * SR]

        movingMean = movmean(EEG_preSubtract, SR * 1)
        EEG_data = EEG_preSubtract .- movingMean

        med_abs_dev = 1.4826 * median(abs.(EEG_data .- median(EEG_data)))
        med_abs_dev_scores = (EEG_data .- median(EEG_data)) ./ med_abs_dev
        OUTL_THRESHOLD = 5
        println("$sum(abs.(med_abs_dev_scores .> OUTL_THRESHOLD)) samples removed as outlier.")

        data_outlierRem = copy(EEG_data)
        data_outlierRem[abs.(med_abs_dev_scores) .> OUTL_THRESHOLD] .= NaN
        data_outlierRem = interpolate_nan(data_outlierRem)

        return data_outlierRem
    end

    function init_data()
        data_path = "Patrick_data"

        # Define the size of the time-window for computing the ERP (event-related potential)
        ERP_tWidth = 1  # [sec]
        ERP_tOffset = ERP_tWidth / 2  # [sec]
        ERP_LOWPASS_FILTER = 30  # set to nan if you want to deactivate it

        POW_SPECTRUM_FREQS = 6:55  # in Hz

        currSubj = "P20"
        filename = "/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1"

        fid = h5open(data_path*"/"*currSubj*"/"*filename*".hdf5", "r")
        data = read(fid["EEG"])
        close(fid)

        CONST_REF_CHAN = "CP1_local"
        CONST_ALT_CHAN = "CP2_local"

        data_outlierRem = parse_eeg_data(data, CONST_REF_CHAN)
        data_outlierRem_alt = parse_eeg_data(data, CONST_ALT_CHAN)

        data_flt_beta = get_beta_data(data_outlierRem)
        data_raw = data_outlierRem
        data_raw_alt = data_outlierRem_alt

        return data_raw, data_flt_beta, data_raw_alt
    end

    function init_slices()
        NUM_SLICES = 10.0

        data_raw, data_flt_beta, data_raw_alt = init_data()
        #zscore
        data_raw = (data_raw .- mean(data_raw)) ./ std(data_raw)
        data_flt_beta = (data_flt_beta .- mean(data_flt_beta)) ./ std(data_flt_beta)

        data_raw_alt = (data_raw_alt .- mean(data_raw_alt)) ./ std(data_raw_alt)

        xPSD = []
        yPSD = []
        xBAPDF = []
        yBAPDF = []
        xBDPF = []
        yBDPF = []
        xPLV = []
        yPLV = []

        for i in 1:NUM_SLICES
            slice_start = Int(i * length(data_raw) / NUM_SLICES)
            slice_end = Int((i+1) * length(data_raw) / NUM_SLICES)
            slice_data = data_raw[slice_start:slice_end]
            slice_data_alt = data_raw_alt[slice_start:slice_end]
            slice_data_flt_beta = data_flt_beta[slice_start:slice_end]
            
            freq, power = get_pow_spec(slice_data)
            push!(xPSD, freq)
            push!(yPSD, power)
            
            x, y, ha = get_hilbert_amplitude_pdf(slice_data_flt_beta)
            S, N = denoise(convert(AbstractArray{Float64}, ha))
            bx, by, burst_durations = get_burst_profiles(S)

            push!(xBAPDF, x)
            push!(yBAPDF, y)
            push!(xBDPF, bx)
            push!(yBDPF, by)

            plvs = []
            freqs = 6:0.1:29
            for f in freqs
                s1f = get_bandpassed_signal(s1, f-0.5, f+0.5)
                s2f = get_bandpassed_signal(s2, f-0.5, f+0.5)
                p1f = get_signal_phase(s1f)
                p2f = get_signal_phase(s2f)
                plv = abs(mean(exp.(1im*(p1f .- p2f))))
                push!(plvs, plv)
            end
            
            push!(xPLV, freqs)
            push!(yPLV, plvs)
        end

        return xPSD, yPSD, xBAPDF, yBAPDF, xBDPF, yBDPF, xPLV, yPLV
    end

    function analyse_all_flat()
        data_path = "Patrick_data"

        # Define the size of the time-window for computing the ERP (event-related potential)
        ERP_tWidth = 1  # [sec]
        ERP_tOffset = ERP_tWidth / 2  # [sec]
        ERP_LOWPASS_FILTER = 30  # set to nan if you want to deactivate it

        POW_SPECTRUM_FREQS = 6:55  # in Hz

        fcount = 0

        data_path = "Patrick_data/"
        for (root, dirs, files) in walkdir(data_path)
            for file in files
                if endswith(file, ".hdf5")
                    full = joinpath(root, file)
                    P = split(full, "/")[2]
                    F = split(split(full, "/")[3], ".")[1]
                    
                    csv_path_parent = "data/"*P
                    csv_path = "data/"*P*"/"*F

                    if !isdir(csv_path_parent)
                        mkdir(csv_path_parent)
                    end
                    if !isdir(csv_path)
                        mkdir(csv_path)
                    end
                    
                    plot_path_parent = "plots/data/"*P
                    plot_path = "plots/data/"*P*"/"*F
                    if !isdir(plot_path_parent)
                        mkdir(plot_path_parent)
                    end
                    if !isdir(plot_path)
                        mkdir(plot_path)
                    end

                    fid = h5open(full, "r")
                    data = read(fid["EEG"])
                    close(fid)

                    CONST_REF_CHAN = "CP1_local"
                    CONST_ALT_CHAN = "CP2_local"

                    data_outlierRem = parse_eeg_data(data, CONST_REF_CHAN)
                    data_outlierRem_alt = parse_eeg_data(data, CONST_ALT_CHAN)

                    data_flt_beta = get_beta_data(data_outlierRem)
                    data_raw = data_outlierRem
                    data_raw_alt = data_outlierRem_alt

                    #zscore
                    data_raw = (data_raw .- mean(data_raw)) ./ std(data_raw)
                    data_flt_beta = (data_flt_beta .- mean(data_flt_beta)) ./ std(data_flt_beta)

                    data_raw_alt = (data_raw_alt .- mean(data_raw_alt)) ./ std(data_raw_alt)
                    
                    #run_spec(data_raw, plot_path, csv_path)
                    #run_beta_burst(data_flt_beta, plot_path, csv_path)
                    run_plv(data_raw, data_raw_alt, plot_path, csv_path)

                    #plot(1:length(data_raw), data_raw, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
                    #savefig(plot_path*"/raw.png")
                    #plot(1:length(data_flt_beta), data_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
                    #savefig(plot_path*"/flt-beta.png")

                    fcount += 1
                    println("Processed $fcount files.")
                end
            end
        end
        return
    end

    function analyse()
        data_raw, data_flt_beta, data_raw_alt = init_data()
        #zscore
        data_raw = (data_raw .- mean(data_raw)) ./ std(data_raw)
        data_flt_beta = (data_flt_beta .- mean(data_flt_beta)) ./ std(data_flt_beta)

        data_raw_alt = (data_raw_alt .- mean(data_raw_alt)) ./ std(data_raw_alt)
        
        run_spec(data_raw, false)
        run_hilbert_pdf(data_raw, false)

        run_beta_burst(data_flt_beta, false)

        #plot(1:100, data_raw[1:1:100], xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        plot(1:length(data_raw), data_raw, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig("plots/optim/data/raw.png")

        plot(1:length(data_flt_beta), data_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig("plots/optim/data/flt_beta.png")

        run_plv(data_raw, data_raw_alt, false)

        print("LEN "*string(length(data_raw))*"\n")
    end

    function get_raw(P, name, lb, ub)
        data_path = "Patrick_data"

        fid = h5open(data_path*"/"*P*"/"*name*".hdf5", "r")
        data = read(fid["EEG"])
        close(fid)

        CONST_REF_CHAN = "CP1_local"
        CONST_ALT_CHAN = "CP2_local"

        data_outlierRem = parse_eeg_data(data, CONST_REF_CHAN)

        data_raw = data_outlierRem
        #zscore
        data_raw = (data_raw .- mean(data_raw)) ./ std(data_raw)
        data_raw = data_raw[lb:ub]

        csv_df = DataFrame(x=1:length(data_raw),y=data_raw)
        CSV.write("./raw.csv", csv_df)
    end

    function get_beta_burst_process(P, name)
        data_path = "Patrick_data"

        fid = h5open(data_path*"/"*P*"/"*name*".hdf5", "r")
        data = read(fid["EEG"])
        close(fid)

        CONST_REF_CHAN = "CP1_local"
        CONST_ALT_CHAN = "CP2_local"

        data_outlierRem = parse_eeg_data(data, CONST_REF_CHAN)
        data_flt_beta = get_beta_data(data_outlierRem)
        data_raw = data_outlierRem

        #zscore
        data_raw = (data_raw .- mean(data_raw)) ./ std(data_raw)
        data_flt_beta = (data_flt_beta .- mean(data_flt_beta)) ./ std(data_flt_beta)
        
        x, y, ha = get_hilbert_amplitude_pdf(data_flt_beta)
        S, N = denoise(convert(AbstractArray{Float64}, ha))

        threshold = percentile(S, 20)
        
        Sc = [S[i] for i in 1:length(S)]
        for i in 1:length(Sc)
            if Sc[i] < threshold
                Sc[i] = threshold
            end
        end

        plot(
            1:length(S), 
            S, 
            xlabel="time (ms)", 
            ylabel="amplitude", 
            xlim=(0, 2500),
            xticks=0:500:2500,
            size=(500,500), 
            linewidth=1.5, 
            xtickfont=12, 
            ytickfont=12, 
            legend=false, 
            titlefont=12, 
            guidefont=12, 
            tickfont=12, 
            title="Beta Bursts Cut-off",
            color="grey",
            right_margin=5Plots.mm
        )
        plot!(
            1:length(S), 
            Sc,
            c=1,
            linewidth=1.5,
        )
        plot!(
            1:length(S), 
            [threshold for i in 1:length(S)], 
            linewidth=3, 
            color="black",
            legend=false, 
        )
        plot!(
            1:length(S),
            [0 for i in 1:length(S)],
            linewidth=0,
            fillrange = [threshold for i in 1:length(S)], 
            fillalpha = 0.25,
            colour="black"
        )
        #=plot!(
            1:length(S),
            [threshold for i in 1:length(S)],
            linewidth=0,
            fillrange = [4.0 for i in 1:length(S)], 
            fillalpha = 0.05,
            color="green"
        )
        plot!(
            1:length(S), 
            [threshold for i in 1:length(S)], 
            linewidth=2, 
            color="red",
            linestyle=:dash,
            legend=false, 
        )=#
        savefig("./plots/diss/beta-burst-process-1.png")
    end

    function plot_example_bursts()
        data_path = "Patrick_data"

        # Define the size of the time-window for computing the ERP (event-related potential)
        ERP_tWidth = 1  # [sec]
        ERP_tOffset = ERP_tWidth / 2  # [sec]
        ERP_LOWPASS_FILTER = 30  # set to nan if you want to deactivate it

        POW_SPECTRUM_FREQS = 6:55  # in Hz

        fcount = 0

        data_path = "Patrick_data/"
        for (root, dirs, files) in walkdir(data_path)
            for file in files
                if endswith(file, ".hdf5")
                    if rand(Int) % 7 > 0
                        continue
                    end

                    full = joinpath(root, file)
                    P = split(full, "/")[2]
                    F = split(split(full, "/")[3], ".")[1]
                    
                    csv_path = "data/"*P*"/"*F

                    csv_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
                    x = csv_df[!, 1] .* 1000.0
                    y = csv_df[!, 2]
                    plot!(
                        x,
                        y, 
                        size=(500,500),
                        xlim=(0, 1000),
                        xlabel="Duration (ms)",
                        title="Beta Burst Duration PDFs",
                        linewidth=3, 
                        xtickfont=12, 
                        ytickfont=12, 
                        legend=false, 
                        titlefont=12, 
                        guidefont=12, 
                        tickfont=12,
                    )

                    fcount += 1
                    println("Processed $fcount files.")
                    if fcount > 4
                        savefig("plots/diss/beta-burst-dpdfs.png")
                        return
                    end
                end
            end
        end
        return
    end

    function plot_feature_ribbons(csv_data_path)
         # Load data
        df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
        df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
        df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
        df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)

        yPSDs = [[] for i in 1:100]
        freq = []
        yBAPDFS = [[] for i in 1:100]
        xBAPDF = []
        yBDPDFS = [[] for i in 1:100]
        xBDPDF = []
        yPLV = [[] for i in 1:100]
        xPLV = []
        for i in 1:100
            csv_path = "data/model-bc-rest-100/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            freq = xPSD = psd_df[!, 1]
            yPSDdat = psd_df[!, 2]
            if any(isnan.(yPSDdat)) 
                continue
            end

            bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
            xBAPDF = bapdf_df[!, 1]
            yBAPDFdat = bapdf_df[!, 2]
            if any(isnan.(yBAPDFdat )) 
                continue
            end

            bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
            xBDPDF = bdpdf_df[!, 1]
            yBDPDFdat = bdpdf_df[!, 2]
            if any(isnan.(yBDPDFdat)) 
                continue
            end

            plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
            xPLV = plvs_df[!, 1]
            yPLVdat = plvs_df[!, 2]
            if any(isnan.(yPLVdat)) 
                continue
            end

            yPSDs[i]=  yPSDdat
            yBAPDFS[i] = yBAPDFdat
            yBDPDFS[i] = yBDPDFdat
            yPLV[i] = yPLVdat
        end

        yPSDs = [y for y in yPSDs if length(y) > 0]
        yBAPDFS = [y for y in yBAPDFS if length(y) > 0]
        yBDPDFS = [y for y in yBDPDFS if length(y) > 0]
        yPLV = [y for y in yPLV if length(y) > 0]

        La = length(yPSDs)

        print(size(yPSDs))

        psd_stds = []
        for i in 1:length(yPSDs[1])
            y = [yPSDs[j][i] for j in 1:La]
            push!(psd_stds, std(y))
        end

        bapdf_stds = []
        for i in 1:length(yBAPDFS[1])
            y = [yBAPDFS[j][i] for j in 1:La]
            push!(bapdf_stds, std(y))
        end

        bdpdf_stds = []
        for i in 1:length(yBDPDFS[1])
            y = [yBDPDFS[j][i] for j in 1:La]
            push!(bdpdf_stds, std(y))
        end

        plv_stds = []
        for i in 1:length(yPLV[1])
            y = [yPLV[j][i] for j in 1:La]
            push!(plv_stds, std(y))
        end

        plot(
            df_psd_data[!, 1],
            df_psd_data[!, 2], 
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            freq,
            mean(yPSDs, dims=1), 
            ribbon=psd_stds,
            fillalpha=.3,
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="model",
            c=3,
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("psd-rib.png")

        plot(
            df_beta_amp_pdf_data[!, 1],
            df_beta_amp_pdf_data[!, 2], 
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xBAPDF,
            mean(yBAPDFS, dims=1), 
            ribbon=bapdf_stds,
            fillalpha=.3,
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="model",
            c=3,
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("bapdf-rib.png")

        plot(
            df_beta_dur_pdf_data[!, 1],
            df_beta_dur_pdf_data[!, 2], 
            xlabel="Duration",
            title="Beta Duration PDF",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xBDPDF,
            mean(yBDPDFS, dims=1), 
            ribbon=bdpdf_stds,
            fillalpha=.3,
            xlabel="Duration",
            title="Beta Duration PDF",
            label="model",
            c=3,
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("bdpdf-rib.png")

        plot(
            df_plvs_data[!, 1],
            df_plvs_data[!, 2], 
            xlabel="Frequency",
            title="Phase Locking Value",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xPLV,
            mean(yPLV, dims=1), 
            ribbon=plv_stds,
            fillalpha=.3,
            legend=false,
            xlabel="Frequency",
            title="Phase Locking Value",
            label="model",
            c=3,
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("plv-rib.png")
    end

    function plot_feature_costs(csv_data_path)
        # Load data
        df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
        df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
        df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
        df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)

        yPSDs = [[] for i in 1:100]
        freq = []
        yBAPDFS = [[] for i in 1:100]
        xBAPDF = []
        yBDPDFS = [[] for i in 1:100]
        xBDPDF = []
        yPLV = [[] for i in 1:100]
        xPLV = []
        for i in 1:100
            csv_path = "data/model/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            freq = xPSD = psd_df[!, 1]
            yPSDdat = psd_df[!, 2]
            yPSDs[i]=  yPSDdat

            bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
            xBAPDF = bapdf_df[!, 1]
            yBAPDFdat = bapdf_df[!, 2]
            yBAPDFS[i] = yBAPDFdat

            bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
            xBDPDF = bdpdf_df[!, 1]
            yBDPDFdat = bdpdf_df[!, 2]
            yBDPDFS[i] = yBDPDFdat

            plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
            xPLV = plvs_df[!, 1]
            yPLVdat = plvs_df[!, 2]
            yPLV[i] = yPLVdat
        end

        psd_costs = []
        for i in 1:length(yPSDs)
            yPSDmod = yPSDs[i]
            yPSDdat = df_psd_data[!, 2]
            if length(yPSDmod) > length(yPSDdat)
                yPSDmod = yPSDmod[1:length(yPSDdat)]
            elseif length(yPSDmod) < length(yPSDdat)
                yPSDdat = yPSDdat[1:length(yPSDmod)]
            end
            cost = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2))
            push!(psd_costs, cost)
        end

        bapdf_costs = []
        for i in 1:length(yBAPDFS)
            cost = (sum((df_beta_amp_pdf_data[!, 2] .- yBAPDFS[i]).^2) / sum((df_beta_amp_pdf_data[!, 2] .- mean(df_beta_amp_pdf_data[!, 2])).^2))
            push!(bapdf_costs, cost)
        end

        bdpdf_costs = []
        for i in 1:length(yBDPDFS)
            cost = (sum((df_beta_dur_pdf_data[!, 2] .- yBDPDFS[i]).^2) / sum((df_beta_dur_pdf_data[!, 2] .- mean(df_beta_dur_pdf_data[!, 2])).^2))
            push!(bdpdf_costs, cost)
        end

        plv_costs = []
        for i in 1:length(yPLV)
            cost = (sum((df_plvs_data[!, 2] .- yPLV[i]).^2) / sum((df_plvs_data[!, 2] .- mean(df_plvs_data[!, 2])).^2))
            push!(plv_costs, cost)
        end

        df = DataFrame(
            Features = ["PSD", "Beta Amplitude PDF", "Beta Duration PDF", "PLV"],
            Costs = [mean(psd_costs), mean(bapdf_costs), mean(bdpdf_costs), mean(plv_costs)]
        )
        for i in 1:100
            push!(df, ["PSD", psd_costs[i]])
            push!(df, ["Beta Amplitude PDF", bapdf_costs[i]])
            push!(df, ["Beta Duration PDF", bdpdf_costs[i]])
            push!(df, ["PLV", plv_costs[i]])
        end

        @df df boxplot(:Features, :Costs, groupby=:Features, legend=false)
        savefig("feature-costs-2.png")
    end

    function plot_feature_ribbons_stim_pair(csv_data_path)
        # Load data
        df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
        df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
        df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
        df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)

        #A = inc-stim

        yPSDs_A = [[] for i in 1:100]
        freq_A = []
        yBAPDFS_A = [[] for i in 1:100]
        xBAPDF_A = []
        yBDPDFS_A = [[] for i in 1:100]
        xBDPDF_A = []
        yPLV_A = [[] for i in 1:100]
        xPLV_A = []
        
        for i in 1:100
            csv_path = "data/model-inc-stim/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            freq_A = xPSD_A = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_A[i]=  yPSDdat_A

            bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
            xBAPDF_A = bapdf_df[!, 1]
            yBAPDFdat_A = bapdf_df[!, 2]
            yBAPDFS_A[i] = yBAPDFdat_A

            bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
            xBDPDF_A = bdpdf_df[!, 1]
            yBDPDFdat_A = bdpdf_df[!, 2]
            yBDPDFS_A[i] = yBDPDFdat_A

            plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
            xPLV_A = plvs_df[!, 1]
            yPLVdat_A = plvs_df[!, 2]
            yPLV_A[i] = yPLVdat_A
        end

        yPSDs_B = [[] for i in 1:100]
        freq_B = []
        yBAPDFS_B = [[] for i in 1:100]
        xBAPDF_B = []
        yBDPDFS_B = [[] for i in 1:100]
        xBDPDF_B = []
        yPLV_B = [[] for i in 1:100]
        xPLV_B = []

        for i in 1:100
            csv_path = "data/model-plus-stim/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            freq_B = xPSD_B = psd_df[!, 1]
            yPSDdat_B = psd_df[!, 2]
            yPSDs_B[i]=  yPSDdat_B

            bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
            xBAPDF_B = bapdf_df[!, 1]
            yBAPDFdat_B = bapdf_df[!, 2]
            yBAPDFS_B[i] = yBAPDFdat_B

            bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
            xBDPDF_B = bdpdf_df[!, 1]
            yBDPDFdat_B = bdpdf_df[!, 2]
            yBDPDFS_B[i] = yBDPDFdat_B

            plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
            xPLV_B = plvs_df[!, 1]
            yPLVdat_B = plvs_df[!, 2]
            yPLV_B[i] = yPLVdat_B
        
        end

        psd_stds_A = []
        for i in 1:length(yPSDs_A[1])
            y = [yPSDs_A[j][i] for j in 1:100]
            push!(psd_stds_A, std(y))
        end

        bapdf_stds_A = []
        for i in 1:length(yBAPDFS_A[1])
            y = [yBAPDFS_A[j][i] for j in 1:100]
            push!(bapdf_stds_A, std(y))
        end

        bdpdf_stds_A = []
        for i in 1:length(yBDPDFS_A[1])
            y = [yBDPDFS_A[j][i] for j in 1:100]
            push!(bdpdf_stds_A, std(y))
        end

        plv_stds_A = []
        for i in 1:length(yPLV_A[1])
            y = [yPLV_A[j][i] for j in 1:100]
            push!(plv_stds_A, std(y))
        end

        psd_stds_B = []
        for i in 1:length(yPSDs_B[1])
            y = [yPSDs_B[j][i] for j in 1:100]
            push!(psd_stds_B, std(y))
        end

        bapdf_stds_B = []
        for i in 1:length(yBAPDFS_B[1])
            y = [yBAPDFS_B[j][i] for j in 1:100]
            push!(bapdf_stds_B, std(y))
        end

        bdpdf_stds_B = []
        for i in 1:length(yBDPDFS_B[1])
            y = [yBDPDFS_B[j][i] for j in 1:100]
            push!(bdpdf_stds_B, std(y))
        end

        plv_stds_B = []
        for i in 1:length(yPLV_B[1])
            y = [yPLV_B[j][i] for j in 1:100]
            push!(plv_stds_B, std(y))
        end

        plot(
            df_psd_data[!, 1],
            df_psd_data[!, 2], 
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            freq_A,
            mean(yPSDs_A, dims=1), 
            ribbon=psd_stds_A,
            fillalpha=.3,
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="stim fit",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            freq_B,
            mean(yPSDs_B, dims=1), 
            ribbon=psd_stds_B,
            fillalpha=.3,
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="rest fit + stim",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("psd-rib.png")

        plot(
            df_beta_amp_pdf_data[!, 1],
            df_beta_amp_pdf_data[!, 2], 
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xBAPDF_A,
            mean(yBAPDFS_A, dims=1), 
            ribbon=bapdf_stds_A,
            fillalpha=.3,
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="stim fit",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xBAPDF_B,
            mean(yBAPDFS_B, dims=1), 
            ribbon=bapdf_stds_B,
            fillalpha=.3,
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="rest fit + stim",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("bapdf-rib.png")

        plot(
            df_beta_dur_pdf_data[!, 1],
            df_beta_dur_pdf_data[!, 2], 
            xlabel="Duration",
            title="Beta Duration PDF",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xBDPDF_A,
            mean(yBDPDFS_A, dims=1), 
            ribbon=bdpdf_stds_A,
            fillalpha=.3,
            xlabel="Duration",
            title="Beta Duration PDF",
            label="stim fit",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xBDPDF_B,
            mean(yBDPDFS_B, dims=1), 
            ribbon=bdpdf_stds_B,
            fillalpha=.3,
            xlabel="Duration",
            title="Beta Duration PDF",
            label="rest fit + stim",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("bdpdf-rib.png")

        plot(
            df_plvs_data[!, 1],
            df_plvs_data[!, 2], 
            xlabel="Frequency",
            title="Phase Locking Value",
            label="data",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xPLV_A,
            mean(yPLV_A, dims=1), 
            ribbon=plv_stds_A,
            fillalpha=.3,
            xlabel="Frequency",
            title="Phase Locking Value",
            label="stim fit",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            xPLV_B,
            mean(yPLV_B, dims=1), 
            ribbon=plv_stds_B,
            fillalpha=.3,
            xlabel="Frequency",
            title="Phase Locking Value",
            label="rest fit + stim",
            size=(500, 500),
            linewidth=3,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("plv-rib.png")

    end

    function plot_feature_costs_stim_pair(csv_data_path)
        # Load data
        df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
        df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
        df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
        df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)

        #A = more noise

        yPSDs_A = [[] for i in 1:100]
        freq_A = []
        yBAPDFS_A = [[] for i in 1:100]
        xBAPDF_A = []
        yBDPDFS_A = [[] for i in 1:100]
        xBDPDF_A = []
        yPLV_A = [[] for i in 1:100]
        xPLV_A = []
        
        for i in 1:100
            csv_path = "data/model/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            freq_A = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_A[i]=  yPSDdat_A

            bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
            xBAPDF_A = bapdf_df[!, 1]
            yBAPDFdat_A = bapdf_df[!, 2]
            yBAPDFS_A[i] = yBAPDFdat_A

            bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
            xBDPDF_A = bdpdf_df[!, 1]
            yBDPDFdat_A = bdpdf_df[!, 2]
            yBDPDFS_A[i] = yBDPDFdat_A

            plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
            xPLV_A = plvs_df[!, 1]
            yPLVdat_A = plvs_df[!, 2]
            yPLV_A[i] = yPLVdat_A
        end

        yPSDs_B = [[] for i in 1:100]
        freq_B = []
        yBAPDFS_B = [[] for i in 1:100]
        xBAPDF_B = []
        yBDPDFS_B = [[] for i in 1:100]
        xBDPDF_B = []
        yPLV_B = [[] for i in 1:100]
        xPLV_B = []

        for i in 1:100
            csv_path = "data/model-less-noise/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            freq_B =  psd_df[!, 1]
            yPSDdat_B = psd_df[!, 2]
            yPSDs_B[i]=  yPSDdat_B

            bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
            xBAPDF_B = bapdf_df[!, 1]
            yBAPDFdat_B = bapdf_df[!, 2]
            yBAPDFS_B[i] = yBAPDFdat_B

            bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
            xBDPDF_B = bdpdf_df[!, 1]
            yBDPDFdat_B = bdpdf_df[!, 2]
            yBDPDFS_B[i] = yBDPDFdat_B

            plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
            xPLV_B = plvs_df[!, 1]
            yPLVdat_B = plvs_df[!, 2]
            yPLV_B[i] = yPLVdat_B
        
        end

        psd_costs_A = []
        for i in 1:length(yPSDs_A)
            yPSDmod = yPSDs_A[i]
            yPSDdat = df_psd_data[!, 2]
            if length(yPSDmod) > length(yPSDdat)
                yPSDmod = yPSDmod[1:length(yPSDdat)]
            elseif length(yPSDmod) < length(yPSDdat)
                yPSDdat = yPSDdat[1:length(yPSDmod)]
            end
            cost = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2))
            push!(psd_costs_A, cost)
        end

        bapdf_costs_A = []
        for i in 1:length(yBAPDFS_A)
            cost = (sum((df_beta_amp_pdf_data[!, 2] .- yBAPDFS_A[i]).^2) / sum((df_beta_amp_pdf_data[!, 2] .- mean(df_beta_amp_pdf_data[!, 2])).^2))
            push!(bapdf_costs_A, cost)
        end

        bdpdf_costs_A = []
        for i in 1:length(yBDPDFS_A)
            cost = (sum((df_beta_dur_pdf_data[!, 2] .- yBDPDFS_A[i]).^2) / sum((df_beta_dur_pdf_data[!, 2] .- mean(df_beta_dur_pdf_data[!, 2])).^2))
            push!(bdpdf_costs_A, cost)
        end

        plv_costs_A = []
        for i in 1:length(yPLV_A)
            cost = (sum((df_plvs_data[!, 2] .- yPLV_A[i]).^2) / sum((df_plvs_data[!, 2] .- mean(df_plvs_data[!, 2])).^2))
            push!(plv_costs_A, cost)
        end

        total_costs_A = (psd_costs_A .+ bapdf_costs_A .+ bdpdf_costs_A .+ plv_costs_A) ./4

        psd_costs_B = []
        for i in 1:length(yPSDs_B)
            yPSDmod = yPSDs_B[i]
            yPSDdat = df_psd_data[!, 2]
            if length(yPSDmod) > length(yPSDdat)
                yPSDmod = yPSDmod[1:length(yPSDdat)]
            elseif length(yPSDmod) < length(yPSDdat)
                yPSDdat = yPSDdat[1:length(yPSDmod)]
            end
            cost = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2))
            push!(psd_costs_B, cost)
        end

        bapdf_costs_B = []
        for i in 1:length(yBAPDFS_B)
            cost = (sum((df_beta_amp_pdf_data[!, 2] .- yBAPDFS_B[i]).^2) / sum((df_beta_amp_pdf_data[!, 2] .- mean(df_beta_amp_pdf_data[!, 2])).^2))
            push!(bapdf_costs_B, cost)
        end

        bdpdf_costs_B = []
        for i in 1:length(yBDPDFS_B)
            cost = (sum((df_beta_dur_pdf_data[!, 2] .- yBDPDFS_B[i]).^2) / sum((df_beta_dur_pdf_data[!, 2] .- mean(df_beta_dur_pdf_data[!, 2])).^2))
            push!(bdpdf_costs_B, cost)
        end

        plv_costs_B = []
        for i in 1:length(yPLV_B)
            cost = (sum((df_plvs_data[!, 2] .- yPLV_B[i]).^2) / sum((df_plvs_data[!, 2] .- mean(df_plvs_data[!, 2])).^2))
            push!(plv_costs_B, cost)
        end

        total_costs_B = (psd_costs_B .+ bapdf_costs_B .+ bdpdf_costs_B .+ plv_costs_B) ./4

        #=dfd = DataFrame(
            Features = ["PSD",],
            Fit = ["stim fit"],
            Costs = [mean(psd_costs_A)]
        )
        for i in 1:100
            push!(dfd, ["PSD", "stim fit", psd_costs_A[i]])
            push!(dfd, ["PSD", "rest fit + stim", psd_costs_B[i]])
            push!(dfd, ["Amp. PDF", "stim fit", bapdf_costs_A[i]])
            push!(dfd, ["Amp. PDF", "rest fit + stim", bapdf_costs_B[i]])
            push!(dfd, ["Dur. PDF", "stim fit", bdpdf_costs_A[i]])
            push!(dfd, ["Dur. PDF", "rest fit + stim", bdpdf_costs_B[i]])
            push!(dfd, ["PLV", "stim fit", plv_costs_A[i]])
            push!(dfd, ["PLV", "rest fit + stim", plv_costs_B[i]])
        end

       stds = [ 
        std(bapdf_costs_B),
        std(bapdf_costs_A), 
        std(bdpdf_costs_B),
        std(bdpdf_costs_A),
        std(plv_costs_B),
        std(plv_costs_A),
        std(psd_costs_B),
        std(psd_costs_A)
       ]=#

       dfn = DataFrame(
          Noise=["0 < Ѯ < 0.3", "0 < Ѯ < 0.05"],
          Costs=[mean(total_costs_A), mean(total_costs_B)]
       )

       print(mean(total_costs_A))
         print(mean(total_costs_B))

       for i in 1:100
           push!(dfn, ["0 < Ѯ < 0.3", total_costs_A[i]])
           push!(dfn, ["0 < Ѯ < 0.05", total_costs_B[i]])
       end


        @df dfn groupedboxplot(
            :Noise, 
            :Costs, 
             groupby=:Costs,
             ylabel="Cost", 
             title="Effect of Noise on Fit Cost & Stability", 
             legend=false, 
             size=(500, 500), 
             xtickfont=12, 
             ytickfont=12, 
             titlefont=12, 
             guidefont=12, 
             tickfont=12, 
             legendfont=12, 
             margin=2.5mm
             )
        savefig("feature-costs-noise.png")

    end

    function plot_data_features_data_stimeffect(csv_data_path, csv_data_path_stim)
        # Load data
        df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
        df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
        df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
        df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)

        df_psd_data_stim = CSV.read(csv_data_path_stim*"/psd.csv", DataFrame)
        df_beta_amp_pdf_data_stim = CSV.read(csv_data_path_stim*"/bapdf.csv", DataFrame)
        df_beta_dur_pdf_data_stim = CSV.read(csv_data_path_stim*"/bdpdf.csv", DataFrame)
        df_plvs_data_stim = CSV.read(csv_data_path_stim*"/plvs.csv", DataFrame)

        plot(
            df_psd_data[!, 1],
            df_psd_data[!, 2], 
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="rest",
            size=(500, 500),
            linewidth=3,
            linestyle=:dot,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            df_psd_data_stim[!, 1],
            df_psd_data_stim[!, 2], 
            xlabel="Frequency (Hz)",
            title="Power Spectral Density",
            label="stim",
            size=(500, 500),
            linewidth=3,
            c=1,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("data-comp-psd-stim.png")

        plot(
            df_beta_amp_pdf_data[!, 1],
            df_beta_amp_pdf_data[!, 2], 
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="rest",
            size=(500, 500),
            linewidth=3,
            c="black",
            linestyle=:dot,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            df_beta_amp_pdf_data_stim[!, 1],
            df_beta_amp_pdf_data_stim[!, 2], 
            xlabel="Amplitude",
            title="Beta Amplitude PDF",
            label="stim",
            size=(500, 500),
            linewidth=3,
            c=1,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("data-comp-bapdf-stim.png")

        plot(
            df_beta_dur_pdf_data[!, 1],
            df_beta_dur_pdf_data[!, 2], 
            xlabel="Duration",
            title="Beta Duration PDF",
            label="rest",
            size=(500, 500),
            linewidth=3,
            linestyle=:dot,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            df_beta_dur_pdf_data_stim[!, 1],
            df_beta_dur_pdf_data_stim[!, 2], 
            xlabel="Duration",
            title="Beta Duration PDF",
            label="stim",
            size=(500, 500),
            linewidth=3,
            c=1,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("data-comp-bdpdf-stim.png")

        plot(
            df_plvs_data[!, 1],
            df_plvs_data[!, 2], 
            xlabel="Frequency",
            title="Phase Locking Value",
            label="rest",
            size=(500, 500),
            linewidth=3,
            linestyle=:dot,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(
            df_plvs_data_stim[!, 1],
            df_plvs_data_stim[!, 2], 
            xlabel="Frequency",
            title="Phase Locking Value",
            label="stim",
            size=(500, 500),
            linewidth=3,
            c=1,
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        savefig("data-comp-plv-stim.png")
    end

    function plot_model_features_stim_effect()
         #A = plus-stim

         yPSDs_A = [[] for i in 1:100]
         freq_A = []
         yBAPDFS_A = [[] for i in 1:100]
         xBAPDF_A = []
         yBDPDFS_A = [[] for i in 1:100]
         xBDPDF_A = []
         yPLV_A = [[] for i in 1:100]
         xPLV_A = []
         
         for i in 1:100
             csv_path = "data/model-wc-plus-stim-block/"*string(i)
 
             psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
             freq_A = xPSD_A = psd_df[!, 1]
             yPSDdat_A = psd_df[!, 2]
             yPSDs_A[i]=  yPSDdat_A
 
             bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
             xBAPDF_A = bapdf_df[!, 1]
             yBAPDFdat_A = bapdf_df[!, 2]
             yBAPDFS_A[i] = yBAPDFdat_A
 
             bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
             xBDPDF_A = bdpdf_df[!, 1]
             yBDPDFdat_A = bdpdf_df[!, 2]
             yBDPDFS_A[i] = yBDPDFdat_A
 
             plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
             xPLV_A = plvs_df[!, 1]
             yPLVdat_A = plvs_df[!, 2]
             yPLV_A[i] = yPLVdat_A
         end
 
         yPSDs_B = [[] for i in 1:100]
         freq_B = []
         yBAPDFS_B = [[] for i in 1:100]
         xBAPDF_B = []
         yBDPDFS_B = [[] for i in 1:100]
         xBDPDF_B = []
         yPLV_B = [[] for i in 1:100]
         xPLV_B = []
 
         for i in 1:100
             csv_path = "data/model/"*string(i)
 
             psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
             freq_B = xPSD_B = psd_df[!, 1]
             yPSDdat_B = psd_df[!, 2]
             yPSDs_B[i]=  yPSDdat_B
 
             bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
             xBAPDF_B = bapdf_df[!, 1]
             yBAPDFdat_B = bapdf_df[!, 2]
             yBAPDFS_B[i] = yBAPDFdat_B
 
             bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
             xBDPDF_B = bdpdf_df[!, 1]
             yBDPDFdat_B = bdpdf_df[!, 2]
             yBDPDFS_B[i] = yBDPDFdat_B
 
             plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
             xPLV_B = plvs_df[!, 1]
             yPLVdat_B = plvs_df[!, 2]
             yPLV_B[i] = yPLVdat_B
         
         end
 
         psd_stds_A = []
         for i in 1:length(yPSDs_A[1])
             y = [yPSDs_A[j][i] for j in 1:100]
             push!(psd_stds_A, std(y))
         end
 
         bapdf_stds_A = []
         for i in 1:length(yBAPDFS_A[1])
             y = [yBAPDFS_A[j][i] for j in 1:100]
             push!(bapdf_stds_A, std(y))
         end
 
         bdpdf_stds_A = []
         for i in 1:length(yBDPDFS_A[1])
             y = [yBDPDFS_A[j][i] for j in 1:100]
             push!(bdpdf_stds_A, std(y))
         end
 
         plv_stds_A = []
         for i in 1:length(yPLV_A[1])
             y = [yPLV_A[j][i] for j in 1:100]
             push!(plv_stds_A, std(y))
         end
 
         psd_stds_B = []
         for i in 1:length(yPSDs_B[1])
             y = [yPSDs_B[j][i] for j in 1:100]
             push!(psd_stds_B, std(y))
         end
 
         bapdf_stds_B = []
         for i in 1:length(yBAPDFS_B[1])
             y = [yBAPDFS_B[j][i] for j in 1:100]
             push!(bapdf_stds_B, std(y))
         end
 
         bdpdf_stds_B = []
         for i in 1:length(yBDPDFS_B[1])
             y = [yBDPDFS_B[j][i] for j in 1:100]
             push!(bdpdf_stds_B, std(y))
         end
 
         plv_stds_B = []
         for i in 1:length(yPLV_B[1])
             y = [yPLV_B[j][i] for j in 1:100]
             push!(plv_stds_B, std(y))
         end
 
         plot(
             freq_A,
             mean(yPSDs_A, dims=1), 
             ribbon=psd_stds_A,
             fillalpha=.3,
             xlabel="Frequency (Hz)",
             title="Power Spectral Density",
             label="stim",
             size=(500, 500),
             linewidth=3,
             c=2,
             xtickfont=12,
             ytickfont=12,
             titlefont=12,
             guidefont=12,
             tickfont=12,
             legendfont=12,
             margin=2.5mm
         )
         plot!(
             freq_B,
             mean(yPSDs_B, dims=1), 
             ribbon=psd_stds_B,
             fillalpha=.3,
             xlabel="Frequency (Hz)",
             title="Power Spectral Density",
             label="rest",
             size=(500, 500),
             linewidth=3,
             linestyle=:dot,
             c="black",
             xtickfont=12,
             ytickfont=12,
             titlefont=12,
             guidefont=12,
             tickfont=12,
             legendfont=12,
             margin=2.5mm
         )
         savefig("model-comp-stim-psd-rib.png")
 
         plot(
             xBAPDF_A,
             mean(yBAPDFS_A, dims=1), 
             ribbon=bapdf_stds_A,
             fillalpha=.3,
             xlabel="Amplitude",
             title="Beta Amplitude PDF",
             label="stim",
             size=(500, 500),
             linewidth=3,
             c=2,
             xtickfont=12,
             ytickfont=12,
             titlefont=12,
             guidefont=12,
             tickfont=12,
             legendfont=12,
             margin=2.5mm
         )
            plot!(
                xBAPDF_B,
                mean(yBAPDFS_B, dims=1), 
                ribbon=bapdf_stds_B,
                fillalpha=.3,
                xlabel="Amplitude",
                title="Beta Amplitude PDF",
                label="rest",
                size=(500, 500),
                linewidth=3,
                linestyle=:dot,
                c="black",
                xtickfont=12,
                ytickfont=12,
                titlefont=12,
                guidefont=12,
                tickfont=12,
                legendfont=12,
                margin=2.5mm
            )

         savefig("model-comp-stim-bapdf-rib.png")
 
         plot(
             xBDPDF_A,
             mean(yBDPDFS_A, dims=1), 
             ribbon=bdpdf_stds_A,
             fillalpha=.3,
             xlabel="Duration",
             title="Beta Duration PDF",
             label="stim",
             size=(500, 500),
             linewidth=3,
             c=2,
             xtickfont=12,
             ytickfont=12,
             titlefont=12,
             guidefont=12,
             tickfont=12,
             legendfont=12,
             margin=2.5mm
         )
            plot!(
                xBDPDF_B,
                mean(yBDPDFS_B, dims=1), 
                ribbon=bdpdf_stds_B,
                fillalpha=.3,
                xlabel="Duration",
                title="Beta Duration PDF",
                label="rest",
                size=(500, 500),
                linewidth=3,
                linestyle=:dot,
                c="black",
                xtickfont=12,
                ytickfont=12,
                titlefont=12,
                guidefont=12,
                tickfont=12,
                legendfont=12,
                margin=2.5mm
            )

         savefig("model-comp-stim-bdpdf-rib.png")

         plot(
             xPLV_A,
             mean(yPLV_A, dims=1), 
             ribbon=plv_stds_A,
             fillalpha=.3,
             xlabel="Frequency",
             title="Phase Locking Value",
             label="stim",
             size=(500, 500),
             linewidth=3,
             c=2,
             xtickfont=12,
             ytickfont=12,
             titlefont=12,
             guidefont=12,
             tickfont=12,
             legendfont=12,
             margin=2.5mm
         )
            plot!(
                xPLV_B,
                mean(yPLV_B, dims=1), 
                ribbon=plv_stds_B,
                fillalpha=.3,
                xlabel="Frequency",
                title="Phase Locking Value",
                label="rest",
                size=(500, 500),
                linewidth=3,
                linestyle=:dot,
                c="black",
                xtickfont=12,
                ytickfont=12,
                titlefont=12,
                guidefont=12,
                tickfont=12,
                legendfont=12,
                margin=2.5mm
            )
         
         savefig("model-comp-stim-plv-rib.png")
    end

    function stim_effect_phase()
        for i in 0:45:315
            csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1"
            df_psd_data = CSV.read(csv_data_path*"/psd.csv", DataFrame)
            df_beta_amp_pdf_data = CSV.read(csv_data_path*"/bapdf.csv", DataFrame)
            df_beta_dur_pdf_data = CSV.read(csv_data_path*"/bdpdf.csv", DataFrame)
            df_plvs_data = CSV.read(csv_data_path*"/plvs.csv", DataFrame)

            csv_data_path_stim = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase="*string(i)*"_STIM_EC_v1"
            df_psd_data_stim = CSV.read(csv_data_path_stim*"/psd.csv", DataFrame)
            df_beta_amp_pdf_data_stim = CSV.read(csv_data_path_stim*"/bapdf.csv", DataFrame)
            df_beta_dur_pdf_data_stim = CSV.read(csv_data_path_stim*"/bdpdf.csv", DataFrame)
            df_plvs_data_stim = CSV.read(csv_data_path_stim*"/plvs.csv", DataFrame)

            data_psd_diff = (maximum(df_psd_data[!, 2]) - maximum(df_psd_data_stim[!, 2])) / maximum(df_psd_data[!, 2])
            data_bapdf_diff = (df_beta_amp_pdf_data[!, 1][argmax(df_beta_amp_pdf_data[!, 2])] - df_beta_amp_pdf_data_stim[!, 1][argmax(df_beta_amp_pdf_data_stim[!, 2])]) / df_beta_amp_pdf_data[!, 1][argmax(df_beta_amp_pdf_data[!, 2])]
            data_bdpdf_diff = (df_beta_dur_pdf_data[!, 1][argmax(df_beta_dur_pdf_data[!, 2])] - df_beta_dur_pdf_data_stim[!, 1][argmax(df_beta_dur_pdf_data_stim[!, 2])]) / df_beta_dur_pdf_data[!, 1][argmax(df_beta_dur_pdf_data[!, 2])]
            data_plv_diff = (maximum(df_plvs_data[!, 2]) - maximum(df_plvs_data_stim[!, 2])) / maximum(df_plvs_data[!, 2])

            yPSDs_A = [[] for i in 1:25]
            freq_A = []
            yBAPDFS_A = [[] for i in 1:25]
            xBAPDF_A = []
            yBDPDFS_A = [[] for i in 1:25]
            xBDPDF_A = []
            yPLV_A = [[] for i in 1:25]
            xPLV_A = []
            
            for j in 1:25
                csv_path = "data/model-wc-plus-stim-resp-"*string(i)*"/"*string(j)
    
                psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
                freq_A = xPSD_A = psd_df[!, 1]
                yPSDdat_A = psd_df[!, 2]
                yPSDs_A[j]=  yPSDdat_A
    
                bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
                xBAPDF_A = bapdf_df[!, 1]
                yBAPDFdat_A = bapdf_df[!, 2]
                yBAPDFS_A[j] = yBAPDFdat_A
    
                bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
                xBDPDF_A = bdpdf_df[!, 1]
                yBDPDFdat_A = bdpdf_df[!, 2]
                yBDPDFS_A[j] = yBDPDFdat_A
    
                plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
                xPLV_A = plvs_df[!, 1]
                yPLVdat_A = plvs_df[!, 2]
                yPLV_A[j] = yPLVdat_A
            end

            yPSDs_B = [[] for i in 1:100]
            freq_B = []
            yBAPDFS_B = [[] for i in 1:100]
            xBAPDF_B = []
            yBDPDFS_B = [[] for i in 1:100]
            xBDPDF_B = []
            yPLV_B = [[] for i in 1:100]
            xPLV_B = []

            for j in 1:100
                csv_path = "data/model/"*string(j)
    
                psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
                freq_B = xPSD_B = psd_df[!, 1]
                yPSDdat_B = psd_df[!, 2]
                yPSDs_B[j]=  yPSDdat_B
    
                bapdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
                xBAPDF_B = bapdf_df[!, 1]
                yBAPDFdat_B = bapdf_df[!, 2]
                yBAPDFS_B[j] = yBAPDFdat_B
    
                bdpdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
                xBDPDF_B = bdpdf_df[!, 1]
                yBDPDFdat_B = bdpdf_df[!, 2]
                yBDPDFS_B[j] = yBDPDFdat_B
    
                plvs_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
                xPLV_B = plvs_df[!, 1]
                yPLVdat_B = plvs_df[!, 2]
                yPLV_B[j] = yPLVdat_B
            
            end

            model_psd_diff = (maximum(mean(yPSDs_B, dims=1)[1]) - maximum(mean(yPSDs_A, dims=1)[1])) / maximum(mean(yPSDs_B, dims=1)[1])
            model_bapdf_diff = (mean(yBAPDFS_B, dims=1)[1][argmax(mean(yBAPDFS_B, dims=1)[1])] - mean(yBAPDFS_A, dims=1)[1][argmax(mean(yBAPDFS_A, dims=1)[1])]) / mean(yBAPDFS_B, dims=1)[1][argmax(mean(yBAPDFS_B, dims=1)[1])]
            model_bdpdf_diff = (mean(yBDPDFS_B, dims=1)[1][argmax(mean(yBDPDFS_B, dims=1)[1])] - mean(yBDPDFS_A, dims=1)[1][argmax(mean(yBDPDFS_A, dims=1)[1])]) / mean(yBDPDFS_B, dims=1)[1][argmax(mean(yBDPDFS_B, dims=1)[1])]
            model_plv_diff = (maximum(mean(yPLV_B, dims=1)[1]) - maximum(mean(yPLV_A, dims=1)[1])) / maximum(mean(yPLV_B, dims=1)[1])

            println("Phase: ", i)

            println("Data PSD: ", data_psd_diff)
            println("Model PSD: ", model_psd_diff)
            println("Data BAPDF: ", data_bapdf_diff)
            println("Model BAPDF: ", model_bapdf_diff)
            println("Data BDPDF: ", data_bdpdf_diff)
            println("Model BDPDF: ", model_bdpdf_diff)
            println("Data PLV: ", data_plv_diff)
            println("Model PLV: ", model_plv_diff) 
            println()
        end
    end

    function plot_focussed_psd()
        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1"
        df_psd_data_rest = CSV.read(csv_data_path*"/psd.csv", DataFrame)   
        
        #0vs180
        p1 = plot(
            df_psd_data_rest[!, 1],
            df_psd_data_rest[!, 2], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="0 (o) vs. 180 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )

        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_STIM_EC_v1"
        df_psd_data_stim_0 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p1, df_psd_data_stim_0[!, 1], df_psd_data_stim_0[!, 2], linewidth=3)
        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1"
        df_psd_data_stim_180 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p1, df_psd_data_stim_180[!, 1], df_psd_data_stim_180[!, 2], linewidth=3)

        #45vs225
        p2 = plot(
            df_psd_data_rest[!, 1],
            df_psd_data_rest[!, 2], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="45 (o) vs. 225 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )

        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=45_STIM_EC_v1"
        df_psd_data_stim_45 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p2, df_psd_data_stim_45[!, 1], df_psd_data_stim_45[!, 2], linewidth=3)
        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=225_STIM_EC_v1"
        df_psd_data_stim_225 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p2, df_psd_data_stim_225[!, 1], df_psd_data_stim_225[!, 2], linewidth=3)

        #90vs270
        p3 = plot(
            df_psd_data_rest[!, 1],
            df_psd_data_rest[!, 2], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="90 (o) vs. 270 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )

        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=90_STIM_EC_v1"
        df_psd_data_stim_90 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p3, df_psd_data_stim_90[!, 1], df_psd_data_stim_90[!, 2], linewidth=3)
        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=270_STIM_EC_v1"
        df_psd_data_stim_270 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p3, df_psd_data_stim_270[!, 1], df_psd_data_stim_270[!, 2], linewidth=3)

        #135vs315
        p4 = plot(
            df_psd_data_rest[!, 1],
            df_psd_data_rest[!, 2], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="135 (o) vs. 315 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )

        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=135_STIM_EC_v1"
        df_psd_data_stim_135 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p4, df_psd_data_stim_135[!, 1], df_psd_data_stim_90[!, 2], linewidth=3)
        csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=270_STIM_EC_v1"
        df_psd_data_stim_315 = CSV.read(csv_data_path*"/psd.csv", DataFrame)  
        plot!(p4, df_psd_data_stim_315[!, 1], df_psd_data_stim_315[!, 2], linewidth=3)

        plot(p1, p2, p3, p4, layout=grid(2, 2))
        savefig("focussed-psd-data.png")

        yPSDs_Rest = [[] for i in 1:100]
        xPSDs_Rest = []
        for i in 1:100
            csv_path = "data/model/"*string(i)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_Rest = psd_df[!, 1]
            yPSDdat = psd_df[!, 2]
            yPSDs_Rest[i]=  yPSDdat
        end

        yPSDs_0 = [[] for i in 1:25]
        xPSDs_0 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-0/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_0 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_0[j]=  yPSDdat_A
        end

        yPSDs_180 = [[] for i in 1:25]
        xPSDs_180 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-180/"*string(j)
            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_180 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_180[j]=  yPSDdat_A
        end

        yPSDs_45 = [[] for i in 1:25]
        xPSDs_45 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-45/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_45 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_45[j]=  yPSDdat_A
        end

        yPSDs_225 = [[] for i in 1:25]
        xPSDs_225 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-225/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_225 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_225[j]=  yPSDdat_A
        end

        yPSDs_90 = [[] for i in 1:25]
        xPSDs_90 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-90/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_90 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_90[j]=  yPSDdat_A
        end

        yPSDs_270 = [[] for i in 1:25]
        xPSDs_270 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-270/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_270 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_270[j]=  yPSDdat_A
        end

        yPSDs_135 = [[] for i in 1:25]
        xPSDs_135 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-135/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_135 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_135[j]=  yPSDdat_A
        end

        yPSDs_315 = [[] for i in 1:25]
        xPSDs_315 = []
        for j in 1:25
            csv_path = "data/model-wc-plus-stim-resp-315/"*string(j)

            psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
            xPSDs_315 = psd_df[!, 1]
            yPSDdat_A = psd_df[!, 2]
            yPSDs_315[j]=  yPSDdat_A
        end

        #0vs180
        p1 = plot(
            xPSDs_Rest,
            mean(yPSDs_Rest, dims=1),
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="0 (o) vs. 180 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p1, xPSDs_0, mean(yPSDs_0, dims=1), linewidth=3)
        plot!(p1, xPSDs_180, mean(yPSDs_180, dims=1), linewidth=3)
        
        #45vs225
        p2 = plot(
            xPSDs_Rest,
            mean(yPSDs_Rest, dims=1),
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="45 (o) vs. 225 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p2, xPSDs_45, mean(yPSDs_45, dims=1), linewidth=3)
        plot!(p2, xPSDs_225, mean(yPSDs_225, dims=1), linewidth=3)

        #90vs270
        p3 = plot(
            xPSDs_Rest,
            mean(yPSDs_Rest, dims=1),
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="90 (o) vs. 270 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p3, xPSDs_90, mean(yPSDs_90, dims=1), linewidth=3)
        plot!(p3, xPSDs_270, mean(yPSDs_270, dims=1), linewidth=3)

        #135vs315
        p4 = plot(
            xPSDs_Rest,
            mean(yPSDs_Rest, dims=1),
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="135 (o) vs. 315 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=3,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p4, xPSDs_135, mean(yPSDs_135, dims=1), linewidth=3)
        plot!(p4, xPSDs_315, mean(yPSDs_315, dims=1), linewidth=3)

        plot(p1, p2, p3, p4, layout=grid(2, 2))
        savefig("focussed-psd-model.png")

        ##SUBTRACT

        #0vs180
        p1 = plot(
            df_psd_data_rest[!, 1],
            [0.0 for i in df_psd_data_rest[!, 2]], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="0 (o) vs. 180 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p1, df_psd_data_stim_0[!, 1], df_psd_data_stim_0[!, 2] - df_psd_data_rest[!, 2], linewidth=3)
        plot!(p1, df_psd_data_stim_180[!, 1], df_psd_data_stim_180[!, 2] - df_psd_data_rest[!, 2], linewidth=3)

        #45vs225
        p2 = plot(
            df_psd_data_rest[!, 1],
            [0.0 for i in df_psd_data_rest[!, 2]], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="45 (o) vs. 225 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p2, df_psd_data_stim_45[!, 1], df_psd_data_stim_45[!, 2] - df_psd_data_rest[!, 2], linewidth=3)
        plot!(p2, df_psd_data_stim_225[!, 1], df_psd_data_stim_225[!, 2] - df_psd_data_rest[!, 2], linewidth=3)

        #90vs270
        p3 = plot(
            df_psd_data_rest[!, 1],
            [0.0 for i in df_psd_data_rest[!, 2]], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="90 (o) vs. 270 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p3, df_psd_data_stim_90[!, 1], df_psd_data_stim_90[!, 2] - df_psd_data_rest[!, 2], linewidth=3)
        plot!(p3, df_psd_data_stim_270[!, 1], df_psd_data_stim_270[!, 2] - df_psd_data_rest[!, 2], linewidth=3)

        #135vs315
        p4 = plot(
            df_psd_data_rest[!, 1],
            [0.0 for i in df_psd_data_rest[!, 2]], 
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="135 (o) vs. 315 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p4, df_psd_data_stim_135[!, 1], df_psd_data_stim_135[!, 2] - df_psd_data_rest[!, 2], linewidth=3)
        plot!(p4, df_psd_data_stim_315[!, 1], df_psd_data_stim_315[!, 2] - df_psd_data_rest[!, 2], linewidth=3)

        plot(p1, p2, p3, p4, layout=grid(2, 2))
        savefig("focussed-psd-diff.png")

        #SUBTRACT MODEL
        p1 = plot(
            xPSDs_Rest,
            [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="0 (o) vs. 180 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p1, xPSDs_0, mean(yPSDs_0, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)
        plot!(p1, xPSDs_180, mean(yPSDs_180, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)

        p2 = plot(
            xPSDs_Rest,
            [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="45 (o) vs. 225 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p2, xPSDs_45, mean(yPSDs_45, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)
        plot!(p2, xPSDs_225, mean(yPSDs_225, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)

        p3 = plot(
            xPSDs_Rest,
            [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="90 (o) vs. 270 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p3, xPSDs_90, mean(yPSDs_90, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)
        plot!(p3, xPSDs_270, mean(yPSDs_270, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)

        p4 = plot(
            xPSDs_Rest,
            [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
            xlabel="Frequency (Hz)",
            ylabel="Power Change",
            title="135 (o) vs. 315 (g)",
            legend=false,
            size=(600, 500),
            xlim=(6, 16),
            xticks=6:2:16,
            linewidth=1.5,
            linestyle=:dash,
            c="black",
            xtickfont=12,
            ytickfont=12,
            titlefont=12,
            guidefont=12,
            tickfont=12,
            legendfont=12,
            margin=2.5mm
        )
        plot!(p4, xPSDs_135, mean(yPSDs_135, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)
        plot!(p4, xPSDs_315, mean(yPSDs_315, dims=1)[1] - mean(yPSDs_Rest, dims=1)[1], linewidth=3)

        plot(p1, p2, p3, p4, layout=grid(2, 2))
        savefig("focussed-psd-diff-model.png")

        
    end


end

#analysis.plot_focussed_psd()

#analysis.analyse_all_flat()
#analysis.get_raw("P20", "15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1", 2001, 3000)
#analysis.get_beta_burst_process("P20", "15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1")
#analysis.plot_example_bursts()

#analysis.plot_feature_ribbons("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1")
analysis.plot_feature_costs("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1")
#analysis.plot_feature_ribbons_stim_pair("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1")
#analysis.plot_feature_costs_stim_pair("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1")
#analysis.plot_data_features_data_stimeffect("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1", "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1")
#analysis.plot_model_features_stim_effect()
#analysis.plot_feature_costs_stim_pair("data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1")

#analysis.stim_effect_phase()