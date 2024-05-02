module analysis

    include("./Signal.jl")

    using ApproxFun, Base.Filesystem, CSV, DataFrames, DSP, FFTW, HDF5, Interpolations, KissSmoothing, LPVSpectral, LsqFit, Measures, NaNStatistics, Plots, StatsBase, Statistics

    using .Signal: get_bandpassed_signal, get_beta_data, get_pow_spec, get_hilbert_amplitude_pdf, get_burst_durations, get_signal_phase

    const SR = 1000  # recording sampling rate in Hz, do not change this

    export run_spec, run_hilbert_pdf, run_beta_burst, run_plv

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

        csv_df = DataFrame(x=x,y=y)
        CSV.write(csv_path*"/bapdf.csv", csv_df)

        plot(x, y, xlabel="amplitude", ylim=(0.0, 2.0), xlim=(0, 6), xticks=0:2:6, yticks=0:0.5:2.0, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig(plot_path*"/bapdf.png")

        csv_df = DataFrame(x=1:length(signal),y=S)
        CSV.write(csv_path*"/bamp.csv", csv_df)

        plot(1:length(signal), S, xlabel="time", size=(500,500), xlim=(0, 10000), xticks=0:2000:10000, linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig(plot_path*"/bamp.png")

        bx, by, burst_durations = get_burst_durations(S)

        csv_df = DataFrame(x=bx,y=by)
        CSV.write(csv_path*"/bdpdf.csv", csv_df)

        plot(bx, by, xlabel="duration", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig(plot_path*"/bdpdf.png")
    end

    function run_plv(s1, s2, plot_path, csv_path)
        plvs = []
        freqs = 6:29
        for f in freqs
            s1f = get_bandpassed_signal(s1, f-0.5, f+0.5)
            s2f = get_bandpassed_signal(s2, f-0.5, f+0.5)
            p1f = get_signal_phase(s1f)
            p2f = get_signal_phase(s2f)
            plv = abs(mean(exp.(1im*(p1f .- p2f))))
            push!(plvs, plv)
        end

        p1 = get_signal_phase(s1)
        p2 = get_signal_phase(s2)

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

        plot(freqs, plvs)
        savefig(plot_path*"/plvs.png")
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
            bx, by, burst_durations = get_burst_durations(S)

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
                    
                    run_spec(data_raw, plot_path, csv_path)
                    run_beta_burst(data_flt_beta, plot_path, csv_path)
                    run_plv(data_raw, data_raw_alt, plot_path, csv_path)

                    plot(1:length(data_raw), data_raw, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
                    savefig(plot_path*"/raw.png")
                    plot(1:length(data_flt_beta), data_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
                    savefig(plot_path*"/flt-beta.png")

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
end

#analysis.analyse_all_flat()