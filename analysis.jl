module analysis

    include("./Signal.jl")

    using ApproxFun, CSV, DataFrames, DSP, FFTW, HDF5, Interpolations, KissSmoothing, LPVSpectral, LsqFit, NaNStatistics, Plots, StatsBase, Statistics

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

    function run_spec(sig, model; freqs=Nothing, sampling_rate=1000)
        freq, power = get_pow_spec(sig, freqs, sampling_rate)
        plot(freq, power, xlabel="frequency (Hz)", xlim=(6, 55), xticks=6:4:55, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)

        csv_df = DataFrame(Frequency = freq, PSD = abs.(power))

        if model
            savefig("plots/optim/model/psd.png")
            CSV.write("data/psd-m.csv", csv_df)
        else 
            savefig("plots/optim/data/psd.png")
            CSV.write("data/psd.csv", csv_df)
        end
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

    function run_beta_burst(signal, model, bandwidth=0.1)
        x, y, ha = get_hilbert_amplitude_pdf(signal, bandwidth=bandwidth)
        S, N = denoise(convert(AbstractArray{Float64}, ha))

        plot(x, y, xlabel="amplitude", ylim=(0.0, 2.0), xlim=(0, 6), xticks=0:2:6, yticks=0:0.5:2.0, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        csv_df = DataFrame(x=x,y=y)
        
        if model
            savefig("plots/optim/model/beta-hpdf.png")
            CSV.write("data/beta-hpdf-m.csv", csv_df)
        else 
            savefig("plots/optim/data/beta-hpdf.png")
            CSV.write("data/beta-hpdf.csv", csv_df)
        end

        plot(1:length(signal), S, xlabel="time", size=(500,500), xlim=(0, 10000), xticks=0:2000:1000, linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        csv_df = DataFrame(x=1:length(signal),y=S)
        
        if model
            savefig("plots/optim/model/beta-hamp.png")
            CSV.write("data/beta-hamp-m.csv", csv_df)
        else 
            savefig("plots/optim/data/beta-hamp.png")
            CSV.write("data/beta-hamp.csv", csv_df)
        end

        bx, by, burst_durations = get_burst_durations(S)
        plot(bx, by, xlabel="duration", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        csv_df = DataFrame(x=bx,y=by)
        
        if model
            savefig("plots/optim/model/beta-dur-pdf.png")
            CSV.write("data/beta-dur-pdf-m.csv", csv_df)
        else 
            savefig("plots/optim/data/beta-dur-pdf.png")
            CSV.write("data/beta-dur-hpdf.csv", csv_df)
        end
    end

    function run_plv(s1, s2, model)
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
        
        plot(1:length(s1), p1)
        csv_df = DataFrame(x=1:length(s1),y=p1)
        if model
            savefig("plots/optim/model/phase-1.png")
            CSV.write("data/phase-1-m.csv", csv_df)
        else 
            savefig("plots/optim/data/phase-1.png")
            CSV.write("data/phase-1.csv", csv_df)
        end

        plot(1:length(s2), p2)
        csv_df = DataFrame(x=1:length(s1),y=p2)
        if model
            savefig("plots/optim/model/phase-2.png")
            CSV.write("data/phase-1-m.csv", csv_df)
        else 
            savefig("plots/optim/data/phase-2.png")
            CSV.write("data/phase-1.csv", csv_df)
        end

        plot(freqs, plvs)
        csv_df = DataFrame(Frequency = freqs, PLV = plvs)
        if model
            savefig("plots/optim/model/plvs.png")
            CSV.write("data/plvs-m.csv", csv_df)
        else 
            savefig("plots/optim/data/plvs.png")
            CSV.write("data/plvs.csv", csv_df)
        end

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

    function analyse()
        data_path = "Patrick_data"

        # Define the size of the time-window for computing the ERP (event-related potential)
        ERP_tWidth = 1  # [sec]
        ERP_tOffset = ERP_tWidth / 2  # [sec]
        ERP_LOWPASS_FILTER = 30  # set to nan if you want to deactivate it

        POW_SPECTRUM_FREQS = 6:55  # in Hz

        currSubj = "P20"

        fid = h5open(data_path*"/"*currSubj*"/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1.hdf5", "r")
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
        
        run_spec(data_raw, false)
        run_hilbert_pdf(data_raw, false)

        run_beta_burst(data_flt_beta, false)

        #plot(1:100, data_raw[1:1:100], xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        plot(1:length(data_raw), data_raw, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig("plots/optim/data/raw.png")

        plot(1:length(data_flt_beta), data_flt_beta, xlabel="time (s)", ylabel="amplitude", size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig("plots/optim/data/flt_beta.png")

        run_plv(data_raw, data_raw_alt, false)
    end
end

#analysis.analyse()