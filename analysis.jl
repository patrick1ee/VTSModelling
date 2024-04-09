using ApproxFun, CSV, DataFrames, DSP, FFTW, HDF5, Interpolations, LsqFit, NaNStatistics, Plots, StatsBase, Statistics

const SR = 1000  # recording sampling rate in Hz, do not change this

function interpolate_nan(arr)
    indices = findall(isnan, arr)  # Find indices of NaN values
    non_nan_indices = findall(!isnan, arr)  # Find indices of non-NaN values
    
    # Interpolate NaN values using linear interpolation
    interp = interpolate(non_nan_indices, arr[non_nan_indices], Gridded(Linear()))
    
    # Replace NaN values with interpolated values
    arr[indices] .= map(x -> round(Int, x), evaluate(interp, indices))
    
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

function plot_spec(sig, freqs=Nothing, sampling_rate=1000)
    if freqs == Nothing 
        freqs = fftshift(fftfreq(length(sig), sampling_rate))
    end
    sig = sig .* 1e6
    spec = fftshift(fft(sig .- mean(sig)))
    print(sig)
    print(spec)
    print(freqs)
    #spec_fit = curve_fit(model, freqs, abs.(spec), zeros(5))
    #fit_fun = Fun(Chebyshev(Interval(0,50)), spec_fit.param)
    #spec_welch = power(welch_pgram(sig), n=length(freqs))
    plot(freqs, abs.(spec), xlabel="frequency (Hz)", xlim=(0, +60), xticks=0:10:60, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
    
    savefig("plots/ml_plots/spec.png")
end

function analyse()
    data_path = "Patrick_data"
    chan_order = CSV.read("matlab_analyses/EEG_channel_order.csv", DataFrame)

    # Define the size of the time-window for computing the ERP (event-related potential)
    ERP_tWidth = 1  # [sec]
    ERP_tOffset = ERP_tWidth / 2  # [sec]
    ERP_LOWPASS_FILTER = 30  # set to nan if you want to deactivate it

    POW_SPECTRUM_FREQS = 6:55  # in Hz

    # Do not change these parameters
    FLT_ORDER = 2

    currSubj = "P20"
    WIN_START = 5  # sec
    WIN_END = 75  # sec

    fid = h5open(data_path*"/"*currSubj*"/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1.hdf5", "r")
    data = read(fid["EEG"])
    close(fid)

    CONST_REF_CHAN = "C3_local"
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
    #data_outlierRem = interpolate_nan(data_outlierRem)

    responsetype = Lowpass(ERP_LOWPASS_FILTER; fs=SR)
    designmethod = Butterworth(FLT_ORDER)

    #data_flt = filt(digitalfilter(responsetype, designmethod), EEG_data)
    data_raw = data_outlierRem
    plot_spec(data_raw)

end

analyse()