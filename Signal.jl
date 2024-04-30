module Signal

    include("./Oscilltrack.jl")
    using .Oscilltrack: Oscilltracker, update!, get_phase
    using DSP, FFTW, KernelDensity, KissSmoothing, StatsBase, Statistics

    export get_bandpassed_signal, get_beta_data, get_pow_spec, get_hilbert_amplitude_pdf, get_burst_durations, get_signal_phase, get_plv_freq

    function interpolate_nan(arr)
        Larr = length(arr)
        for i in 1:Larr
            if isnan(arr[i])
                j = i
                while isnan(arr[j]) 
                    j += 1
                    if j > Larr
                        j -= 1
                        arr[j] = 0.0
                        break
                    end
                end
                if i == 1
                    arr[i:j] .= arr[j]
                else
                    arr[i:j-1] .= arr[i-1] .+ (1:j-i) .* (arr[j] .- arr[i-1]) ./ (j-i)
                end
            end
        end
        
        return arr
    end

    function get_bounds(arr, lb, ub)
        li = 1
        while arr[li] < lb
            li += 1
        end
        ui = 1
        while arr[ui] < ub
            ui += 1
        end
        return li, ui
    end

    function get_bandpassed_signal(signal, low, high)
        SR = 1000
        FLT_ORDER = 2
        responsetype = Bandpass(low, high; fs=SR)
        designmethod = Butterworth(FLT_ORDER)
        data_flt = filt(digitalfilter(responsetype, designmethod), signal)
        return data_flt
    end

    function get_beta_data(sig, width=6.0)
        FLT_ORDER = 2
        LOW_FREQ = 13.0
        HIGH_FREQ = 35.0
        SR = 1000

        freq, power = get_pow_spec(sig)
        freqSize = length(freq)
        peak_freq = 0.0
        peak_spec = 0.0
        for i in 1:freqSize
            if freq[i] >= LOW_FREQ && freq[i] <= HIGH_FREQ && power[i] > peak_spec
                peak_freq = freq[i]
                peak_spec = power[i]
            end
        end

        data_flt_beta = get_bandpassed_signal(sig, peak_freq - (width / 2.0), peak_freq + (width / 2.0))

        return data_flt_beta
    end

    function get_pow_spec(sig, freqs=Nothing, sampling_rate=1000)
        if freqs == Nothing 
            freqs = fftshift(fftfreq(length(sig), sampling_rate))
        end
        #spec = fftshift(fft(sig))
        spec = welch_pgram(sig, fs=sampling_rate)

        #spec_fit = curve_fit(model, freqs, abs.(spec), zeros(5))
        #fit_fun = Fun(Chebyshev(Interval(0,50)), spec_fit.param)
        #spec_welch = power(welch_pgram(sig), n=length(freqs))

        # Filter out frequencies between 0 and 50
        li, ui = get_bounds(spec.freq, 6, 55)
        filtered_freqs = spec.freq[li:ui]
        filtered_spec = spec.power[li:ui]

        S, N = denoise(convert(AbstractArray{Float64}, abs.(filtered_spec)))
        #S = filtered_spec
        S = S .* 10
        return filtered_freqs, S
    end

    function get_hilbert_amplitude_pdf(signal; bandwidth=0.1)
        hilbert_transform = hilbert(signal)
        hilbert_amp = abs.(hilbert_transform)

        hilbert_amp = interpolate_nan(hilbert_amp)
        
         # Estimate PDF using kernel density estimation
        try
            U = kde(hilbert_amp)
            return U.x, U.density, hilbert_amp
        catch err
            println("Error: ", err)
            println("Data: ", hilbert_amp)
            return [], [], []
        end
    end

    function get_burst_durations(signal, threshold=0.5)
        burst_durations = []
        burst_start = 0
        burst_end = 0
        burst_duration = 0 #ms
        burst = false
        signalSize = length(signal)
        for i in 1:signalSize
            if signal[i] >= threshold
                if !burst
                    burst_start = i
                    burst = true
                else
                    burst_duration += 1
                end
            else
                if burst && burst_duration >= 100
                    burst_end = i
                    push!(burst_durations, burst_duration)
                    burst = false
                end
                burst_duration = 0
            end
        end

        burst_durations = burst_durations ./ 1000

        try
            U = kde(burst_durations)
            return U.x, U.density, burst_duration
        catch err
            println("Error: ", err)
            println("Data: ", burst_duration)
            return [], [], []
        end
    end

    function get_signal_phase(signal)
        SR = 1000.0
        gamma_param = 0.1 # or 0.05
        OT_suppress = 0.3
        target_phase = 0.0
        target_freq = 10.0
        phase_search = false

        Lt = length(signal)
        osc = Oscilltracker(target_freq, [target_phase], SR, OT_suppress, gamma_param, phase_search)
        phase = zeros(Lt)
        for i in 1:Lt
            update!(osc, Float64(signal[i]))
            phase[i] = get_phase(osc)
        end

        return phase
    end

    function get_plv_freq(s1, s2)
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
        return plvs
    end

end