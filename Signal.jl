module Signal

    using DSP, FFTW, KernelDensity

    export get_pow_spec, get_hilbert_amplitude_pdf

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

    function get_pow_spec(sig, freqs=Nothing, sampling_rate=1000)
        if freqs == Nothing 
            freqs = fftshift(fftfreq(length(sig), sampling_rate))
        end
        #spec = fftshift(fft(sig .- mean(sig)))
        spec = welch_pgram(sig; fs=sampling_rate)

        #spec_fit = curve_fit(model, freqs, abs.(spec), zeros(5))
        #fit_fun = Fun(Chebyshev(Interval(0,50)), spec_fit.param)
        #spec_welch = power(welch_pgram(sig), n=length(freqs))
        return spec.freq, spec.power
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
end