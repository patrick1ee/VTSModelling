module Signal

    using DSP, FFTW, KernelDensity

    export get_pow_spec, get_hilbert_amplitude_pdf

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
        
        # Estimate PDF using kernel density estimation
        U = kde(hilbert_amp, bandwidth=bandwidth)
        
        return U.x, U.density, hilbert_amp
    end
end