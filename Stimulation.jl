module Stimulation

    using PyCall, Plots

    pushfirst!(PyVector(pyimport("sys")."path"), "./")
    @pyimport Receptors

    export create_stimulus, create_stim_response, yousif_transfer, create_stim_block

    function create_stimulus_sin(A, f, range_t, padding)
        Lt = length(range_t)
        R = [-1.0 for i in 1:Lt]
        for i in padding:Lt
            R[i] = A * sin(f * 2 * pi * range_t[i])
        end
        return R
    end

    function create_stimulus_square(A, f, range_t, padding)
        Lt = length(range_t)
        R = [-1.0 for i in 1:Lt]
        T = 1000.0 / f
        for i in padding:Lt
            R[i] = A * (i % T < T / 2 ? 1.0 : -1.0)
        end
        return R
    end

    function yousif_transfer(A, f, range_t)
        N=1001
        Lt = length(range_t)
        R = zeros(length(range_t))
        for i in 1:Lt
            sum = 0.0
            for j in 1:2:N
                sum += (1.0 / j) * sin(f * 2 * j * pi * range_t[i])
            end
            R[i] = A * (4 / pi) * sum
        end
        return R
    end

    function create_stim_response(stim, range_t)
        response = 3.5*Receptors.get_response(stim, range_t)
        return response
    end

    function create_stim_block(freq, ontime, offtime, range_t, delay) #ms
        Lt = length(range_t)
        stim = zeros(Lt)
        T = 1000.0 / freq
        rs = 0
        for i in delay:Lt
            if (i - delay) % (ontime + offtime) < ontime
                if rs == 0
                    rs = i
                end
                stim[i] = (i - rs) % T == 0 ? 1.0e-3 : 0.0
            else
                rs = 0
                stim[i] = 0.0
            end
        end
        return stim
    end

    function test_stim()
        range_t = 0:0.001:0.25

        S = create_stim_block(200.0, 25,12.5, range_t, 1) #ms
        #S = (create_stimulus_sin(1.0, 30.0, range_t, 50) .+ 1.0) .* 1.0e-3
        #S = zeros(length(range_t))
        R = create_stim_response(S, range_t)
        p1 = plot(
            1:length(range_t),
            S,
            title="Stimulation",
            xlabel="Time (ms)",
            xlim=(0, 200),
            legend=false,
            bottom_margin=10Plots.mm,
            right_margin=10Plots.mm,
            linewidth=2,
            xtickfont=18,
            ytickfont=18,
            titlefont=18,
            guidefont=18,
            tickfont=18,
            size=(1000, 300)
        )
        p2 = plot(
            1:length(range_t),
            R,
            title="Response",
            xlabel="Time (ms)",
            ylabel="Voltage (mV)",
            xlim=(0, 200),
            legend=false,
            bottom_margin=10Plots.mm,
            right_margin=10Plots.mm,
            linewidth=2,
            xtickfont=18,
            ytickfont=18,
            titlefont=18,
            guidefont=18,
            tickfont=18,
            size=(1000, 300),
        )

        plot(p1, p2, layout=(2,1), size=(1000, 700))
        savefig("./plots/diss/aff-resp-train-2.png")   
    end

end

Stimulation.test_stim()