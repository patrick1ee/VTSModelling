module Stimulation

    using PyCall, Plots

    pushfirst!(PyVector(pyimport("sys")."path"), "./")
    @pyimport Receptors

    export create_stimulus, create_stim_response, yousif_transfer, create_stim_block

    function create_stimulus(A, f, range_t)
        Lt = length(range_t)
        R = zeros(Lt)
        for i in 1:Lt
            R[i] = A * sin(f * 2 * pi * range_t[i])
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

    function create_stim_block(freq, ontime, offtime, range_t) #s
        Lt = length(range_t)
        stim = zeros(Lt)
        T = 1000.0 / freq
        rs = 0
        for i in 1:Lt
            if i % (ontime + offtime) < ontime
                if rs == 0
                    rs = i
                end
                stim[i] = (i - rs) % T == 0 ? 1.0 : 0.0
            else
                rs = 0
                stim[i] = 0.0
            end
        end
        return stim
    end

    function test_stim()
        range_t = 0:0.001:1.0
        S = create_stim_block(100.0, 250 ,250, range_t) #ms
        R = create_stim_response(S, range_t)
        p1 = plot(range_t, S, title="Stimulation", xlabel="Time (ms)", ylabel="Stimulus")
        p2 = plot(range_t, R, title="Stimulation", xlabel="Time (ms)", ylabel="Response")

        plot(p1, p2, layout=(2,1))
        savefig("./stimulation.png")   
    end

end

Stimulation.test_stim()