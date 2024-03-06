module Stimulation

    using PyCall

    pushfirst!(PyVector(pyimport("sys")."path"), "./")
    @pyimport Receptors

    export create_stimulus, create_stim_response, yousif_transfer

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

end