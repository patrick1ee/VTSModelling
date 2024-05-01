module Oscilltrack

    using Plots

    export Oscilltracker, update!, get_phase

    mutable struct Oscilltracker
        a::Float64
        b::Float64
        re::Float64
        im::Float64
        theta::Float64
        sinv::Float64
        cosv::Float64
        
        phase_target::Float64
        w::Float64
        gamma::Float64
        suppression_reset::Int
        suppression_count::Int
        is_prev_above_thrs::Bool
        
        function Oscilltracker(freq_target::Float64, phase_target::Float64, freq_sample::Float64, suppression_cycle::Float64, gamma::Union{Float64, Nothing}=nothing)
            phase_target = phase_target
            w = 2 * π * freq_target / freq_sample
            gamma = isnothing(gamma) ? 125 / freq_sample : gamma
            suppression_reset = round(Int, suppression_cycle * freq_sample / freq_target)
            suppression_count = 0
            is_prev_above_thrs = false
            new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, phase_target, w, gamma, suppression_reset, suppression_count, is_prev_above_thrs)
        end
    end

    function update!(osc::Oscilltracker, data::Float64)
        Delta = pred_error(osc, data)
        osc.a += osc.gamma * Delta * osc.sinv
        osc.b += osc.gamma * Delta * osc.cosv

        osc.theta += osc.w
        # Wrap theta in the range [-pi, pi] for numerical stability
        if osc.theta >= π
        osc.theta -= 2 * π
        end

        osc.sinv = sin(osc.theta)
        osc.cosv = cos(osc.theta)
        osc.re = osc.a * osc.sinv + osc.b * osc.cosv
        osc.im = osc.b * osc.sinv - osc.a * osc.cosv 
    end

    function pred_error(osc::Oscilltracker, data::Float64)
        # Calculates the error between the predicted signal value and the actual data at a timestep.
        # Used internally in the update function to update self.a and self.b .
        # This is a separate function for debug purposes.
        return data - osc.re
    end

    function get_phase(osc::Oscilltracker)
        return atan(osc.im, osc.re)
    end

    function get_phase_target(osc::Oscilltracker)
        return osc.phase_target
    end

    function decide_stim(osc::Oscilltracker)
        phase = get_phase(osc)
        phase_rotated = phase - osc.phase_target
        if phase_rotated >= π
            phase_rotated -= 2 * π
        elseif phase_rotated < -π
            phase_rotated += 2 * π
        end

        is_stim = false
        is_above_thrs = phase_rotated >= 0
        
        if is_above_thrs && (!osc.is_prev_above_thrs) && (phase_rotated < π/2)
            is_stim = osc.suppression_count == 0
            osc.suppression_count = osc.suppression_reset
        end

        osc.is_prev_above_thrs = is_above_thrs
        if osc.suppression_count > 0
            osc.suppression_count -= 1
        else
            osc.suppression_count = 0
        end
        
        return is_stim
    end
    
    function test_oscilltrack()
        SR = 750.0
        gamma_param = 0.1 # or 0.05
        OT_suppress = 0.3
        target_phase = pi / 4.0
        target_freq = 10.0

        A = 1.0
        f = 10.0
        phase = 0.0
        base = 0.0

        range_t = 0:0.001:1.0
        Lt = length(range_t)
        S = zeros(Lt)
        for i in 1:Lt
            S[i] = A * sin(f * 2 * pi * range_t[i] + phase) + base
        end

        osc = Oscilltracker(target_freq, target_phase, SR, OT_suppress, gamma_param)

        phase = zeros(Lt)
        stim = zeros(Lt)
        for i in 1:Lt
            update!(osc, S[i])
            phase[i] = get_phase(osc)
            stim[i] = decide_stim(osc)
        end
        plot(range_t, [S, phase, stim])
        savefig("test_oscilltrack.png")
    end
end

Oscilltrack.test_oscilltrack()