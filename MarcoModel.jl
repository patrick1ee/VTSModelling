module marcoModel

    using DiffEqNoiseProcess, JSON, Plots

    export MarcoModel, create_marco_model, simulate_marco_model

    struct Network
        tau_E::Float32
        tau_I::Float32
        w_EE::Float32
        w_EI::Float32
        w_IE::Float32
        beta::Float32
        WE
        WI
    end
    

    function create_marco_model(tau_E::Float32, tau_I::Float32, w_EE::Float32, w_EI::Float32, w_IE::Float32, beta::Float32)
        WE = WienerProcess(0.0, 1.0, 1.0)
        WI = WienerProcess(0.0, 1.0, 1.0)
        return Network(tau_E, tau_I, w_EE, w_EI, w_IE, beta, WE, WI)
    end

    function sigmoid(x, beta)
        return 1.0 / (1.0 + exp(-beta * (x - 1)))
    end

    function simulate_marco_model(n::Network, range_t, dt, theta_E, theta_I)
        # Initial conditions
        Lt = length(range_t)
        rE = zeros(Lt)
        rI = zeros(Lt)

        rWE = zeros(Lt)
        rWI = zeros(Lt)
        n.WE.dt = dt
        n.WI.dt = dt
        uWE = nothing;
        pWE = nothing; # for state-dependent distributions
        uWI = nothing;
        pWI = nothing;
        calculate_step!(n.WE, dt, uWE, pWE)
        calculate_step!(n.WI, dt, uWI, pWI)

        # Simulate the model
        for i in 2:Lt
            drE = (dt / n.tau_E) * (-rE[i-1] + sigmoid(theta_E[i-1] + n.w_EE * rE[i-1] - n.w_IE * rI[i-1], n.beta))
            drI = (dt / n.tau_I) * (-rI[i-1] + sigmoid(theta_I[i-1] + n.w_EI * rE[i-1], n.beta))

            accept_step!(n.WE, dt, uWE, pWE)
            accept_step!(n.WI, dt, uWI, pWI)

            rWE[i] = n.WE[i]
            rWI[i] = n.WI[i]
  
            rE[i] = rE[i-1] + drE + n.WE[i] - n.WE[i - 1]
            rI[i] = rI[i-1] + drI + n.WI[i] - n.WI[i - 1]
        end
        return rE, rI
    end
end