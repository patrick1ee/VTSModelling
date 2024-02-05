module RafalModel

export RafalModel, create_rafal_model, simulate_rafal_model

    struct Network
        tau_E::Float32
        tau_I::Float32
        w_EE::Float32
        w_EI::Float32
        w_IE::Float32
        beta::Float32
    end
    

    function create_rafal_model(tau_E::Float32, tau_I::Float32, w_EE::Float32, w_EI::Float32, w_IE::Float32, beta::Float32)
        return Network(tau_E, tau_I, w_EE, w_EI, w_IE, beta)
    end

    function sigmoid(x, beta)
        return 1.0 / (1.0 + exp(-beta * (x - 1)))
    end

    function simulate_rafal_model(n::Network, range_t, dt, theta_E, theta_I)
        # Initial conditions
        Lt = length(range_t)
        rE = zeros(Lt)
        rI = zeros(Lt)

        # Simulate the model
        for i in 2:Lt
            drE = (dt / n.tau_E) * (-rE[i-1] + sigmoid(theta_E[i-1] + n.w_EE * rE[i-1] - n.w_IE * rI[i-1], n.beta))
            drI = (dt / n.tau_I) * (-rI[i-1] + sigmoid(theta_I[i-1] + n.w_EI * rE[i-1], n.beta))

            rE[i] = rE[i-1] + drE
            rI[i] = rI[i-1] + drI
        end
        return rE, rI
    end
end