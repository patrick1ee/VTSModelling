module BenoitModel

    using DiffEqNoiseProcess, JSON, Plots

    export BenoitModel, create_benoit_model, simulate_benoit_model

    struct Node
        tau_E::Float32
        tau_I::Float32
        w_EE::Float32
        w_EI::Float32
        w_IE::Float32
        beta::Float32
        WE
        WI
    end

    struct Network
        nodes::Array{Node, 1}
        W::Matrix{Float32}
        etta::Float32
    end

    struct NodeActivity
        rE::Array{Float32, 1}
        rI::Array{Float32, 1}
    end
    
    NOISE_DEV = 0.4

    function create_benoit_model(N::Int64, W::Matrix{Float32}, etta::Float32, tau_E::Float32, tau_I::Float32, w_EE::Float32, w_EI::Float32, w_IE::Float32, beta::Float32)
        nodes = [Node(tau_E, tau_I, w_EE, w_EI, w_IE, beta, WienerProcess(0.0, 1.0, 1.0), WienerProcess(0.0, 1.0, 1.0)) for _ in 1:N]
        return Network(nodes, W, etta)
    end

    function sigmoid(x, beta)
        return 1.0 / (1.0 + exp(-beta * (x - 1)))
    end

    function simulate_benoit_model(N::Network, range_t, dt, theta_E, theta_I)
        # Initial conditions
        Lt = length(range_t)
        R = [NodeActivity(zeros(Lt), zeros(Lt)) for _ in N.nodes]

        rWE = [zeros(Lt) for _ in N.nodes]
        rWI = [zeros(Lt) for _ in N.nodes]

        uWE = [nothing for _ in N.nodes];
        pWE = [nothing for _ in N.nodes]; # for state-dependent distributions
        uWI = [nothing for _ in N.nodes];
        pWI = [nothing for _ in N.nodes];
        
        for (j, n) in enumerate(N.nodes)
            n.WE.dt = dt
            n.WI.dt = dt
            calculate_step!(n.WE, dt, uWE[j], pWE[j])
            calculate_step!(n.WI, dt, uWI[j], pWI[j])
        end

        # Simulate the model
        for i in 2:Lt
            for (j, n) in enumerate(N.nodes)

                C = 0.0
                for (k, _) in enumerate(N.nodes)
                    C += N.W[k,j] * R[k].rE[i-1]
                end
                C = N.etta * C / length(N.nodes)

                drE = (dt / n.tau_E) * (-R[j].rE[i-1] + sigmoid(theta_E[j][i-1] + n.w_EE * R[j].rE[i-1] - n.w_IE * R[j].rI[i-1], n.beta))
                drI = (dt / n.tau_I) * (-R[j].rI[i-1] + sigmoid(theta_I[j][i-1] + n.w_EI * R[j].rE[i-1], n.beta))

                accept_step!(n.WE, dt, uWE[j], pWE[j])
                accept_step!(n.WI, dt, uWI[j], pWI[j])

                rWE[j][i] = n.WE[i]
                rWI[j][i] = n.WI[i]
    
                R[j].rE[i] = R[j].rE[i-1] + drE + NOISE_DEV * (n.WE[i] - n.WE[i - 1])
                R[j].rI[i] = R[j].rI[i-1] + drI + NOISE_DEV * (n.WI[i] - n.WI[i - 1])
            end
        end
        return R
    end
end