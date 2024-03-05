module RafalModel

export RafalModel, create_rafal_model, simulate_rafal_model

    struct Node
        tau_E::Float32
        tau_I::Float32
        w_EE::Float32
        w_EI::Float32
        w_IE::Float32
        beta::Float32
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
    

    function create_rafal_model(N::Int64, W::Matrix{Float32}, etta::Float32, tau_E::Float32, tau_I::Float32, w_EE::Float32, w_EI::Float32, w_IE::Float32, beta::Float32)
        nodes = [Node(tau_E, tau_I, w_EE, w_EI, w_IE, beta) for i in 1:N]
        return Network(nodes, W, etta)
    end

    function sigmoid(x, beta)
        return 1.0 / (1.0 + exp(-beta * (x - 1)))
    end

    function simulate_rafal_model(N::Network, range_t, dt, theta_E, theta_I)
        # Initial conditions
        Lt = length(range_t)
        R = [NodeActivity(zeros(Lt), zeros(Lt)) for _ in N.nodes]

        # Simulate the model
        for i in 2:Lt
            for (j, n) in enumerate(N.nodes)

                C = 0.0
                for (k, _) in enumerate(N.nodes)
                    C += N.W[k,j] * R[k].rE[i-1]
                end
                C = N.etta * C / length(N.nodes)
                
                drE = (dt / n.tau_E) * (-R[j].rE[i-1] + sigmoid(theta_E[j][i-1] + n.w_EE * R[j].rE[i-1] - n.w_IE * R[j].rI[i-1] + C, n.beta))
                drI = (dt / n.tau_I) * (-R[j].rI[i-1] + sigmoid(theta_I[j][i-1] + n.w_EI * R[j].rE[i-1], n.beta))

                R[j].rE[i] = R[j].rE[i-1] + drE
                R[j].rI[i] = R[j].rI[i-1] + drI
            end
        end
        return R
    end
end