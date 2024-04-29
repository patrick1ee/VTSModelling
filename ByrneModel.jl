module ByrneModel

    using DiffEqNoiseProcess, Distributions, JSON, Plots

    export ByrneModel, create_byrne_pop, create_byrne_pop_EI, create_byrne_network, create_if_pop, simulate_if_pop, simulate_byrne_EI_network

    struct IFPopulation
        N::Int32
        ex::Float32
        ks::Float32
        kv::Float32
        gamma::Float32 # Hetereogeneity parameter
        tau::Float32
        alpha::Float32
        vth
        vr
    end

    struct EIPopulation
        ex::Float32
        gamma::Float32 # Hetereogeneity parameter
        tau::Float32
    end

    struct Node
        kv_EE::Float32
        kv_EI::Float32
        kv_IE::Float32
        kv_II::Float32
        ks_EE::Float32
        ks_EI::Float32
        ks_IE::Float32
        ks_II::Float32
        alpha_EE::Float32
        alpha_EI::Float32
        alpha_IE::Float32
        alpha_II::Float32
        E::EIPopulation
        I::EIPopulation
        WE
        WI
    end

    struct Network
        nodes::Array{Node, 1}
        W::Matrix{Float32}
        etta::Float32
    end

    struct NodeActivity
        rR_E::Array{Float32, 1}
        rV_E::Array{Float32, 1}
        rZ_E::Array{Float32, 1}
        rR_I::Array{Float32, 1}
        rV_I::Array{Float32, 1}
        rZ_I::Array{Float32, 1}
    end
    

    function create_byrne_pop(ex::Float32, ks::Float32, kv::Float32, gamma::Float32, tau::Float32, alpha::Float32)
        return Population(ex, ks, kv, gamma, tau, alpha)
    end

    function create_byrne_pop_EI(ex::Float32, gamma::Float32, tau::Float32)
        return EIPopulation(ex, gamma, tau)
    end

    NOISE_DEV = 0.0457

    function create_byrne_network(N::Int64, W::Matrix{Float32}, etta::Float32, E::EIPopulation, I::EIPopulation, ks::Float32, kv::Float32, alpha::Float32)
        nodes = [Node(kv, kv, kv, kv, ks, ks, ks, ks, alpha, alpha, alpha, alpha, E, I, WienerProcess(0.0,NOISE_DEV, 1.0), WienerProcess(0.0,NOISE_DEV, 1.0)) for _ in 1:N]
        return Network(nodes, W, etta)
    end

    function create_if_pop(N, ex::Float32, ks::Float32, kv::Float32, gamma::Float32, tau::Float32, alpha::Float32, vth, vr)
        return IFPopulation(N, ex, ks, kv, gamma, tau, alpha, vth, vr)
    end

    function get_kuramoto_parameter(r, v, tau)
        w = pi * tau * r - v*im
        return (1 - w) / (1 + w)
    end

    function get_synaptic_activity(alpha, dt, R)
        return R / (1 + (1 / alpha) * dt)^2
    end

    function simulate_if_pop_timestep(p, rV_prev, dt)

    end

    function simulate_if_pop_post()
    end

    function simulate_if_pop(p::IFPopulation, range_t, dt)
        Lt = length(range_t)
        rV = [zeros(p.N) for i in 1:Lt]
        rVu = [0.0 for i in 1:Lt]
        T = [[0.0] for i in 1:p.N]
        cauchy_dist = Cauchy(p.ex, p.gamma)

        for i in 2:Lt
            EX = rand(cauchy_dist, p.N)
            for j in 1:p.N
                # Sum of differences with other neurons - gap junctions
                sf = 0.0
                for k in 1:p.N
                    sf += rV[i-1][j] - rV[i-1][k]
                end
                sf = p.kv * sf / p.N
                
                # Sum of synaptic activities - offset by last firing time
                st = 0.0
                for k in 1:p.N
                    for (_, t) in enumerate(T[k])
                        st += get_synaptic_activity(p, dt * i - t)
                    end
                end
                st = p.ks * st / p.N

                dV = (dt / p.tau) * (EX[j] + rV[i-1][j] + sf + st)
                rV[i][j] = rV[i-1][j] + dV
            end
            # Record firing times and reset
            for j in 1:p.N
                if rV[i][j] >= p.vth
                    push!(T[j], dt * i)
                    rV[i][j] = p.vr
                end
                rVu[i] += rV[i][j]
            end
            rVu[i] /= p.N
            print(string(i) * "/" * string(Lt) * "\r")
        end
        return rV,rVu
    end

    function simulate_byrne_EI_network(N::Network, range_t, dt, theta_E, theta_I, stim)
        # Initial conditions
        Lt = length(range_t)
        R = [NodeActivity(zeros(Lt), zeros(Lt), zeros(Lt), zeros(Lt), zeros(Lt), zeros(Lt)) for _ in N.nodes]

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
                    C += N.W[k,j] * R[k].rR_E[i-1]
                end
                C = N.etta * C / length(N.nodes)

                drR_E = (dt / n.E.tau) * (-R[j].rR_E[i-1] * (n.kv_EE + n.kv_EI) + 2 * R[j].rR_E[i-1] * R[j].rV_E[i-1] + n.E.gamma / (pi * n.E.tau) + theta_E[j][i-1])
                drR_I = (dt / n.I.tau) * (-R[j].rR_I[i-1] * (n.kv_IE + n.kv_II) + 2 * R[j].rR_I[i-1] * R[j].rV_I[i-1] + n.I.gamma / (pi * n.I.tau) + theta_I[j][i-1])

                U_E = n.ks_EE *get_synaptic_activity(n.alpha_EE, dt, R[j].rR_E[i-1]) + n.ks_EI *get_synaptic_activity(n.alpha_EI, dt, R[j].rR_E[i-1])
                U_I = n.ks_IE *get_synaptic_activity(n.alpha_IE, dt, R[j].rR_I[i-1]) + n.ks_II *get_synaptic_activity(n.alpha_II, dt, R[j].rR_I[i-1])

                drV_E = (dt / n.E.tau) * (n.E.ex + R[j].rV_E[i-1]^2 - pi^2 * n.E.tau^2 * R[j].rR_E[i-1]^2 + U_E + n.kv_EI * (R[j].rV_I[i-1] - R[j].rV_E[i-1]))
                drV_I = (dt / n.I.tau) * (n.I.ex + R[j].rV_I[i-1]^2 - pi^2 * n.I.tau^2 * R[j].rR_I[i-1]^2 + U_I + n.kv_IE * (R[j].rV_E[i-1] - R[j].rV_I[i-1]))

                accept_step!(n.WE, dt, uWE[j], pWE[j])
                accept_step!(n.WI, dt, uWI[j], pWI[j])

                R[j].rR_E[i] = R[j].rR_E[i-1] + drR_E
                R[j].rR_I[i] = R[j].rR_I[i-1] + drR_I
                R[j].rV_E[i] = R[j].rV_E[i-1] + drV_E + NOISE_DEV * (n.WE[i] - n.WE[i - 1])
                R[j].rV_I[i] = R[j].rV_I[i-1] + drV_I + NOISE_DEV * (n.WI[i] - n.WI[i - 1])

                R[j].rZ_E[i] = abs(get_kuramoto_parameter(R[j].rR_E[i], R[j].rV_E[i], n.E.tau))
                R[j].rZ_I[i] = abs(get_kuramoto_parameter(R[j].rR_I[i], R[j].rV_I[i], n.I.tau))
            end

        end
        return R
    end


end