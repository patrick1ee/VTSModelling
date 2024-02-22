module ByrneModel

    using DiffEqNoiseProcess, Distributions, JSON, Plots

    export ByrneModel, create_byrne_pop, create_if_pop, simulate_byrne_pop, simulate_if_pop

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

    struct Population
        ex::Float32
        ks::Float32
        kv::Float32
        gamma::Float32 # Hetereogeneity parameter
        tau::Float32
        alpha::Float32
    end

    struct Network
        ex::Float32
        ks::Float32
        kv::Float32
        gamma::Float32 # Hetereogeneity parameter
        tau::Float32
        w_IE::Float32
        beta::Float32
        WE
        WI
    end
    

    function create_byrne_pop(ex::Float32, ks::Float32, kv::Float32, gamma::Float32, tau::Float32, alpha::Float32)
        return Population(ex, ks, kv, gamma, tau, alpha)
    end

    function create_if_pop(N, ex::Float32, ks::Float32, kv::Float32, gamma::Float32, tau::Float32, alpha::Float32, vth, vr)
        return IFPopulation(N, ex, ks, kv, gamma, tau, alpha, vth, vr)
    end

    function get_kuramoto_parameter(r, v, tau)
        w = pi * tau * r - v*im
        return (1 - w) / (1 + w)
    end

    function get_synaptic_activity(p, t)
        return p.alpha^2 * t * exp(-p.alpha * t)
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

    function simulate_byrne_pop(p::Population, range_t, dt)
        # Initial conditions
        Lt = length(range_t)
        rR = zeros(Lt)
        rV = zeros(Lt)
        rZ = zeros(Lt)

        # Simulate the model
        for i in 2:Lt
            drR = (dt / p.tau) * (-p.kv * rR[i-1] + 2 * rR[i-1] * rV[i-1] + p.gamma / (pi * p.tau))

            U = get_synaptic_activity(p, range_t[i-1])
            drV = (dt / p.tau) * (p.ex + rV[i-1]^2 - pi^2 * p.tau^2 * rR[i-1]^2 + p.ks * U)

            rR[i] = rR[i-1] + drR
            rV[i] = rV[i-1] + drV
            rZ[i] = abs(get_kuramoto_parameter(rR[i], rV[i], p.tau))

        end
        return rR, rV, rZ
    end


end