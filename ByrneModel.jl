module ByrneModel

    using DiffEqNoiseProcess, JSON, Plots

    export ByrneModel, create_byrne_pop, simulate_byrne_pop

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

    function get_kuramoto_parameter(r, v, tau)
        w = pi * tau * r - v*im
        return (1 - w) / (1 + w)
    end

    function get_synaptic_activity(p::Population, t)
        return p.alpha^2 * t * exp(-p.alpha * t)
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