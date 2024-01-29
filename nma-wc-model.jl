using NeuralDynamics, Plots

function F(x, a, theta)
    return (1.0 + exp(-a * (x - theta)))^(-1) - (1 + exp(a * theta))^(-1)
end

function simulate(tau_E, a_E, thresh_E, tau_I, a_I, thresh_I, wEE, wEI, wIE, wII, rE_init, rI_init, range_t, dt)
    Lt = length(range_t)
    rE = zeros(Lt)
    rI = zeros(Lt)

    rE[1] = rE_init
    rI[1] = rI_init

    for i in 2:Lt
        drE = (dt / tau_E) * (-rE[i-1] + F(wEE * rE[i-1] - wEI * rI[i-1], a_E, thresh_E))
        drI = (dt / tau_I) * (-rI[i-1] + F(wIE * rE[i-1] - wII * rI[i-1], a_I, thresh_I))

        rE[i] = rE[i-1] + drE
        rI[i] = rI[i-1] + drI
    end

    return rE, rI
end

function main()
    tau_E = 1.0
    a_E = 1.2
    thresh_E = 2.8

    tau_I = 2.0
    a_I = 1.0
    thresh_I = 4.0

    wEE = 9.0
    wEI = 4.0
    wIE = 13.0
    wII = 11.0

    rE_init = 0.33
    rI_init = 0.15

    T = 50
    dt = 0.1

    range_t = 0.0:dt:T

    rE, rI = simulate(tau_E, a_E, thresh_E, tau_I, a_I, thresh_I, wEE, wEI, wIE, wII, rE_init, rI_init, range_t, dt)

    # Plot the results
    x = range_t
    plot(x, [rE, rI], label=["E" "I"], xlabel="Time", ylabel="Activity")
    savefig("myplot.png")
end

main()