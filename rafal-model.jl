using NeuralDynamics, Plots

function sigmoid(x, beta)
    return 1.0 / (1.0 + exp(-beta * (x - 1)))
end

function simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
    # Initial conditions
    Lt = length(range_t)
    rE = zeros(Lt)
    rI = zeros(Lt)

    # Simulate the model
    for i in 2:Lt
        drE = (dt / tau_E) * (-rE[i-1] + sigmoid(theta_E + w_EE * rE[i-1] - w_IE * rI[i-1], beta))
        drI = (dt / tau_I) * (-rI[i-1] + sigmoid(theta_I + w_EI * rE[i-1], beta))

        rE[i] = rE[i-1] + drE
        rI[i] = rI[i-1] + drI
    end
    return rE, rI
end

function main()
    # Parameters (time in s)
    tau_E = 0.0032
    tau_I = 0.0032
    theta_E = 0.4
    theta_I = 0.0
    w_EE = 2.4
    w_EI = 2.0
    w_IE = 2.0
    beta = 4.0
    T = 1
    dt = 0.001

    range_t = 0.0:dt:T

    # Plot F/I
    #x = range(0, 2.5, 100)
    #y = [(sigmoid(i, beta)) for i in x]
    #plot(x, y, xlabel="Input", ylabel="Firing Rate")
    #savefig("myplot.png")  


    # Simulate the model
    rE, rI = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)

    # Plot the results
    x = range_t
    plot(x, [rE, rI], label=["E" "I"], xlabel="Time", ylabel="Activity")
    savefig("myplot.png")
end

main()

