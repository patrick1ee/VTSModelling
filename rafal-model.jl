using NeuralDynamics, Plots

function oscill_time_input(A, f, base, range_t)
    R = zeros(length(range_t))
    for i in 1:length(range_t)
        R[i] = A * sin((f / (2 * pi)) * range_t[i] * 4) + base
    end
    return R
end

function sigmoid(x, beta)
    return 1.0 / (1.0 + exp(-beta * (x - 1)))
end

function Enullcline(rE, theta_E, w_EE, w_IE, beta)
    return (theta_E + w_EE * rE - 1 + (1 / beta) * log((-rE + 1) / rE)) / w_IE
end

function Inullcline(rE, theta_I, w_EI, beta)
    return 1 / (1 + exp(-beta * theta_I - beta * w_EI * rE + beta))
end

function max_min_after(R, Lt, start_t=0.1)
    max = 0
    min = 1000
    idx = trunc(Int, start_t * Lt)
    for i in idx:Lt
        if R[i] > max
            max = R[i]
            idx = i
        end
    end
    for i in idx:Lt
        if R[i] < min
            min = R[i]
            idx = i
        end
    end

    return max, min
end

function plot_max_min(tau_E, tau_I, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt, range_theta_E)
    Lt = length(range_t)
    Lte = length(range_theta_E)
    rE_max = zeros(Lte)
    rE_min = zeros(Lte)

    window = [0.05, 0.1]

    for i in 1:Lte
        theta_E = fill(range_theta_E[i], Lt)
        rE, _ = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
        rE_max[i], _ = findmax(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
        rE_min[i], _ = findmin(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
    end
    
    # Plot the results
    x = range_theta_E
    plot(x, [rE_max, rE_min], label=["Emax" "Emin"], xlabel="theta_E", ylabel="E amplitude")
    savefig("myplot.png")
end

function plot_act_time(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
    rE, rI = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)

    # Plot the results
    plot(range_t, [rE, rI], label=["E" "I"], xlabel="t", ylabel="Activity")
    savefig("myplot.png")
end

function plot_act_oscill_time(tau_E, tau_I, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt, A, f, base)
    Lt = length(range_t)
    theta_E = oscill_time_input(A, f, base, range_t)
    rE, rI = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)

    # Plot the results
    p1 = plot(range_t, theta_E, xlabel="t", ylabel="Input")
    p2 = plot(range_t, [rE, rI], xlabel="t", ylabel="Activity")
    plot(p1, p2, layout=(2,1))
    savefig("myplot.png")
end


function simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
    # Initial conditions
    Lt = length(range_t)
    rE = zeros(Lt)
    rI = zeros(Lt)

    # Simulate the model
    for i in 2:Lt
        drE = (dt / tau_E) * (-rE[i-1] + sigmoid(theta_E[i-1] + w_EE * rE[i-1] - w_IE * rI[i-1], beta))
        drI = (dt / tau_I) * (-rI[i-1] + sigmoid(theta_I[i-1] + w_EI * rE[i-1], beta))

        rE[i] = rE[i-1] + drE
        rI[i] = rI[i-1] + drI
    end
    return rE, rI
end

function main()
    # Parameters (time in s)
    tau_E = 0.0032
    tau_I = 0.0032
    w_EE = 2.4
    w_EI = 2.0
    w_IE = 2.0
    beta = 4.0
    T = 1.0
    dt = 0.001

    range_t = 0.0:dt:T
    range_theta_E = 0.0:0.001:2.0

    theta_E = fill(1.4, length(range_t))
    theta_I = fill(0.0, length(range_t))

    #plot_act_time(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
    #plot_max_min(tau_E, tau_I, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt, range_theta_E)

    A = 0.1
    f = 55
    base = 0.6
    plot_act_oscill_time(tau_E, tau_I, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt, A, f, base)

end

main()

