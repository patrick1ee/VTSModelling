using NeuralDynamics, Plots

function oscill_time_input(A, f, base, phase, range_t)
    Lt = length(range_t)
    R = zeros(length(range_t))
    for i in 1:Lt
        R[i] = A * sin(f * 2 * pi * range_t[i] + phase) + base
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

function plot_max_min(tau_E, tau_I, w_EE, w_EI, w_IE, beta, range_t, dt, range_theta_input, theta_const, input_pop)
    Lt = length(range_t)
    Lte = length(range_theta_input)
    rE_max = zeros(Lte)
    rE_min = zeros(Lte)

    window = [0.05, 0.1]

    for i in 1:Lte
        theta_E = input_pop == "E" ? fill(range_theta_input[i], Lt) : fill(theta_const, Lt)
        theta_I = input_pop == "I" ? fill(range_theta_input[i], Lt) : fill(theta_const, Lt)

        rE, _ = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
        rE_max[i], _ = findmax(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
        rE_min[i], _ = findmin(rE[trunc(Int, window[1] / dt):trunc(Int, window[2] / dt)])
    end
    
    # Plot the results
    x = range_theta_input
    plot(x, [rE_max, rE_min], label=["max" "min"], xlabel="theta_"*input_pop, ylabel="E amplitude")
    savefig("myplot.png")
end

function plot_act_time(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)
    rE, rI = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)

    # Plot the results
    plot(range_t, [rE, rI], label=["E" "I"], xlabel="t", ylabel="Activity")
    savefig("myplot.png")
end

function plot_act_oscill_time(tau_E, tau_I, w_EE, w_EI, w_IE, beta, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)
    theta_E = oscill_time_input(E_A, E_f, E_base, E_phase, range_t)
    theta_I = oscill_time_input(I_A, I_f, I_base, I_phase, range_t)
    rE, rI = simulate(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)

    # Plot the results
    p1 = plot(range_t, [theta_E, theta_I], label=["E", "I"], xlabel="t", ylabel="Input")
    p2 = plot(range_t, [rE, rI], label=["E", "I"], xlabel="t", ylabel="Activity")
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

    #theta_E = fill(1.3, length(range_t))
    #theta_I = fill(0.75, length(range_t))
    #plot_act_time(tau_E, tau_I, theta_E, theta_I, w_EE, w_EI, w_IE, beta, range_t, dt)

    #range_theta = 0.0:0.001:2.0
    #const_theta = 1.3
    #plot_max_min(tau_E, tau_I, w_EE, w_EI, w_IE, beta, range_t, dt, range_theta, const_theta, "I")

    E_A = 0.5
    E_f = 4
    E_base = 0.5
    E_phase = 0.0
    I_A = 0.5
    I_f = 4
    I_base = 0.5
    I_phase = 0.2 / dt
    plot_act_oscill_time(tau_E, tau_I, w_EE, w_EI, w_IE, beta, range_t, dt, E_A, E_f, E_base, E_phase, I_A, I_f, I_base, I_phase)

end

main()

