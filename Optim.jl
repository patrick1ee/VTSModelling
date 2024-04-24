include("./BenoitModel.jl")
include("./Signal.jl")

using BlackBoxOptim, CSV, DataFrames, Distributions, DSP, FFTW, KernelDensity, KissSmoothing, Optimization, OptimizationCMAEvolutionStrategy, Plots, Statistics

using .BenoitModel: create_benoit_model, create_benoit_node, create_benoit_network, simulate_benoit_model
using .Signal: get_beta_data, get_pow_spec, get_hilbert_amplitude_pdf, get_burst_durations, get_plv_freq

const SR = 1000  # recording sampling rate in Hz, do not change this

function run_act_time(m, simulate, range_t, dt, theta_E, theta_I, stim_response)
    theta_E_t = [fill(i, length(range_t)) for i in theta_E]
    theta_I_t = [fill(i, length(range_t)) for i in theta_I]

    for i in 1:length(stim_response)
        theta_E_t[1][i] = theta_E_t[1][i] .+ stim_response[i]
    end

    R = simulate(m, range_t, dt, theta_E_t, theta_I_t)
    lR = length(R)
    for i in 1:lR
        urE = mean(R[i].rE)
        srE = std(R[i].rE)
        urI = mean(R[i].rI)
        srI = std(R[i].rI)

        for j in 1:length(R[i].rE)
            R[i].rE[j] = (R[i].rE[j] - urE) / srE
            R[i].rI[j] = (R[i].rI[j] - urI) / srI
        end
    end

    T = [range_t for i in 1:length(theta_E)]
    return DataFrame(T=T, R=R, theta_E=theta_E_t, theta_I=theta_I_t)
end

function cost_bb(params)
    SR = 1000

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    tau_E = Float32(params[1])
    tau_I = Float32(params[1])
    w_EE = Float32(params[2])
    w_EI = Float32(params[3])
    w_IE = Float32(params[4])
    beta = Float32(params[5])
    theta_E_A_param = Float32(params[6])
    theta_I_A_param = Float32(params[7])
    theta_E_B_param = Float32(params[6])
    theta_I_B_param = Float32(params[7])

    model = create_benoit_model(N, W, etta, tau_E, tau_I, w_EE, w_EI, w_IE, beta)
    
    T = 100.0
    dt = 0.001
    range_t = 0.0:dt:T

    response = fill(0.0, length(range_t)) 

    #for i in 1:6:T-6
    #    #Start pulse
    #    for j in 0:24
    #        for k in 0:2:10
    #            response[Int64(trunc(i*1000+j*200+k*(1000/130)))] = 0.001684
    #        end
    #    end
    #end

    theta_E = [theta_E_A_param, theta_E_B_param]
    theta_I = [theta_I_A_param, theta_I_B_param]
    stim = response
    df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

    psd_df = CSV.read("data/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]

    beta_amp_pdf_df = CSV.read("data/beta-hpdf.csv", DataFrame)
    yBAPDFdat = beta_amp_pdf_df[!, 2]

    beta_dur_pdf_df = CSV.read("data/beta-dur-hpdf.csv", DataFrame)
    yBDPDFdat = beta_dur_pdf_df[!, 2]

    plv_df = CSV.read("data/plvs.csv", DataFrame)
    yPLVdat = plv_df[!, 2]

    #zscore
    Eproc = df.R[1].rE .- mean(df.R[1].rE) / std(df.R[1].rE)
    Eproc_alt = df.R[2].rE .- mean(df.R[2].rE) / std(df.R[2].rE)
    Ebeta = get_beta_data(df.R[1].rE)
    Ebeta = Ebeta .- mean(Ebeta) / std(Ebeta)

    freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)
    _, yBAPDFmod, Ebeta_ha = get_hilbert_amplitude_pdf(Ebeta)
    _, yBDPDFmod, _ = get_burst_durations(Ebeta_ha)
    yPLVmod = get_plv_freq(Eproc, Eproc_alt)

    if length(yPSDmod) > length(yPSDdat)
        yPSDmod = yPSDmod[1:length(yPSDdat)]
    elseif length(yPSDmod) < length(yPSDdat)
        yPSDdat = yPSDdat[1:length(yPSDmod)]
    end
    
    coeffs = [1.3, 0.9, 0.9, 0.9]
    cost1 = 1.0
    cost2 = 1.0
    cost3 = 1.0
    cost4 = 1.0

    cost1 = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2)) / 4

    if length(yBAPDFmod) > 0
        cost2 = (sum((yBAPDFdat .- yBAPDFmod).^2) / sum((yBAPDFdat .- mean(yBAPDFdat)).^2)) / 4
    end
    if length(yBDPDFmod) > 0
        cost3 = (sum((yBDPDFdat .- yBDPDFmod).^2) / sum((yBDPDFdat .- mean(yBDPDFdat)).^2)) / 4
    end

    cost4 = (sum((yPLVdat .- yPLVmod).^2) / sum((yPLVdat .- mean(yPLVdat)).^2)) / 4

    cost = coeffs[1]*cost1 + coeffs[2]*cost2 + coeffs[3]*cost3 + coeffs[4]*cost4

    filename="costs.txt"

    if isfile(filename)
        fileID = open(filename, "w")
        println(fileID, tau_E, tau_I, w_EE, w_EI, w_IE, beta, "::", cost1, "::", cost2, "::", cost3, "::", cost4, "\n")
        close(fileID)
    end

    return cost
end

function init_param(bounds, NPARAMS=2500)
    tau_A_dist = Uniform(bounds[1][1], bounds[1][2])
    w_EE_A_dist = Uniform(bounds[2][1], bounds[2][2])
    w_EI_A_dist = Uniform(bounds[3][1], bounds[3][2])
    w_IE_A_dist = Uniform(bounds[4][1], bounds[4][2])
    beta_A_dist = Uniform(bounds[5][1], bounds[5][2])
    theta_E_A_dist = Uniform(bounds[6][1], bounds[6][2])
    theta_I_A_dist = Uniform(bounds[7][1], bounds[7][2])

    tau_B_dist = Uniform(bounds[8][1], bounds[8][2])
    w_EE_B_dist = Uniform(bounds[9][1], bounds[9][2])
    w_EI_B_dist = Uniform(bounds[10][1], bounds[10][2])
    w_IE_B_dist = Uniform(bounds[11][1], bounds[11][2])
    beta_B_dist = Uniform(bounds[12][1], bounds[12][2])
    theta_E_B_dist = Uniform(bounds[13][1], bounds[13][2])
    theta_I_B_dist = Uniform(bounds[14][1], bounds[14][2])

    etta_dist = Uniform(bounds[15][1], bounds[15][2])
    w12_dist = Uniform(bounds[16][1], bounds[16][2])
    w21_dist = Uniform(bounds[17][1], bounds[17][2])

    psd_df = CSV.read("data/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    peakDat = argmax(yPSDdat)

    csv_df_w = DataFrame(
        etta = [0.0], w12 = [0.0], w21 = [0.0],
        tau_A=[0.0], w_EE_A=[0.0], w_EI_A=[0.0], w_IE_A=[0.0], beta_A=[0.0], theta_E_A=[0.0], theta_I_A=[0.0],
        tau_B=[0.0], w_EE_B=[0.0], w_EI_B=[0.0], w_IE_B=[0.0], beta_B=[0.0], theta_E_B=[0.0], theta_I_B=[0.0]
    )
    CSV.write("data/params-2.csv", csv_df_w)

    count = 0

    while count < NPARAMS
        etta_p = 0.5
        w12_p = 1.0
        w21_p = 1.0

        tau_A_p = Float32(rand(tau_A_dist))
        w_EE_A_p = Float32(rand(w_EE_A_dist))
        w_EI_A_p = Float32(rand(w_EI_A_dist))
        w_IE_A_p = Float32(rand(w_IE_A_dist))
        beta_A_p = Float32(rand(beta_A_dist))
        theta_E_A_p = Float32(rand(theta_E_A_dist))
        theta_I_A_p = Float32(rand(theta_I_A_dist))

        #tau_B_p = Float32(rand(tau_B_dist))
        #w_EE_B_p = Float32(rand(w_EE_B_dist))
        #w_EI_B_p = Float32(rand(w_EI_B_dist))
        #w_IE_B_p = Float32(rand(w_IE_B_dist))
        #beta_B_p = Float32(rand(beta_B_dist))
        #theta_E_B_p = Float32(rand(theta_E_B_dist))
        #theta_I_B_p = Float32(rand(theta_I_B_dist))

        tau_B_p = tau_A_p
        w_EE_B_p = w_EE_A_p
        w_EI_B_p = w_EI_A_p
        w_IE_B_p = w_IE_A_p
        beta_B_p = beta_A_p
        theta_E_B_p = Float32(rand(theta_E_B_dist))
        theta_I_B_p = Float32(rand(theta_I_B_dist))

        node_A = create_benoit_node(tau_A_p, tau_A_p, w_EE_A_p, w_EI_A_p, w_IE_A_p, beta_A_p)
        node_B = create_benoit_node(tau_B_p, tau_B_p, w_EE_B_p, w_EI_B_p, w_IE_B_p, beta_B_p)
        model = create_benoit_network([node_A, node_B], [Float32(0.0) Float32(w12_p); Float32(w21_p) Float32(0.0)], Float32(etta_p))

        T = 100.0
        dt = 0.001
        range_t = 0.0:dt:T
        response = fill(0.0, length(range_t)) 
        #for i in 1:6:T-6
        #    #Start pulse
        #    for j in 0:24
        #        for k in 0:2:10
        #            response[Int64(trunc(i*1000+j*200+k*(1000/130)))] = 0.001684
        #        end
        #    end
        #end
        theta_E = [theta_E_A_p, theta_E_B_p]
        theta_I = [theta_I_A_p, theta_I_B_p]
        stim = response
        df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

        #zscore
        Eproc = df.R[1].rE .- mean(df.R[1].rE) / std(df.R[1].rE)

        freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)
        peakMod = argmax(yPSDmod)
        #print(string(freq[peakMod])*" "*string(xPSD[peakDat])*"\n")

        if abs(xPSD[peakDat] - freq[peakMod]) <= 1.0 && abs(yPSDmod[peakMod] - yPSDdat[peakDat]) <= 0.25*yPSDdat[peakDat]
            new_row= (
                etta=etta_p, w12=w12_p, w21=w21_p,
                tau_A=tau_A_p, w_EE_A=w_EE_A_p, w_EI_A=w_EI_A_p, w_IE_A=w_IE_A_p, beta_A=beta_A_p, theta_E_A=theta_E_A_p, theta_I_A=theta_I_A_p,
                tau_B=tau_B_p, w_EE_B=w_EE_B_p, w_EI_B=w_EI_A_p, w_IE_B=w_IE_B_p, beta_B=beta_B_p, theta_E_B=theta_E_B_p, theta_I_B=theta_I_B_p
            )
            df_csv_r = CSV.read("data/params-2.csv", DataFrame)
            push!(df_csv_r, new_row)
            CSV.write("data/params-2.csv", df_csv_r)
            count += 1
            print("Added " * string(count) * " / 2500 parameters\n")
            print("pmidx: " * string(peakMod) * "pdidx: " * string(peakDat) * "peak mod: " * string(freq[peakMod]) * " peak dat: " * string(xPSD[peakDat]) * " diff: " * string(abs(xPSD[peakDat] - freq[peakMod])) * " " * string(abs(yPSDmod[peakMod] - yPSDdat[peakDat])) * " " * string(0.25*yPSDdat[peakDat]) * "\n")
        else
            #print("Rejected " * string(tau_p) * " " * string(w_EE_p) * " " * string(w_EI_p) * " " * string(w_IE_p) * " " * string(beta_p) * " " * string(theta_E_p) * " " * string(theta_I_p) * " -> " * string(abs(xPSD[peakDat] - xPSD[peakMod])) * "\n")
        end
    end
end

#=function cost(M, dt, range_t, nTr)

    rampUpTime = 40  # in s
    nBlperTr = 12
    interBlTime = 1  # in s
    interTrTime = 5  # in s
    blTime = 5  # in s

    nBl = nBlperTr * nTr

    toIdx(t) = round(Int, t / dt)
    rampUpN = toIdx(rampUpTime)
    interBlN = toIdx(interBlTime)
    interTrN = toIdx(interTrTime)
    blN = toIdx(blTime)

    N = rampUpN + (nTr + 1) * interTrN + (nBl - nTr) * interBlN + nBl * blN

    stim_prog = zeros(N, 3)
    stim_prog[:, 2] .= -99

    ## STIM STUFF ##

    ################
    theta_E = []
    theta_I = []

    R = simulate_benoit_model(M, range_t, theta_E, theta_I)
    E = R[1].rE

    #Emean = mean(E[rampUpN:end])
    #Estd = std(E[rampUpN:end])
    #Eproc = (E .- Emean) / Estd
    Eproc = E
    
    # FEATURE COMPUTATION
    nPartialTr = 9
    partialIdx = rand(1:nTr, nPartialTr)
    partialTrIdx = trialIdx[partialIdx, :]

    # PSD
    PSDverbose = false
    #yPSDmod = getTrPSD(Eproc, 1 / dt, partialTrIdx, dataFeatures.PSD.xPSD, dataFeatures.PSD.wLratio, PSDverbose)
    yPSDmod = [get_PSD(Eproc, 1 / dt, PSDverbose)]
    maxPSD, maxIdx = findmax(yPSDmod)
    fCenter = dataFeatures.PSD.xPSD[maxIdx]

    ##Filter?

    #yPSDdat = dataFeatures.PSD.yPSD


    #cost1 = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2)) / 1
    cost1 = sum((yPSDdat .- yPSDmod).^2)
    cost = cost1p = [0.0165082, 4.79867, 7.75704, 9.93353, 2.27035, -0.115528, -1.50204]

    filename("costs.txt")

    if isfile(filename)
        fileID = open(filename, "a")
        println(fileID, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f \n",
            M.nodes[1].tau_E, M.nodes[1].tau_I, M.nodes[1].w_EE, M.nodes[1].w_EI, M.nodes[1].w_IE, M.nodes[1].beta, cost1)
        close(fileID)
    end

    return cost

end=#

function opt_param()
    best_params = [[],[],[],[],[],[],[],[],[],[]]
    best_costs = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    df_csv_r = CSV.read("data/params.csv", DataFrame)
    for (i, row) in enumerate(eachrow(df_csv_r))
        p = [v for v in values(df_csv_r[i,:])]
        cost = cost_bb(p)

        max_index = argmax(best_costs)
        if cost < best_costs[max_index]
            best_costs[max_index] = cost
            best_params[max_index] = p
            println("Costs: ", best_costs)   
        end
    end
    for (i, p) in enumerate(best_params)
        println("Best params ", i, ": ", p)
    end
end

function rosenbrock2d(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function Optim()
    p_range=[(0.016, 0.017), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-2.0, 10.0), (-10.0, 2.0)]
    df_csv_r = CSV.read("data/params-wc-mir-1.3-0.9-0.9-0.9.csv", DataFrame)
    good_guess = []
    for i in 1:nrow(df_csv_r)
        push!(good_guess, [v for v in values(df_csv_r[i,:])])
    end
    res = bboptimize(cost_bb, good_guess; SearchRange=p_range)
    return

    p_bounds = [
        (0.016, 0.017), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-2.0, 10.0), (-10.0, 2.0),
        (0.016, 0.017), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-2.0, 10.0), (-10.0, 2.0),
        (0.0, 0.1), (0.0, 1.0), (0.0, 1.0)
    ]
    init_param(p_bounds)
    #opt_param()

    return

    df_csv_r = CSV.read("data/params.csv", DataFrame)
    V = []
    for (i, row) in enumerate(eachrow(df_csv_r))
        push!(V, [v for v in values(df_csv_r[i,:])])
    end

    good_guess = [0.016324788331985474,8.401309967041016,6.917157173156738,9.229533195495605,4.105310916900635,0.19816090166568756,-1.3066587448120117]
    #p_range = [(0.016, 0.017), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-2.0, 10.0), (-10.0, 2.0)]
    p_range=[(0.0, 0.5), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)]

    #res = bboptimize(cost_bb, good_guess, SearchRange=p_range, MaxSteps=100000)

    lb = [0.0, 0.0, 0.0, 0.0, 0.0, -30.0, -30.0]
    ub = [0.5, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    f = OptimizationFunction(cost_bb)
    prob = Optimization.OptimizationProblem(f, good_guess; lb=lb, ub=ub)
    sol = solve(prob, CMAEvolutionStrategyOpt())
    println(sol)
end

#opt_param()
Optim()



