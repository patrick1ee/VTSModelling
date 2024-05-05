include("./BenoitModel.jl")
include("./ByrneModel.jl")
include("./Oscilltrack.jl")
include("./Signal.jl")
include("./Stimulation.jl")

using BlackBoxOptim, CSV, DataFrames, Distributions, DSP, FFTW, KernelDensity, KissSmoothing, Optimization, OptimizationCMAEvolutionStrategy, Plots, Statistics

using .BenoitModel: create_benoit_model, create_benoit_node, create_benoit_network, simulate_benoit_model
using .ByrneModel: create_byrne_pop, create_byrne_pop_EI, create_byrne_network, create_if_pop, simulate_if_pop, simulate_byrne_EI_network
using .Signal: get_beta_data, get_pow_spec, get_hilbert_amplitude_pdf, get_burst_profiles, get_plv_freq
using .Stimulation: create_stim_block

using .Oscilltrack: Oscilltracker

const SR = 1000.0  # recording sampling rate in Hz, do not change this

function run_act_time(m, simulate, range_t, dt, theta_E, theta_I, oscilltracker, stimblock)
    theta_E_t = [fill(i, length(range_t)) for i in theta_E]
    theta_I_t = [fill(i, length(range_t)) for i in theta_I]

    R, _ = simulate(m, range_t, dt, theta_E_t, theta_I_t, oscilltracker, stimblock)

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


function run_byrne_net(N, simulate, range_t, T, dt, theta_E, theta_I, stim)
    theta_E_t = [fill(i, length(range_t)) for i in theta_E]
    theta_I_t = [fill(i, length(range_t)) for i in theta_I]

    R = simulate(N, range_t, dt, theta_E_t, theta_I_t, stim)
    T = [range_t for i in 1:length(theta_E)]   # Convert to s
    return DataFrame(T=T, R=R, theta_E=theta_E_t, theta_I=theta_I_t)
end

function cost_bb_bc(params)
    csv_path = "data/P7/06_02_2024_P7_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1"
    SR = 1000

    psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    beta_amp_pdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
    yBAPDFdat = beta_amp_pdf_df[!, 2]
    beta_dur_pdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
    yBDPDFdat = beta_dur_pdf_df[!, 2]
    plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
    yPLVdat = plv_df[!, 2]

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    tau_p = Float32(params[1])
    ex_p = Float32(params[2])
    ks_p = Float32(params[3])
    kv_p = Float32(params[4])
    gamma_p = Float32(params[5])
    alpha_p = Float32(params[6])
    theta_E_p = Float32(params[7])
    theta_I_p = Float32(params[8])

    E = create_byrne_pop_EI(ex_p, gamma_p, tau_p)
    I = create_byrne_pop_EI(ex_p, gamma_p, tau_p)
    net = create_byrne_network(N, W, etta, E, I, ks_p, kv_p, alpha_p)
    
    #timescale now ms
    T = 100000.0
    dt = 1.0  
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
    theta_E = [theta_E_p, theta_E_p]
    theta_I = [theta_I_p, theta_I_p]
    stim = response
    df= run_byrne_net(net, simulate_byrne_EI_network, range_t, T, dt, theta_E, theta_I, [])
    #zscore
    Eproc = df.R[1].rV_E 
    Eproc = Eproc .- mean(Eproc) / std(Eproc)
    Eproc_alt = df.R[2].rV_E
    Eproc_alt = Eproc_alt .- mean(Eproc_alt) / std(Eproc_alt)

    Ebeta = df.R[1].rV_E
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
    
    coeffs = [1.0, 1.0, 1.0, 1.0]
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
        println(fileID, tau_p, ", ", ex_p, ", ", ks_p, ", ", kv_p, ", ", gamma_p, ", ", alpha_p, ", ", theta_E_p, ", ", theta_I_p, "::", cost1, "::", cost2, "::", cost3, "::", cost4, "\n")
        close(fileID)
    end

    return cost
end


stimBlock = create_stim_block(100.0, 25, 25, 0.0:0.001:100.0, 1)
gamma_param = 0.1 # or 0.05
OT_suppress = 0.3
target_phase = pi / 2.0
target_freq = 10.0
oscilltracker = Oscilltracker(target_freq, target_phase, SR, OT_suppress, gamma_param)

function cost_bb(params)
    P= ARGS[1]
    name = ARGS[2]
    
    csv_path = "data/"*P*"/"*name

   # print(csv_path)

    psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    beta_amp_pdf_df = CSV.read(csv_path*"/bapdf.csv", DataFrame)
    yBAPDFdat = beta_amp_pdf_df[!, 2]
    beta_dur_pdf_df = CSV.read(csv_path*"/bdpdf.csv", DataFrame)
    yBDPDFdat = beta_dur_pdf_df[!, 2]
    plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
    yPLVdat = plv_df[!, 2]

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    tau_E = Float32(params[1])
    tau_I = Float32(params[1])
    w_EE = Float32(params[2])
    w_EI = Float32(params[3])
    w_IE = Float32(params[4])
    beta = Float32(params[5])
    noise_dev = Float32(params[6])
    stim_mag = Float32(params[7])
    theta_E_A_param = Float32(params[8])
    theta_I_A_param = Float32(params[9])
    theta_E_B_param = Float32(params[8])
    theta_I_B_param = Float32(params[9])

    model = create_benoit_model(N, W, etta, tau_E, tau_I, w_EE, w_EI, w_IE, beta, noise_dev, stim_mag)
    
    T = 100.0
    dt = 0.001
    range_t = 0.0:dt:T

    theta_E = [theta_E_A_param, theta_E_B_param]
    theta_I = [theta_I_A_param, theta_I_B_param]

    init_slice = 100
    #num_trials = 1

    # Fold into arrays of first trial
    #=df_trials = []
    for _ in 1:num_trials
        df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)
        push!(df_trials, df)
    end
    for i in init_slice:length(df_trials[1].R[1].rE)
        sum = [0.0, 0.0]
        for j in num_trials
            sum[1] += df_trials[j].R[1].rE[i]
            sum[2] += df_trials[j].R[2].rE[i]
        end
        df_trials[1].R[1].rE[i] = sum[1] / num_trials
        df_trials[1].R[2].rE[i] = sum[2] / num_trials
    end=#

    df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, oscilltracker, stimBlock)
    #zscore
    Eproc = df.R[1].rE[init_slice:end] .- mean(df.R[1].rE[init_slice:end]) / std(df.R[1].rE[init_slice:end])
    Eproc_alt = df.R[2].rE[init_slice:end] .- mean(df.R[2].rE[init_slice:end]) / std(df.R[2].rE[init_slice:end])
    Ebeta = get_beta_data(df.R[1].rE[init_slice:end])
    Ebeta = Ebeta .- mean(Ebeta) / std(Ebeta)

    freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)

    _, _, ha = get_hilbert_amplitude_pdf(Ebeta)
    S, N = denoise(convert(AbstractArray{Float64}, ha))
    xBAPDFmod, yBAPDFmod, aBDPDF, yBDPDFmod, _, _ = get_burst_profiles(S)

    xPLVmod, yPLVmod = get_plv_freq(Eproc, Eproc_alt)

    if length(yPSDmod) > length(yPSDdat)
        yPSDmod = yPSDmod[1:length(yPSDdat)]
    elseif length(yPSDmod) < length(yPSDdat)
        yPSDdat = yPSDdat[1:length(yPSDmod)]
    end
    
    coeffs = [1.0, 1.0, 1.0, 1.0]
    cost1 = 2.0
    cost2 = 2.0
    cost3 = 2.0
    cost4 = 2.0

    cost1 = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2))
    if length(yBAPDFmod) > 0
        cost2 = (sum((yBAPDFdat .- yBAPDFmod).^2) / sum((yBAPDFdat .- mean(yBAPDFdat)).^2))
    end
    if length(yBDPDFmod) > 0
        cost3 = (sum((yBDPDFdat .- yBDPDFmod).^2) / sum((yBDPDFdat .- mean(yBDPDFdat)).^2))
    end
    cost4 = (sum((yPLVdat .- yPLVmod).^2) / sum((yPLVdat .- mean(yPLVdat)).^2))

    cost = coeffs[1]*cost1 + coeffs[2]*cost2 + coeffs[3]*cost3 + coeffs[4]*cost4
    cost = cost / 4

    filename="./jobs-out/costs-"*name*".txt"

    if isfile(filename)
        fileID = open(filename, "w")
        println(fileID, "::", tau_E, "::", tau_I, "::", w_EE, "::", w_EI, "::", w_IE, "::", beta, "::", cost1, "::", cost2, "::", cost3, "::", cost4, "\n")
        close(fileID)
    end

    new_row= (cost=cost,)
    df_csv_r = CSV.read("./jobs-out/costs-"*name*".csv", DataFrame)
    push!(df_csv_r, new_row)
    CSV.write("./jobs-out/costs-"*name*".csv", df_csv_r)

    return cost
end

function init_param(bounds, P, name, NPARAMS=2500)
    csv_path = "data/"*P*"/"*name

    # Free parameters
    tau_A_dist = Uniform(bounds[1][1], bounds[1][2])
    w_EE_A_dist = Uniform(bounds[2][1], bounds[2][2])
    w_EI_A_dist = Uniform(bounds[3][1], bounds[3][2])
    w_IE_A_dist = Uniform(bounds[4][1], bounds[4][2])
    beta_A_dist = Uniform(bounds[5][1], bounds[5][2])
    noise_A_dist = Uniform(bounds[6][1], bounds[6][2])
    theta_E_A_dist = Uniform(bounds[7][1], bounds[7][2])
    theta_I_A_dist = Uniform(bounds[8][1], bounds[8][2])

    #Fixed parameters
    etta = Float32(0.5)
    W = [Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    stim_mag = Float32(0.0)

    psd_df = CSV.read(csv_path*"/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    peakDat = argmax(yPSDdat)

    #csv_df_w = DataFrame(tau=[0.0], w_EE=[0.0], w_EI=[0.0], w_IE=[0.0], beta=[0.0], noise=[0.0], theta_E=[0.0], theta_I=[0.0])
    #CSV.write(csv_path*"/params.csv", csv_df_w)

    count = 0
    while count < NPARAMS
        tau_A_p = Float32(rand(tau_A_dist))
        w_EE_A_p = Float32(rand(w_EE_A_dist))
        w_EI_A_p = Float32(rand(w_EI_A_dist))
        w_IE_A_p = Float32(rand(w_IE_A_dist))
        beta_A_p = Float32(rand(beta_A_dist))
        noise_p = Float32(rand(noise_A_dist))
        theta_E_A_p = Float32(rand(theta_E_A_dist))
        theta_I_A_p = Float32(rand(theta_I_A_dist))

        node_A = create_benoit_node(tau_A_p, tau_A_p, w_EE_A_p, w_EI_A_p, w_IE_A_p, beta_A_p, noise_p)
        node_B = create_benoit_node(tau_A_p, tau_A_p, w_EE_A_p, w_EI_A_p, w_IE_A_p, beta_A_p, noise_p)
        model = create_benoit_network([node_A, node_B], W, etta, noise_p, stim_mag)

        T = 100.0
        dt = 0.001
        range_t = 0.0:dt:T
        theta_E = [theta_E_A_p, theta_E_A_p]
        theta_I = [theta_I_A_p, theta_I_A_p]

        stimBlock = create_stim_block(100.0, 100, 100, 10, 1)
        SR = 1000.0
        gamma_param = 0.1 # or 0.05
        OT_suppress = 0.3
        target_phase = pi / 2.0
        target_freq = 10.0
        oscilltracker = Oscilltracker(target_freq, target_phase, SR, OT_suppress, gamma_param)

        df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, oscilltracker, stimBlock)

        #zscore
        Eproc = df.R[1].rE .- mean(df.R[1].rE) / std(df.R[1].rE)

        freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)
        peakMod = argmax(yPSDmod)
        #print(string(freq[peakMod])*" "*string(xPSD[peakDat])*"\n")

        if abs(xPSD[peakDat] - freq[peakMod]) <= 1.0 && abs(yPSDmod[peakMod] - yPSDdat[peakDat]) <= 0.25*yPSDdat[peakDat]
            new_row= (
                tau=tau_A_p, w_EE=w_EE_A_p, w_EI=w_EI_A_p, w_IE=w_IE_A_p, beta=beta_A_p, noise=noise_p, theta_E=theta_E_A_p, theta_I=theta_I_A_p,
             )
            df_csv_r = CSV.read(csv_path*"/params.csv", DataFrame)
            push!(df_csv_r, new_row)
            CSV.write(csv_path*"/params.csv", df_csv_r)
            count += 1
            print("Added " * string(count) * " / 2500 parameters\n")
            print("pmidx: " * string(peakMod) * "pdidx: " * string(peakDat) * "peak mod: " * string(freq[peakMod]) * " peak dat: " * string(xPSD[peakDat]) * " diff: " * string(abs(xPSD[peakDat] - freq[peakMod])) * " " * string(abs(yPSDmod[peakMod] - yPSDdat[peakDat])) * " " * string(0.25*yPSDdat[peakDat]) * "\n")
        else
            #print("Rejected " * string(tau_p) * " " * string(w_EE_p) * " " * string(w_EI_p) * " " * string(w_IE_p) * " " * string(beta_p) * " " * string(theta_E_p) * " " * string(theta_I_p) * " -> " * string(abs(xPSD[peakDat] - xPSD[peakMod])) * "\n")
        end
    end
end

function init_param_bc_mir(bounds, NPARAMS=10)
    tau_dist = Uniform(bounds[1][1], bounds[1][2])
    ex_dist = Uniform(bounds[2][1], bounds[2][2])
    ks_dist = Uniform(bounds[3][1], bounds[3][2])
    kv_dist = Uniform(bounds[4][1], bounds[4][2])
    gamma_dist = Uniform(bounds[5][1], bounds[5][2])
    alpha_dist = Uniform(bounds[6][1], bounds[6][2])
    theta_E_dist = Uniform(bounds[7][1], bounds[7][2])
    theta_I_dist = Uniform(bounds[8][1], bounds[8][2])

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    psd_df = CSV.read("data/P9/07_02_2024_P9_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v2/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    peakDat = argmax(yPSDdat)

    csv_df_w = DataFrame(tau=[0.0], ex=[0.0], ks=[0.0], kv=[0.0], gamma=[0.0], alpha=[0.0], theta_E=[0.0], theta_I=[0.0])
    CSV.write("data/params/params-bc-mir.csv", csv_df_w)

    count = 0

    while count < NPARAMS
        tau_p = Float32(rand(tau_dist))
        ex_p = Float32(rand(ex_dist))
        ks_p = Float32(rand(ks_dist))
        kv_p = Float32(rand(kv_dist))
        gamma_p = Float32(rand(gamma_dist))
        alpha_p = Float32(rand(alpha_dist))
        theta_E_p = Float32(rand(theta_E_dist))
        theta_I_p = Float32(rand(theta_I_dist))

        E = create_byrne_pop_EI(ex_p, gamma_p, tau_p)
        I = create_byrne_pop_EI(ex_p, gamma_p, tau_p)
        net = create_byrne_network(N, W, etta, E, I, ks_p, kv_p, alpha_p)
    
        #timescale now ms
        T = 100000.0
        dt = 1.0  
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
        theta_E = [theta_E_p, theta_E_p]
        theta_I = [theta_I_p, theta_I_p]
        stim = response
        df= run_byrne_net(net, simulate_byrne_EI_network, range_t, T, dt, theta_E, theta_I, [])

        #zscore
        Eproc = df.R[1].rV_E[100:end]
        Eproc = Eproc .- mean(Eproc) / std(Eproc)

        freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)
        peakMod = argmax(yPSDmod)
        #print(string(freq[peakMod])*" "*string(xPSD[peakDat])*"\n")

        if abs(xPSD[peakDat] - freq[peakMod]) <= 1.0 && abs(yPSDmod[peakMod] - yPSDdat[peakDat]) <= 0.25*yPSDdat[peakDat]
            new_row= (
                tau=tau_p, ex=ex_p, ks=ks_p, kv=kv_p, gamma=gamma_p, alpha=alpha_p, theta_E=theta_E_p, theta_I=theta_I_p,
             )
            df_csv_r = CSV.read("data/params/params-bc-mir.csv", DataFrame)
            push!(df_csv_r, new_row)
            CSV.write("data/params/params-bc-mir.csv", df_csv_r)
            count += 1
            print("Added " * string(count) * " / 2500 parameters\n")
            print("pmidx: " * string(peakMod) * "pdidx: " * string(peakDat) * "peak mod: " * string(freq[peakMod]) * " peak dat: " * string(xPSD[peakDat]) * " diff: " * string(abs(xPSD[peakDat] - freq[peakMod])) * " " * string(abs(yPSDmod[peakMod] - yPSDdat[peakDat])) * " " * string(0.25*yPSDdat[peakDat]) * "\n")
        else
            #println("Reject::pmidx: " * string(peakMod) * "pdidx: " * string(peakDat) * "peak mod: " * string(freq[peakMod]) * " peak dat: " * string(xPSD[peakDat]) * " diff: " * string(abs(xPSD[peakDat] - freq[peakMod])) * " " * string(abs(yPSDmod[peakMod] - yPSDdat[peakDat])) * " " * string(0.25*yPSDdat[peakDat]) * "\n")
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

function init_param_wc_single(bounds, NPARAMS=2500)
    tau_dist = Uniform(bounds[1][1], bounds[1][2])
    w_EE_dist = Uniform(bounds[2][1], bounds[2][2])
    w_EI_dist = Uniform(bounds[3][1], bounds[3][2])
    w_IE_dist = Uniform(bounds[4][1], bounds[4][2])
    beta_dist = Uniform(bounds[5][1], bounds[5][2])
    theta_E_dist = Uniform(bounds[6][1], bounds[6][2])
    theta_I_dist = Uniform(bounds[7][1], bounds[7][2])

    psd_df = CSV.read("data/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    peakDat = argmax(yPSDdat)

    csv_df_w = DataFrame(tau=[0.0], w_EE=[0.0], w_EI=[0.0], w_IE=[0.0], beta=[0.0], theta_E=[0.0], theta_I=[0.0])
    CSV.write("data/params-wc-single-1.2-0.9-0.9.csv", csv_df_w)

    count = 0

    while count < NPARAMS
        tau_p = Float32(rand(tau_dist))
        w_EE_p = Float32(rand(w_EE_dist))
        w_EI_p = Float32(rand(w_EI_dist))
        w_IE_p = Float32(rand(w_IE_dist))
        beta_p = Float32(rand(beta_dist))
        theta_E_p = Float32(rand(theta_E_dist))
        theta_I_p = Float32(rand(theta_I_dist))

        model = create_benoit_model(1, [Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)], Float32(0.5), tau_p, tau_p, w_EE_p, w_EI_p, w_IE_p, beta_p)

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
        theta_E = [theta_E_p]
        theta_I = [theta_I_p]
        stim = response
        df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

        #zscore
        Eproc = df.R[1].rE .- mean(df.R[1].rE) / std(df.R[1].rE)

        freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)
        peakMod = argmax(yPSDmod)
        #print(string(freq[peakMod])*" "*string(xPSD[peakDat])*"\n")

        if abs(xPSD[peakDat] - freq[peakMod]) <= 1.0 && abs(yPSDmod[peakMod] - yPSDdat[peakDat]) <= 0.25*yPSDdat[peakDat]
            new_row= (
                tau=tau_p, w_EE=w_EE_p, w_EI=w_EI_p, w_IE=w_IE_p, beta=beta_p, theta_E=theta_E_p, theta_I=theta_I_p,
             )
            df_csv_r = CSV.read("data/params-wc-single-1.2-0.9-0.9.csv", DataFrame)
            push!(df_csv_r, new_row)
            CSV.write("data/params-wc-single-1.2-0.9-0.9.csv", df_csv_r)
            count += 1
            print("Added " * string(count) * " / 2500 parameters\n")
            print("pmidx: " * string(peakMod) * "pdidx: " * string(peakDat) * "peak mod: " * string(freq[peakMod]) * " peak dat: " * string(xPSD[peakDat]) * " diff: " * string(abs(xPSD[peakDat] - freq[peakMod])) * " " * string(abs(yPSDmod[peakMod] - yPSDdat[peakDat])) * " " * string(0.25*yPSDdat[peakDat]) * "\n")
        else
            #print("Rejected " * string(tau_p) * " " * string(w_EE_p) * " " * string(w_EI_p) * " " * string(w_IE_p) * " " * string(beta_p) * " " * string(theta_E_p) * " " * string(theta_I_p) * " -> " * string(abs(xPSD[peakDat] - xPSD[peakMod])) * "\n")
        end
    end
end

function opt_param(P, name, N)
    best_params = [[] for i in 1:N]
    best_costs = [1000.0 for i in 1:N]
    csv_path = "data/"*P*"/"*name
    df_csv_r = CSV.read(csv_path*"/params.csv", DataFrame)
    for (i, row) in enumerate(eachrow(df_csv_r))
        p = [v for v in values(df_csv_r[i,:])]
        cost = cost_bb(p)

        max_index = argmax(best_costs)
        if cost < best_costs[max_index]
            best_costs[max_index] = cost
            best_params[max_index] = p  
            println("Cost: " * string(cost) * "\n")
        end
    end
    for (i, p) in enumerate(best_params)
        println("Best params ", i, ": ", p, ", Cost: ", best_costs[i])
    end
    return best_params[1]
end

function rosenbrock2d(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function Optim()
    #
    #df_csv_r = CSV.read("data/params-wc-mir-1.3-0.9-0.9-0.9.csv", DataFrame)
    good_guess = [0.016686, 1.30585, 4.18644, 7.88385, 5.09226, 0.04, 0.005, 0.496913, -0.904573]
    #for i in 1:nrow(df_csv_r)
    #    push!(good_guess, [v for v in values(df_csv_r[i,:])])
    #end
    #  0 theta_I
    #good_guess = [0.016523340716958046,5.432121276855469,3.8098607063293457,6.350203514099121,6.220411777496338,0.5953848361968994]
    p_range=[(0.016, 0.017), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 0.3), (0.0, 0.02), (-2.0, 10.0), (-10.0, 2.0)]

    #Fixed W,beta
    #good_guess = [0.016686, 0.496913, -0.904573]
    #p_range=[(0.016, 0.017), (0.0, 10.0), (-2.0, 10.0)]

    res = bboptimize(cost_bb; SearchRange=p_range)
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

function plot_loss(name)
    df_csv_r = CSV.read("jobs-out/costs-"*name*".csv", DataFrame)
    y = [df_csv_r[!, 1][2]]
    for d in 3:length(df_csv_r[!, 1])
        if df_csv_r[!, 1][d] < y[end]
            push!(y, df_csv_r[!, 1][d])
        end
    end

 
    plot(y)
    savefig("loss.png")
end

#Optim()

#p_range = [(23.09, 23.11), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 10.0), (-2.0, 10.0), (-10.0, 2.0)]
#res = bboptimize(cost_bb_bc; SearchRange=p_range)

function O(P, name, mode)
    p_bounds=[
            (0.016, 0.017),
            (0.0, 10.0),
            (0.0, 10.0),
            (0.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.3),
            (0.0, 0.5),
            (-2.0, 10.0),
            (-10.0, 2.0)
        ]
    if mode == "init"
        init_param(p_bounds, P, name, 100)
    elseif mode == "bp"
        opt_param(P, name, 1)
    elseif mode == "opt"
        good_guess = [0.0167907, 1.84502, 8.10264, 4.90234, 3.76054, 0.0673752, 0.001, 0.275149, -2.27837]
        res = bboptimize(cost_bb, good_guess; SearchRange=p_bounds)
        filename="./jobs-out/"*P*"-"*name*".txt"

        fileID = open(filename, "w")
        println(fileID, best_candidate(res))
        close(fileID)
    end
end

 function main(args)
    O(args[1], args[2], "opt")
 end

 #O("P4", "05_02_2024_P4_Ch14_FRQ=11Hz_FULL_CL_phase=0_REST_EC_v1", "opt")
 #main(ARGS)
 #plot_loss(ARGS[1])