include("./BenoitModel.jl")
include("./Signal.jl")

using BlackBoxOptim, CSV, DataFrames, Distributions, DSP, FFTW, KernelDensity, Plots, Statistics

using .BenoitModel: create_benoit_model, simulate_benoit_model
using .Signal: get_pow_spec, get_hilbert_amplitude_pdf

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


params_df = CSV.read("data/params.csv", DataFrame)
function cost_bb(params)

    row = trunc(Int, params[1])

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    tau_E = Float32(params_df[row, 1])
    tau_I = Float32(params_df[row, 1])
    w_EE = Float32(params_df[row, 2])
    w_EI = Float32(params_df[row, 3])
    w_IE = Float32(params_df[row, 4])
    beta = Float32(params_df[row, 5])
    theta_E_param = Float32(params_df[row, 6])
    theta_I_param = Float32(params_df[row, 7])

    #w_EE = Float32(9.09084)
    #w_EI = Float32(24.6724)
    #w_IE = Float32(23.7999)
    #beta = Float32(0.0892136)
    #theta_E_param = Float32(-24.2169)
    #theta_I_param = Float32(9.29577)

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

    theta_E = [theta_E_param, theta_E_param]
    theta_I = [theta_I_param, theta_I_param]
    stim = response
    df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

    #zscore
    Eproc = df.R[1].rE .- mean(df.R[1].rE) / std(df.R[1].rE)

    psd_df = CSV.read("data/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)

    hpdf_df = CSV.read("data/hpdf.csv", DataFrame)
    yHPDFdat = hpdf_df[!, 2]
    _, yHPDFmod, ha = get_hilbert_amplitude_pdf(Eproc)

    #hpsd_df = CSV.read("data/hilbert-psd.csv", DataFrame)
    #yHPSDx = hpsd_df[!, 1]
    #yHPSDdat = hpsd_df[!, 2]

    #F_HA = fftshift(fft(ha))
    #yHPSDmod = abs.(F_HA)

    if length(yPSDmod) > length(yPSDdat)
        yPSDmod = yPSDmod[1:length(yPSDdat)]
    elseif length(yPSDmod) < length(yPSDdat)
        yPSDdat = yPSDdat[1:length(yPSDmod)]
    end

    cost1 = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2)) / 2
    cost2 = (sum((yHPDFdat .- yHPDFmod).^2) / sum((yHPDFdat .- mean(yHPDFdat)).^2)) / 2
    #cost3 = (sum((yHPSDdat .- yHPSDmod).^2) / sum((yHPSDdat .- mean(yHPSDdat)).^2)) / 3

    cost = cost1 + cost2 #+ cost3

    filename="costs.txt"

    if isfile(filename)
        fileID = open(filename, "w")
        println(fileID, tau_E, tau_I, w_EE, w_EI, w_IE, beta, "::", cost1, "::", cost2)
        close(fileID)
    end

    return cost
end

function init_param(bounds, NPARAMS=2500)
    tau_dist = Uniform(bounds[1][1], bounds[1][2])
    w_EE_dist = Uniform(bounds[2][1], bounds[2][2])
    w_EI_dist = Uniform(bounds[3][1], bounds[3][2])
    w_IE_dist = Uniform(bounds[4][1], bounds[4][2])
    beta_dist = Uniform(bounds[5][1], bounds[5][2])
    theta_E_dist = Uniform(bounds[6][1], bounds[6][2])
    theta_I_dist = Uniform(bounds[7][1], bounds[7][2])

    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.50)

    psd_df = CSV.read("data/psd.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    peakDat = argmax(yPSDdat)

    csv_df_w = DataFrame(tau=[0.0], w_EE=[0.0], w_EI=[0.0], w_IE=[0.0], beta=[0.0], theta_E=[0.0], theta_I=[0.0])
    CSV.write("data/params.csv", csv_df_w)

    count = 0

    while count < NPARAMS
        tau_p = Float32(rand(tau_dist))
        w_EE_p = Float32(rand(w_EE_dist))
        w_EI_p = Float32(rand(w_EI_dist))
        w_IE_p = Float32(rand(w_IE_dist))
        beta_p = Float32(rand(beta_dist))
        theta_E_p = Float32(rand(theta_E_dist))
        theta_I_p = Float32(rand(theta_I_dist))

        model = create_benoit_model(N, W, etta, tau_p, tau_p, w_EE_p, w_EI_p, w_IE_p, beta_p)
        T = 10.0
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
        theta_E = [theta_E_p, theta_E_p]
        theta_I = [theta_I_p, theta_I_p]
        stim = response
        df = run_act_time(model, simulate_benoit_model, range_t, dt, theta_E, theta_I, stim)

        #zscore
        Eproc = df.R[1].rE .- mean(df.R[1].rE) / std(df.R[1].rE)

        freq, yPSDmod = get_pow_spec(Eproc, xPSD, SR)
        peakMod = argmax(yPSDmod)
        #print(string(freq[peakMod])*" "*string(xPSD[peakDat])*"\n")

        if abs(xPSD[peakDat] - freq[peakMod]) <= 1.0 && abs(yPSDmod[peakMod] - yPSDdat[peakDat]) <= 0.25*yPSDdat[peakDat]
            new_row = (tau=tau_p, w_EE=w_EE_p, w_EI=w_EI_p, w_IE=w_IE_p, beta=beta_p, theta_E=theta_E_p, theta_I=theta_I_p)
            df_csv_r = CSV.read("data/params.csv", DataFrame)
            push!(df_csv_r, new_row)
            CSV.write("data/params.csv", df_csv_r)
            count += 1
            print("Added " * string(count) * " / 2500 parameters\n")
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
    cost = cost1

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
    best_param = 1
    best_cost = 1000.0
    for i in 1:2500
        cost = cost_bb([i])
        if cost < best_cost
            best_cost = cost
            best_param = i
            println("Best cost: ", best_cost, " Best param: ", best_param)
        end
    end
end

function rosenbrock2d(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

#p_range=[[0.0, 0.1, 0.6], (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)]
#p_range = [(0.028, 0.030 ), (7.0, 9.0), (18.0, 20.0), (26.0, 28.0), (18.0, 20.0), (8.0, 9.0), (-20.0, -18.0)]
#p_range=[(1.0, 2501.0)]
#res = bboptimize(cost_bb, SearchRange=p_range)

p_bounds = [(0.016, 0.017), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-2.0, 10.0), (-10.0, 2.0)]
init_param(p_bounds)
opt_param()

