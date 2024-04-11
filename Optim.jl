include("./BenoitModel.jl")
include("./Signal.jl")

using BlackBoxOptim, CSV, DataFrames, DSP, FFTW, KernelDensity, Plots, Statistics

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

function cost_bb(params)
    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    tau_E = Float32(params[1])
    tau_I = Float32(params[1])

    #w_EE = Float32(params[2])
    #w_EI = Float32(params[3])
    #w_IE = Float32(params[4])
    #beta = Float32(params[5])
    #theta_E_param = Float32(params[6])
    #theta_I_param = Float32(params[7])

    w_EE = Float32(9.09084)
    w_EI = Float32(24.6724)
    w_IE = Float32(23.7999)
    beta = Float32(0.0892136)
    theta_E_param = Float32(-24.2169)
    theta_I_param = Float32(9.29577)

    model = create_benoit_model(N, W, etta, tau_E, tau_I, w_EE, w_EI, w_IE, beta)
    
    T = 100.0
    dt = 0.001
    range_t = 0.0:dt:T

    response = fill(0.0, length(range_t)) 
    for i in 1:6:T-6
        #Start pulse
        for j in 0:24
            for k in 0:2:10
                response[Int64(trunc(i*1000+j*200+k*(1000/130)))] = 0.001684
            end
        end
    end
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

function opt()
    tau_E = Float32(0.29984)
    tau_I = Float32(0.29984)
    w_EE = Float32(1.548)
    w_EI = Float32(25.3384)
    w_IE = Float32(26.048)
    beta = Float32(2.4234)

    #tau_E = Float32(0.0758)
    #tau_I = Float32(0.0758)
    #w_EE = Float32(6.7541)
    #w_EI = Float32(9.6306)
    #w_IE = Float32(9.4014)
    #beta = Float32(1.1853)

    #print(cost_bb(tau_E, tau_I, w_EE, w_EI, w_IE, beta, 1.4240, -3.2345))
    print(cost_bb([tau_E, tau_I, w_EE, w_EI, w_IE, beta, 22.8621, -9.9279]))
end

function rosenbrock2d(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

#p_range=[(0.0, 0.1), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)]
#p_range=[(0.0, 0.1)]
#res = bboptimize(cost_bb; SearchRange = p_range)