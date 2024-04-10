using BlackBoxOptim, CSV, DataFrames, DSP, FFTW, KernelDensity, Plots, Statistics

include("./BenoitModel.jl")
using .BenoitModel: create_benoit_model, simulate_benoit_model

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

function get_PSD(E, freqs, verbose)
    psd = fftshift(fft(E .- mean(E) / std(E)))

    if verbose
        plot(freqs, abs.(psd ./ (1.5*10e3)), xlabel="frequency (Hz)", xlim=(0, +100), xticks=0:50:100, yticks=0:0.5:1.6, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        savefig("plots/spec-opt.png")
    end

    return abs.(psd)
end

function get_hilbert_amplitude_pdf(E, bandwidth=0.001)
    hilbert_transform = hilbert(E .- mean(E) / std(E))
    hilbert_amp = abs.(hilbert_transform)

    cleaned_data = replace(hilbert_amp, NaN => 0.0)
    
    # Estimate PDF using kernel density estimation
    try
        U = kde(cleaned_data)
        return U.x, U.density, cleaned_data
    catch err
        println("Error: ", err)
        println("Data: ", cleaned_data)
        return [], [], []
    end

end


function getTrPSD_init(sig, sampling_rate, verbose)
    freqs = fftshift(fftfreq(length(sig), sampling_rate))
    yPSD = fftshift(sig)

    #yPSDmean = mean(yPSD, dims=1)
    yPSDmean = yPSD

    if verbose
        #plot(freqs, yPSDmean, linewidth=3.5, label="average")
        plot(freqs, abs.(yPSD ./ (1.5*10e3)), xlabel="frequency (Hz)", xlim=(0, +10), xticks=0:5:10, yticks=0:0.5:1.6, size=(500,500), linewidth=3, xtickfont=16, ytickfont=16, legend=false, titlefont=16, guidefont=16, tickfont=16, legendfont=16)
        xlabel!("frequency (Hz)")
        ylabel!("PSD")
        savefig("plots/PSD-opt.png")
    end

    return yPSDmean
end

function getTrPSD(sig, fs, trialIdx, freqRange, wLratio, verbose)
    nTr = size(trialIdx, 1)
    yPSD = zeros(nTr, length(freqRange))

    for tr in 1:nTr
        psdIdx = 1#trialIdx[tr, 1]:trialIdx[tr, 2]
        wLength = round(length(sig[psdIdx]) / wLratio)

        yPSDtr  = fftshift(fft(sig[psdIdx]))
        yPSD[tr, :] = yPSDtr
    end

    yPSDmean = mean(yPSD, dims=1)

    if verbose
        plt = plot(freqRange, yPSD', label=[string("trial #", i) for i in 1:nTr], legend=:topright)
        plot!(freqRange, yPSDmean', linewidth=3.5, label="average")
        xlabel!("frequency (Hz)")
        ylabel!("PSD")
        savefig("plots/PSD-opt.png")
    end

    return yPSDmean
end

function cost_bb(params)
    N=2
    W=[Float32(0.0) Float32(1.0); Float32(1.0) Float32(0.0)]
    etta=Float32(0.5)

    tau_E = Float32(params[1])
    tau_I = Float32(params[1])
    w_EE = Float32(params[2])
    w_EI = Float32(params[3])
    w_IE = Float32(params[4])
    beta = Float32(params[5])
    theta_E_param = Float32(params[6])
    theta_I_param = Float32(params[7])

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
    Eproc = df.R[1].rE

    psd_df = CSV.read("data/psd-1.csv", DataFrame)
    xPSD = psd_df[!, 1]
    yPSDdat = psd_df[!, 2]
    yPSDmod = get_PSD(Eproc, xPSD, false)

    hpdf_df = CSV.read("data/hilbert-amp-pdf.csv", DataFrame)
    yHPDFdat = hpdf_df[!, 2]
    _, yHPDFmod, ha = get_hilbert_amplitude_pdf(Eproc)

    #hpsd_df = CSV.read("data/hilbert-psd.csv", DataFrame)
    #yHPSDx = hpsd_df[!, 1]
    #yHPSDdat = hpsd_df[!, 2]

    #F_HA = fftshift(fft(ha))
    #yHPSDmod = abs.(F_HA)

    cost1 = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2)) / 3
    cost2 = (sum((yHPDFdat .- yHPDFmod).^2) / sum((yHPDFdat .- mean(yHPDFdat)).^2)) / 3
    #cost3 = (sum((yHPSDdat .- yHPSDmod).^2) / sum((yHPSDdat .- mean(yHPSDdat)).^2)) / 3

    cost = cost1 + cost2 #+ cost3

    filename="costs.txt"

    if isfile(filename)
        fileID = open(filename, "w")
        println(fileID, tau_E, tau_I, w_EE, w_EI, w_IE, beta, "::", cost1, "::", cost2, "::", cost3)
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

#p_range=[(0.0, 0.5), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (0.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)]
#res = bboptimize(cost_bb; SearchRange = p_range, NumDimensions = 8)