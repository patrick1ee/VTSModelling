using BlackBoxOptim, DSP, Snopt

include("./BenoitModel.jl")
using .BenoitModel: create_benoit_model, simulate_benoit_model


function getTrPSD(sig, fs, trialIdx, freqRange, wLratio, verbose)
    nTr = size(trialIdx, 1)
    yPSD = zeros(nTr, length(freqRange))

    for tr in 1:nTr
        psdIdx = trialIdx[tr, 1]:trialIdx[tr, 2]
        wLength = round(length(sig[psdIdx]) / wLratio)
        yPSDtr = psd(sig[psdIdx], wLength, freqRange, fs)
        yPSD[tr, :] = yPSDtr
    end

    yPSDmean = mean(yPSD, dims=1)

    if verbose
        plt = plot(freqRange, yPSD', label=[string("trial #", i) for i in 1:nTr], legend=:topright)
        plot!(freqRange, yPSDmean', linewidth=3.5, label="average")
        xlabel!("frequency (Hz)")
        ylabel!("PSD")
    end

    return yPSDmean
end

function cost(M, dt, range_t, nTr)

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

    Emean = mean(E[rampUpN:end])
    Estd = std(E[rampUpN:end])
    Eproc = (E .- Emean) / Estd
    
    # FEATURE COMPUTATION
    nPartialTr = 9
    partialIdx = rand(1:nTr, nPartialTr)
    partialTrIdx = trialIdx[partialIdx, :]

    # PSD
    PSDverbose = false
    yPSDmod = getTrPSD(Eproc, 1 / dt, partialTrIdx, dataFeatures.PSD.xPSD, dataFeatures.PSD.wLratio, PSDverbose)
    maxPSD, maxIdx = findmax(yPSDmod)
    fCenter = dataFeatures.PSD.xPSD[maxIdx]

    ##Filter?

    yPSDdat = dataFeatures.PSD.yPSD
    cost1 = (sum((yPSDdat .- yPSDmod).^2) / sum((yPSDdat .- mean(yPSDdat)).^2)) / 1

end

function opt
    
end

function rosenbrock2d(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

res = bboptimize(rosenbrock2d; SearchRange = (-5.0, 5.0), NumDimensions = 2)