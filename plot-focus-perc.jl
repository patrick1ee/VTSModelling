using ApproxFun, Base.Filesystem, CSV, DataFrames, DSP, FFTW, HDF5, Interpolations, KernelDensity, KissSmoothing, LPVSpectral, LsqFit, Measures, NaNStatistics, Plots, StatsBase, StatsPlots, Statistics

function plot_focussed_plv()
    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_REST_EC_v1"
    df_plv_data_rest = CSV.read(csv_data_path*"/plvs.csv", DataFrame)   
    
    #0vs180
    p1 = plot(
        df_plv_data_rest[!, 1],
        df_plv_data_rest[!, 2], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="0 (o) vs. 180 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )

    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=0_STIM_EC_v1"
    df_plv_data_stim_0 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p1, df_plv_data_stim_0[!, 1], df_plv_data_stim_0[!, 2], linewidth=3)
    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=180_STIM_EC_v1"
    df_plv_data_stim_180 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p1, df_plv_data_stim_180[!, 1], df_plv_data_stim_180[!, 2], linewidth=3)

    #45vs225
    p2 = plot(
        df_plv_data_rest[!, 1],
        df_plv_data_rest[!, 2], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="45 (o) vs. 225 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )

    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=45_STIM_EC_v1"
    df_plv_data_stim_45 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p2, df_plv_data_stim_45[!, 1], df_plv_data_stim_45[!, 2], linewidth=3)
    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=225_STIM_EC_v1"
    df_plv_data_stim_225 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p2, df_plv_data_stim_225[!, 1], df_plv_data_stim_225[!, 2], linewidth=3)

    #90vs270
    p3 = plot(
        df_plv_data_rest[!, 1],
        df_plv_data_rest[!, 2], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="90 (o) vs. 270 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )

    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=90_STIM_EC_v1"
    df_plv_data_stim_90 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p3, df_plv_data_stim_90[!, 1], df_plv_data_stim_90[!, 2], linewidth=3)
    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=270_STIM_EC_v1"
    df_plv_data_stim_270 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p3, df_plv_data_stim_270[!, 1], df_plv_data_stim_270[!, 2], linewidth=3)

    #135vs315
    p4 = plot(
        df_plv_data_rest[!, 1],
        df_plv_data_rest[!, 2], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="135 (o) vs. 315 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )

    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=135_STIM_EC_v1"
    df_plv_data_stim_135 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p4, df_plv_data_stim_135[!, 1], df_plv_data_stim_90[!, 2], linewidth=3)
    csv_data_path = "data/P20/15_02_2024_P20_Ch14_FRQ=10Hz_FULL_CL_phase=270_STIM_EC_v1"
    df_plv_data_stim_315 = CSV.read(csv_data_path*"/plvs.csv", DataFrame)  
    plot!(p4, df_plv_data_stim_315[!, 1], df_plv_data_stim_315[!, 2], linewidth=3)

    plot(p1, p2, p3, p4, layout=grid(2, 2))
    savefig("focussed-plv-data.png")

    yPSDs_Rest = [[] for i in 1:100]
    xPSDs_Rest = []
    for i in 1:100
        csv_path = "data/model/"*string(i)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_Rest = plv_df[!, 1]
        yPSDdat = plv_df[!, 2]
        yPSDs_Rest[i]=  yPSDdat
    end

    yPSDs_0 = [[] for i in 1:25]
    xPSDs_0 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-0/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_0 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_0[j]=  yPSDdat_A
    end

    yPSDs_180 = [[] for i in 1:25]
    xPSDs_180 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-180/"*string(j)
        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_180 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_180[j]=  yPSDdat_A
    end

    yPSDs_45 = [[] for i in 1:25]
    xPSDs_45 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-45/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_45 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_45[j]=  yPSDdat_A
    end

    yPSDs_225 = [[] for i in 1:25]
    xPSDs_225 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-225/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_225 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_225[j]=  yPSDdat_A
    end

    yPSDs_90 = [[] for i in 1:25]
    xPSDs_90 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-90/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_90 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_90[j]=  yPSDdat_A
    end

    yPSDs_270 = [[] for i in 1:25]
    xPSDs_270 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-270/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_270 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_270[j]=  yPSDdat_A
    end

    yPSDs_135 = [[] for i in 1:25]
    xPSDs_135 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-135/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_135 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_135[j]=  yPSDdat_A
    end

    yPSDs_315 = [[] for i in 1:25]
    xPSDs_315 = []
    for j in 1:25
        csv_path = "data/model-wc-plus-stim-resp-315/"*string(j)

        plv_df = CSV.read(csv_path*"/plvs.csv", DataFrame)
        xPSDs_315 = plv_df[!, 1]
        yPSDdat_A = plv_df[!, 2]
        yPSDs_315[j]=  yPSDdat_A
    end

    #0vs180
    p1 = plot(
        xPSDs_Rest,
        mean(yPSDs_Rest, dims=1),
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="0 (o) vs. 180 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )
    plot!(p1, xPSDs_0, mean(yPSDs_0, dims=1), linewidth=3)
    plot!(p1, xPSDs_180, mean(yPSDs_180, dims=1), linewidth=3)
    
    #45vs225
    p2 = plot(
        xPSDs_Rest,
        mean(yPSDs_Rest, dims=1),
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="45 (o) vs. 225 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )
    plot!(p2, xPSDs_45, mean(yPSDs_45, dims=1), linewidth=3)
    plot!(p2, xPSDs_225, mean(yPSDs_225, dims=1), linewidth=3)

    #90vs270
    p3 = plot(
        xPSDs_Rest,
        mean(yPSDs_Rest, dims=1),
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="90 (o) vs. 270 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )
    plot!(p3, xPSDs_90, mean(yPSDs_90, dims=1), linewidth=3)
    plot!(p3, xPSDs_270, mean(yPSDs_270, dims=1), linewidth=3)

    #135vs315
    p4 = plot(
        xPSDs_Rest,
        mean(yPSDs_Rest, dims=1),
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="135 (o) vs. 315 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=3,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm
    )
    plot!(p4, xPSDs_135, mean(yPSDs_135, dims=1), linewidth=3)
    plot!(p4, xPSDs_315, mean(yPSDs_315, dims=1), linewidth=3)

    plot(p1, p2, p3, p4, layout=grid(2, 2))
    savefig("focussed-plv-model.png")

    ##SUBTRACT

    #0vs180
    p1 = plot(
        df_plv_data_rest[!, 1],
        [0.0 for i in df_plv_data_rest[!, 2]], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="0 (o) vs. 180 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p1, df_plv_data_stim_0[!, 1], (df_plv_data_stim_0[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)
    plot!(p1, df_plv_data_stim_180[!, 1], (df_plv_data_stim_180[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)

    #45vs225
    p2 = plot(
        df_plv_data_rest[!, 1],
        [0.0 for i in df_plv_data_rest[!, 2]], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="45 (o) vs. 225 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p2, df_plv_data_stim_45[!, 1], (df_plv_data_stim_45[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)
    plot!(p2, df_plv_data_stim_225[!, 1], (df_plv_data_stim_225[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)

    #90vs270
    p3 = plot(
        df_plv_data_rest[!, 1],
        [0.0 for i in df_plv_data_rest[!, 2]], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="90 (o) vs. 270 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p3, df_plv_data_stim_90[!, 1], (df_plv_data_stim_90[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)
    plot!(p3, df_plv_data_stim_270[!, 1], (df_plv_data_stim_270[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)

    #135vs315
    p4 = plot(
        df_plv_data_rest[!, 1],
        [0.0 for i in df_plv_data_rest[!, 2]], 
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="135 (o) vs. 315 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p4, df_plv_data_stim_135[!, 1], (df_plv_data_stim_135[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)
    plot!(p4, df_plv_data_stim_315[!, 1], (df_plv_data_stim_315[!, 2] .- df_plv_data_rest[!, 2]) ./ df_plv_data_rest[!, 2], linewidth=3)

    plot(p1, p2, p3, p4, layout=grid(2, 2))
    savefig("focussed-plv-diff.png")

    #SUBTRACT MODEL
    p1 = plot(
        xPSDs_Rest,
        [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="0 (o) vs. 180 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p1, xPSDs_0, (mean(yPSDs_0, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)
    plot!(p1, xPSDs_180, (mean(yPSDs_180, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)

    p2 = plot(
        xPSDs_Rest,
        [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="45 (o) vs. 225 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p2, xPSDs_45, (mean(yPSDs_45, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)
    plot!(p2, xPSDs_225, (mean(yPSDs_225, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)

    p3 = plot(
        xPSDs_Rest,
        [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="90 (o) vs. 270 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p3, xPSDs_90, (mean(yPSDs_90, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)
    plot!(p3, xPSDs_270, (mean(yPSDs_270, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)

    p4 = plot(
        xPSDs_Rest,
        [0.0 for i in mean(yPSDs_Rest, dims=1)[1]],
        xlabel="Frequency (Hz)",
        ylabel="PLV Change (%)",
        title="135 (o) vs. 315 (g)",
        legend=false,
        size=(600, 500),
        xlim=(6, 30),
        xticks=6:4:30,
        linewidth=1.5,
        linestyle=:dash,
        c="black",
        xtickfont=12,
        ytickfont=12,
        titlefont=12,
        guidefont=12,
        tickfont=12,
        legendfont=12,
        margin=2.5mm,
        yformatter = x->x*100
    )
    plot!(p4, xPSDs_135, (mean(yPSDs_135, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)
    plot!(p4, xPSDs_315, (mean(yPSDs_315, dims=1)[1] .- mean(yPSDs_Rest, dims=1)[1]) ./ mean(yPSDs_Rest, dims=1)[1], linewidth=3)

    plot(p1, p2, p3, p4, layout=grid(2, 2))
    savefig("focussed-plv-diff-model.png")

    
end

plot_focussed_plv()