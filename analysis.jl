using HDF5, Plots

fid = h5open("matlab_analyses/data/P2/08_12_2023_P2_Ch14_FRQ=10Hz_FULL_CL_phase=0_NOSTIM_EC_v2.hdf5", "r")
A = read(fid["EEG"])

plot(A[4,10:78000], xlabel="t", ylabel="EEG")
savefig("plots/myplot.png")

close(fid)