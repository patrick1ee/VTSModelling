import h5py, matplotlib.pyplot as plt

def main():
    filename = "matlab_analyses/data/P2/08_12_2023_P2_Ch14_FRQ=10Hz_FULL_CL_phase=135_STIM_EC_v1.hdf5"

    with h5py.File(filename, "r") as f:
        s = f["EEG"][2][:-7]
        plt.plot(range(len(s)), s)
        plt.show()


        #After you are done
        f.close()

main()