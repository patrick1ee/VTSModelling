import numpy as np
import matplotlib.pyplot as plt
from Receptors import tactile_receptors, sin_wave


def main():
    tsensor=tactile_receptors(Ttype='SA1',simTime=1.0,sample_rate=3000,sample_num=4000)

    intentation=100*1e-6 #um

    stimulus=np.zeros((1,tsensor.t.size))
    stimulus[0, :] = sin_wave(tsensor.t, 2*10*np.pi, intentation)
    tsensor.tactile_units_simulating(stimulus)

    print(len(stimulus[0]))
    print(len(tsensor.Va[0]))
    print(np.mean(tsensor.Va, axis=0).shape)
    print(tsensor.Va[0][0])
    print(tsensor.Va[24][0])

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.plot(tsensor.t, stimulus[0])

    ax1.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax1.grid()

    tsensor.Va[0] = (tsensor.Va[0] - np.mean(tsensor.Va[0], axis=0)) / np.std(tsensor.Va[0], axis=0)

    ax2.plot(tsensor.t, tsensor.Va[0])

    ax2.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax2.grid()

    ax3.plot(tsensor.t, np.mean(tsensor.Va, axis=0))

    ax3.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax3.grid()

    fig.savefig("test.png")
    plt.show()

main()