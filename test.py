import numpy as np
import matplotlib.pyplot as plt
from Receptors import tactile_receptors, sin_wave


def main():
    tsensor=tactile_receptors(Ttype='PC',simTime=1.0,sample_rate=3000,sample_num=4000)

    intentation=100*1e-6 #um

    stimulus=np.zeros((1,tsensor.t.size))
    stimulus[0, :] = sin_wave(tsensor.t, 2*10*np.pi, intentation)
    tsensor.tactile_units_simulating(stimulus)

    print(len(stimulus[0]))
    print(len(tsensor.Va[0]))

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(tsensor.t, stimulus[0])

    ax1.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax1.grid()

    ax2.plot(tsensor.t, tsensor.Va[0])

    ax2.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax2.grid()

    fig.savefig("test.png")
    plt.show()

main()