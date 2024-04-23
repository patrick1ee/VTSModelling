from eegvibe import run_tracking, run_replay #, constants

from time import sleep
import numpy


# CODING TASK IDEAS 
#[/] -Improve list of channel names (pass it on from the main function, so that we can modify it during experiment)
#[/] -Print impedance values in colours at the start of each recording (via a user-defined list) to spot any issues with high impedance in channels that are not displayed
#           in read() get_impedance(amplifier_ID), <20kOhm green, <40kOhn is yellow, >40kOhm is red
#[/]   -Listen out for keyboard button presses, to store key events (e.g. talking, etc, "T" or "Q" for quiet, "M" for general movement, "S" for sitting still)
#[/] -Update the x-axis of the plot so it reflects the real time?
#    -Integrate Delsys sensor EMG
#[/] -Is it possible to extend the recording duration "on-the-fly" by 20 seconds with a key press? plus print message that it was added


#    -Improve filename, switch between "FULL_STIM", "NO_STIM", "CONT_STIM", remove "_" at the end
#    -Transfer code to Linux PC and Linux laptop

# - orientation flag


if __name__ == '__main__':
    
    print("STARTING - PLEASE WAIT")
    
    partic_ID   = 1
    
    CHAN_TRACK = 14 # C3 = 14
    #CHAN_TRACK = 29 # O1 = 29
    CHAN_REFS =  28 # POz = 28, P3 = 24 # range(0,31), # take P3 as ref channel (23) or avg ref
    # CHAN_REFS =  [8, 9, 19, 20]  # for avg ref: range(0,31)m P3= 24 
       
    USE_ORIENTATION = 1  # use the orientation sensor, need to open Arduino IDE and adjust the port in orientation.py, and SERIAL_PORT = 'COM7'
        
    CHANS_EMG = [33, 34]
    
    # ============================
    # CHANGE THESE SETTINGS
    TARGET_FREQ = 11
         
    REC_DURATION = 120 # 5*60
    STIM_DURATION = 110
    
    STIM_MODE = 3 #constants.STIM_MODES.CL_STIM # CONT_STIM, NO_STIM, CL_STIM  
    # NO_STIM = 1
    # CONT_STIM = 2
    # CL_STIM = 3
      
    #REC_LABEL = 'EO_BackSpine_O1POz' 
    
    REC_LABEL = 'EC_Wrist_O1POz' 
    REC_LABEL = 'EO_Wrist_C3POz_NO_STIM_recTrigTimes' 
    REC_LABEL = '70Hz_CONT_STIM_WOOJER' 
    # REC_LABEL = 'EO_Hip' 
    
    #phase = numpy.radians(-40)  # the opposite phase of -40 is 140
    phase = numpy.radians(-40)
    #phase = numpy.radians(50)
    #phase = numpy.radians(230)
    #phase = numpy.radians(280)   
       
    # CHANGE PKL FILENAME WHEN REPLAY ON
    REPLAY_ON   =  0 # 0 or 1
    REPLAY_PKL_FILE = './out_data/11_09_2023_P1_Ch14_FRQ=11Hz_FULL_CL_phase=-40EO_Wrist_C3POz_v1.pkl'
    
    
    # ============================
    OT_suppress = 0.3 # the lower the value, the more often we stimulate, but with less accuracy when alpha is low (during eyes open)
    gamma_param = 0.1 # or 0.05
    pulse_duration_s = 0.04 # since moving to PulsePal this is not used anymore to control the duration in Python, but make sure to match param to what is sent from Matlab     
  
    # Show impedances for the folloiwng channels only
    CHAN_PRIORITY = [ 'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 
                  'C4', 'T8',  'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'EMG1', 'EMG2']
    
    
    REC_SR = 1000     # recording sample rate
    
                       # 0      1      2      3     4     5     6     7     8     9      10     11     12     13    14    15      
    FULL_CHAN_NAMES = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 
                  #16    17    18    19    20     21     22     23     24    25    26    27    28    29     30    31    
                  'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 
                  #32    33
                  'EMG1', 'EMG2', 'Acc1', 'Acc2', 'Acc3', 'NA', 
                  'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
    
 
    
    if not REPLAY_ON:    
        run_tracking(freq_sample = REC_SR, freq_target = TARGET_FREQ, phase_target = phase, freq_high_pass = 3, 
            oscilltrack_suppresion = OT_suppress, oscilltrack_gamma = gamma_param,
            stim_mode=STIM_MODE, is_CL_stim=True, N_pulses = 100, pulse_duration = pulse_duration_s, ITI = 0.001, IPI = 0.20, stim_device_ID = 6,  # stim_device_ID is only relevant when using audio output
            channel_track = CHAN_TRACK, channels_ref =  CHAN_REFS, channels_EMG = CHANS_EMG, full_chan_names = FULL_CHAN_NAMES,  
            N_plot_samples = 500, plot_labels = ["Track", "EMG1 ", "EMG2"], plot_signal_range = (-0.0002, 0.0002),
            participant_ID = partic_ID, condition_label = REC_LABEL, recording_duration = REC_DURATION, stim_duration=STIM_DURATION,
            channels_priority_ref = CHAN_PRIORITY, use_orientation=USE_ORIENTATION #,
            #filename_data = './out_data/28_03_2023_P15_Ch14_FRQ=9Hz_FULL_CL_phase=90_v1.hdf5' 
            )
     
    if REPLAY_ON: 
        run_replay(freq_sample = REC_SR, freq_high_pass = 3,
            filename_stim = REPLAY_PKL_FILE,
            channel_track = CHAN_TRACK, channels_ref = CHAN_REFS, channels_EMG = CHANS_EMG, full_chan_names = FULL_CHAN_NAMES,
            N_plot_samples = 500, plot_labels = ["Track", "EMG1", "EMG2"], plot_signal_range = (-0.0002, 0.0002), 
            participant_ID = partic_ID, condition_label = REC_LABEL, channels_priority_ref = CHAN_PRIORITY, use_orientation=USE_ORIENTATION #,
            # filename_data = './out_data/28_03_2023_P15_Ch14_FRQ=9Hz_FULL_CL_phase=90_v1.hdf5'
            )