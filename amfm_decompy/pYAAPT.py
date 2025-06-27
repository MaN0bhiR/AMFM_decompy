# -*- coding: utf-8 -*-
"""
This a ported version for Python from the YAAPT algorithm. The original MATLAB
program was written by Hongbing Hu and Stephen A.Zahorian.

The YAAPT program, designed for fundamental frequency tracking,
is extremely robust for both high quality and telephone speech.

The YAAPT program was created by the Speech Communication Laboratory of
the state university of New York at Binghamton. The original program is
available at http://www.ws.binghamton.edu/zahorian as free software. Further
information about the program could be found at Stephen A. Zahorian, and
Hongbing Hu, "A spectral/temporal method for robust fundamental frequency
tracking," J. Acoust. Soc. Am. 123(6), June 2008.

It must be noticed that, although this ported version is almost equal to the
original, some few changes were made in order to make the program more "pythonic"
and improve its performance. Nevertheless, the results obtained with both
algorithms were similar.

USAGE:
    pitch = yaapt(signal, <options>)

INPUTS:
    signal: signal object created by amfm_decompy.basic_tools. For more
    information about its properties, please consult the documentation file.

    <options>: must be formated as follows:
               **{'option_name1' : value1, 'option_name2' : value2, ...}
               The default configuration values for all of them are the same as
               in the original version. The main yaapt function in this file
               provides a short description about each option.
               For more information, please refer to the original bibliography.

OUTPUTS:
    pitch: pitch object. For more information about its properties, please
           consult the documentation file.

Version 1.0.12
16/May/2025 Bernardo J.B. Schmitt - bernardo.jb.schmitt@gmail.com
"""

import numpy as np
import numpy.lib.stride_tricks as stride_tricks
from scipy.signal import firwin, medfilt, lfilter
from scipy.signal.windows import hann, kaiser
import scipy.interpolate as scipy_interp
import os

import amfm_decompy.basic_tools as basic

# Try to import CUDA utilities
try:
    from .yaapT_cuda import (
        CUDA_AVAILABLE, 
        get_cuda_info,
        cuda_rfft,
        cuda_compute_shc,
        cuda_compute_nccf,
        cuda_dynamic_programming
    )
    _has_cuda = True
except ImportError:
    _has_cuda = False
    CUDA_AVAILABLE = False

"""
--------------------------------------------
                Classes.
--------------------------------------------
"""
"""
Auxiliary class to handle the class properties.
"""
class ClassProperty(object):

    def __init__(self, initval=None):
        self.val = initval

    def __get__(self, obj, objtype):
        return self.val

    def __set__(self, obj, val):
        self.val = val


"""
Creates a pitch object.
"""
class PitchObj(object):

    PITCH_HALF = ClassProperty(0)
    PITCH_HALF_SENS = ClassProperty(2.9)
    PITCH_DOUBLE = ClassProperty(0)
    PITCH_DOUBLE_SENS = ClassProperty(2.9)
    SMOOTH_FACTOR = ClassProperty(5)
    SMOOTH = ClassProperty(5)
    PTCH_TYP = ClassProperty(100.0)

    def __init__(self, frame_size, frame_jump, nfft=8192):
        self.nfft = nfft
        self.frame_size = frame_size
        self.frame_jump = frame_jump
        self.noverlap = self.frame_size-self.frame_jump
        self.use_cuda = CUDA_AVAILABLE and os.environ.get('AMFM_USE_CUDA', '1') == '1'

    def set_energy(self, energy, threshold):
        self.mean_energy = np.mean(energy)
        self.energy = energy/self.mean_energy
        self.vuv = (self.energy > threshold)

    def set_frames_pos(self, frames_pos):
        self.frames_pos = frames_pos
        self.nframes = len(self.frames_pos)

    def set_values(self, samp_values, file_size, interp_tech='pchip'):
        self.samp_values = samp_values
        self.fix()
        self.values = self.upsample(self.samp_values, file_size, 0, 0,
                                    interp_tech)
        self.edges = self.edges_finder(self.values)
        self.interpolate()
        self.values_interp = self.upsample(self.samp_interp, file_size,
                                           self.samp_interp[0],
                                           self.samp_interp[-1], interp_tech)

    """
    For the voiced/unvoiced version of the pitch data, finds the n samples where
    the transitions between these two states occur.
    """
    def edges_finder(self, values):
        vec1 = (np.abs(values[1:]+values[:-1]) > 0)
        vec2 = (np.abs(values[1:]*values[:-1]) == 0)
        edges = np.logical_and(vec1, vec2)
        # The previous logical operation detects where voiced/unvoiced transitions
        # occur. Thus, a 'True' in the edges[n] sample indicates that the sample
        # value[n+1] has a different state than value[n](i.e. if values[n] is
        # voiced, then values[n+1] is unvoiced - and vice-versa). Consequently,
        # the last sample from edges array will always be 'False' and is not
        # calculated (because "there is no n+1 sample" for it. That's why
        # len(edges) = len(values)-1). However, just for sake of comprehension
        # (and also to avoid python warnings about array length mismatchs), I
        # add a 'False' to edges the array. But in pratice, this 'False' is
        # useless.
        edges = np.append(edges,[False])
        index = np.arange(len(values))
        index = index[edges > 0]
        return index.tolist()

    """
    This method corresponds to the first half of the ptch_fix.m file. It tries
    to fix half pitch and double pitch errors.
    """
    def fix(self):
        if self.PITCH_HALF > 0:
            nz_pitch = self.samp_values[self.samp_values > 0]
            idx = self.samp_values < (np.mean(nz_pitch)-self.PITCH_HALF_SENS *
                                      np.std(nz_pitch))
            if self.PITCH_HALF == 1:
                self.samp_values[idx] = 0
            elif self.PITCH_HALF == 2:
                self.samp_values[idx] = 2*self.samp_values[idx]

        if self.PITCH_DOUBLE > 0:
            nz_pitch = self.samp_values[self.samp_values > 0]
            idx = self.samp_values > (np.mean(nz_pitch)+self.PITCH_DOUBLE_SENS *
                                      np.std(nz_pitch))
            if self.PITCH_DOUBLE == 1:
                self.samp_values[idx] = 0
            elif self.PITCH_DOUBLE == 2:
                self.samp_values[idx] = 0.5*self.samp_values[idx]

    """
    Corresponds to the second half of the ptch_fix.m file. Creates the
    interpolated pitch data.
    """
    def interpolate(self):
        pitch = np.zeros((self.nframes))
        pitch[:] = self.samp_values
        pitch2 = medfilt(self.samp_values, self.SMOOTH_FACTOR)

        # This part in the original code is kind of confused and caused
        # some problems with the extrapolated points before the first
        # voiced frame and after the last voiced frame. So, I made some
        # small modifications in order to make it work better.
        edges = self.edges_finder(pitch)
        first_sample = pitch[0]
        last_sample = pitch[-1]

        if len(np.nonzero(pitch2)[0]) < 2:
            pitch[pitch == 0] = self.PTCH_TYP
        else:
            nz_pitch = pitch2[pitch2 > 0]
            pitch2 = scipy_interp.pchip(np.nonzero(pitch2)[0],
                                        nz_pitch)(range(self.nframes))
            pitch[pitch == 0] = pitch2[pitch == 0]
        if self.SMOOTH > 0:
            pitch = medfilt(pitch, self.SMOOTH_FACTOR)
        try:
            if first_sample == 0:
                pitch[:edges[0]-1] = pitch[edges[0]]
            if last_sample == 0:
                pitch[edges[-1]+1:] = pitch[edges[-1]]
        except:
            pass
        self.samp_interp = pitch

    """
    Upsample the pitch data so that its length becomes the same as the speech
    signal.
    """
    def upsample(self, samp_values, file_size, first_samp=0, last_samp=0,
                 interp_tech='pchip'):
        if interp_tech == 'step':
            beg_pad = int((self.noverlap)/2)
            up_version = np.zeros((file_size))
            up_version[:beg_pad] = first_samp
            up_version[beg_pad:beg_pad+self.frame_jump*self.nframes] = \
                                    np.repeat(samp_values, self.frame_jump)
            up_version[beg_pad+self.frame_jump*self.nframes:] = last_samp

        elif interp_tech in ['pchip', 'spline']:
            if np.amin(samp_values) > 0:
                if interp_tech == 'pchip':
                    up_version = scipy_interp.pchip(self.frames_pos,
                                                    samp_values)(range(file_size))

                elif interp_tech == 'spline':
                    tck, u_original = scipy_interp.splprep(
                                                [self.frames_pos, samp_values],
                                                u=self.frames_pos)
                    up_version = scipy_interp.splev(range(file_size), tck)[1]
            else:
                beg_pad = int((self.noverlap)/2)
                up_version = np.zeros((file_size))
                up_version[:beg_pad] = first_samp
                voiced_frames = np.nonzero(samp_values)[0]

                if len(voiced_frames) > 0:
                    edges = np.nonzero((voiced_frames[1:]-voiced_frames[:-1]) > 1)[0]
                    edges = np.insert(edges, len(edges), len(voiced_frames)-1)
                    voiced_frames = np.split(voiced_frames, edges+1)[:-1]

                for frame in voiced_frames:
                    up_interval = self.frames_pos[frame]
                    tot_interval = np.arange(int(up_interval[0]-(self.frame_jump/2)),
                                          int(up_interval[-1]+(self.frame_jump/2)))

                    if interp_tech == 'pchip' and len(frame) > 2:
                        up_version[tot_interval] = scipy_interp.pchip(
                                                    up_interval,
                                                    samp_values[frame])(tot_interval)

                    elif interp_tech == 'spline' and len(frame) > 3:
                        tck, u_original = scipy_interp.splprep(
                                            [up_interval, samp_values[frame]],
                                             u=up_interval)
                        up_version[tot_interval] = scipy_interp.splev(tot_interval, tck)[1]

                    # In case len(frame)==2, above methods fail.
                    # Therefore, linear interpolation is used instead.
                    elif len(frame) > 1:
                        up_version[tot_interval] = scipy_interp.interp1d(
                                                    up_interval,
                                                    samp_values[frame],
                                        fill_value='extrapolate')(tot_interval)

                    elif len(frame) == 1:
                        up_version[tot_interval] = samp_values[frame]


                up_version[beg_pad+self.frame_jump*self.nframes:] = last_samp

        return up_version

"""
Creates a bandpass filter object.
"""
class BandpassFilter(object):

    def __init__(self, fs, parameters):

        fs_min = 1000.0
        if (fs > fs_min):
            dec_factor = parameters['dec_factor']
        else:
            dec_factor = 1

        filter_order = parameters['bp_forder']
        f_hp = parameters['bp_low']
        f_lp = parameters['bp_high']

        f1 = f_hp/(fs/2)
        f2 = f_lp/(fs/2)

        self.b = firwin(filter_order+1, [f1, f2], pass_zero=False)
        self.a = 1
        self.dec_factor = dec_factor


"""
--------------------------------------------
                Main function.
--------------------------------------------
"""
def yaapt(signal, **kwargs):

    # Rename the YAAPT v4.0 parameter "frame_lengtht" to "tda_frame_length"
    # (if provided).
    if 'frame_lengtht' in kwargs:
        if 'tda_frame_length' in kwargs:
            warning_str = 'WARNING: Both "tda_frame_length" and "frame_lengtht" '
            warning_str += 'refer to the same parameter. Therefore, the value '
            warning_str += 'of "frame_lengtht" is going to be discarded.'
            print(warning_str)
        else:
            kwargs['tda_frame_length'] = kwargs.pop('frame_lengtht')

    #---------------------------------------------------------------
    # Set the default values for the parameters.
    #---------------------------------------------------------------
    parameters = {}
    parameters['frame_length'] = kwargs.get('frame_length', 35.0)   #Length of each analysis frame (ms)
    # WARNING: In the original MATLAB YAAPT 4.0 code the next parameter is called
    # "frame_lengtht" which is quite similar to the previous one "frame_length".
    # Therefore, I've decided to rename it to "tda_frame_length" in order to
    # avoid confusion between them. Nevertheless, both inputs ("frame_lengtht"
    # and "tda_frame_length") are accepted when the function is called.
    parameters['tda_frame_length'] = \
                              kwargs.get('tda_frame_length', 35.0)  #Frame length employed in the time domain analysis (ms)
    parameters['frame_space'] = kwargs.get('frame_space', 10.0)     #Spacing between analysis frames (ms)
    parameters['f0_min'] = kwargs.get('f0_min', 60.0)               #Minimum F0 searched (Hz)
    parameters['f0_max'] = kwargs.get('f0_max', 400.0)              #Maximum F0 searched (Hz)
    parameters['fft_length'] = kwargs.get('fft_length', 8192)       #FFT length
    parameters['bp_forder'] = kwargs.get('bp_forder', 150)          #Order of band-pass filter
    parameters['bp_low'] = kwargs.get('bp_low', 50.0)               #Low frequency of filter passband (Hz)
    parameters['bp_high'] = kwargs.get('bp_high', 1500.0)           #High frequency of filter passband (Hz)
    parameters['nlfer_thresh1'] = kwargs.get('nlfer_thresh1', 0.75) #NLFER boundary for voiced/unvoiced decisions
    parameters['nlfer_thresh2'] = kwargs.get('nlfer_thresh2', 0.1)  #Threshold for NLFER definitely unvoiced
    parameters['shc_numharms'] = kwargs.get('shc_numharms', 3)      #Number of harmonics in SHC calculation
    parameters['shc_window'] = kwargs.get('shc_window', 40.0)       #SHC window length (Hz)
    parameters['shc_maxpeaks'] = kwargs.get('shc_maxpeaks', 4)      #Maximum number of SHC peaks to be found
    parameters['shc_pwidth'] = kwargs.get('shc_pwidth', 50.0)       #Window width in SHC peak picking (Hz)
    parameters['shc_thresh1'] = kwargs.get('shc_thresh1', 5.0)      #Threshold 1 for SHC peak picking
    parameters['shc_thresh2'] = kwargs.get('shc_thresh2', 1.25)     #Threshold 2 for SHC peak picking
    parameters['f0_double'] = kwargs.get('f0_double', 150.0)        #F0 doubling decision threshold (Hz)
    parameters['f0_half'] = kwargs.get('f0_half', 150.0)            #F0 halving decision threshold (Hz)
    parameters['dp5_k1'] = kwargs.get('dp5_k1', 11.0)               #Weight used in dynamic program
    parameters['dec_factor'] = kwargs.get('dec_factor', 1)          #Factor for signal resampling
    parameters['nccf_thresh1'] = kwargs.get('nccf_thresh1', 0.3)    #Threshold for considering a peak in NCCF
    parameters['nccf_thresh2'] = kwargs.get('nccf_thresh2', 0.9)    #Threshold for terminating serach in NCCF
    parameters['nccf_maxcands'] = kwargs.get('nccf_maxcands', 3)    #Maximum number of candidates found
    parameters['nccf_pwidth'] = kwargs.get('nccf_pwidth', 5)        #Window width in NCCF peak picking
    parameters['merit_boost'] = kwargs.get('merit_boost', 0.20)     #Boost merit
    parameters['merit_pivot'] = kwargs.get('merit_pivot', 0.99)     #Merit assigned to unvoiced candidates in
                                                                    #defintely unvoiced frames
    parameters['merit_extra'] = kwargs.get('merit_extra', 0.4)      #Merit assigned to extra candidates
                                                                    #in reducing F0 doubling/halving errors
    parameters['median_value'] = kwargs.get('median_value', 7)      #Order of medial filter
    parameters['dp_w1'] = kwargs.get('dp_w1', 0.15)                 #DP weight factor for V-V transitions
    parameters['dp_w2'] = kwargs.get('dp_w2', 0.5)                  #DP weight factor for V-UV or UV-V transitions
    parameters['dp_w3'] = kwargs.get('dp_w3', 0.1)                  #DP weight factor of UV-UV transitions
    parameters['dp_w4'] = kwargs.get('dp_w4', 0.9)                  #Weight factor for local costs

    # Exclusive from pYAAPT.

    parameters['spec_pitch_min_std'] = kwargs.get('spec_pitch_min_std', 0.05)
                                                                    #Weight factor that sets a minimum
                                                                    #spectral pitch standard deviation,
                                                                    #which is calculated as
                                                                    #min_std = pitch_avg*spec_pitch_min_std

    # Enable CUDA usage
    parameters['use_cuda'] = kwargs.get('use_cuda', True)           #Use CUDA acceleration if available

    #---------------------------------------------------------------
    # Create the signal objects and filter them.
    #---------------------------------------------------------------
    fir_filter = BandpassFilter(signal.fs, parameters)
    nonlinear_sign = basic.SignalObj(signal.data**2, signal.fs)

    signal.filtered_version(fir_filter)
    nonlinear_sign.filtered_version(fir_filter)

    #---------------------------------------------------------------
    # Create the pitch object.
    #---------------------------------------------------------------
    nfft = parameters['fft_length']
    frame_size = int(np.fix(parameters['frame_length']*signal.fs/1000))
    frame_jump = int(np.fix(parameters['frame_space']*signal.fs/1000))
    pitch = PitchObj(frame_size, frame_jump, nfft)

    assert pitch.frame_size > 15, 'Frame length value {} is too short.'.format(pitch.frame_size)
    assert pitch.frame_size < 2048, 'Frame length value {} exceeds the limit.'.format(pitch.frame_size)

    # Check CUDA availability and configure usage
    use_cuda = parameters['use_cuda'] and CUDA_AVAILABLE and _has_cuda
    if use_cuda:
        # Attempt to enable CUDA
        os.environ['AMFM_USE_CUDA'] = '1'
    else:
        # Disable CUDA
        os.environ['AMFM_USE_CUDA'] = '0'
    
    # Update pitch object with CUDA settings
    pitch.use_cuda = use_cuda

    #---------------------------------------------------------------
    # Calculate NLFER and determine voiced/unvoiced frames.
    #---------------------------------------------------------------
    nlfer(signal, pitch, parameters)

    #---------------------------------------------------------------
    # Calculate an approximate pitch track from the spectrum.
    #---------------------------------------------------------------
    spec_pitch, pitch_std = spec_track(nonlinear_sign, pitch, parameters)

    #---------------------------------------------------------------
    # Temporal pitch tracking based on NCCF.
    #---------------------------------------------------------------
    time_pitch1, time_merit1 = time_track(signal, spec_pitch, pitch_std, pitch,
                                          parameters)

    time_pitch2, time_merit2 = time_track(nonlinear_sign, spec_pitch, pitch_std,
                                          pitch, parameters)

    # Added in YAAPT 4.0
    if time_pitch1.shape[1] < len(spec_pitch):
        time_pitch_dummy = np.zeros((2, len(spec_pitch)))
        time_pitch_dummy[:, :time_pitch1.shape[1]] = time_pitch1[:, :]
        time_pitch1 = time_pitch_dummy

        time_pitch_dummy = np.zeros((2, len(spec_pitch)))
        time_pitch_dummy[:, :time_pitch2.shape[1]] = time_pitch2[:, :]
        time_pitch2 = time_pitch_dummy

        time_merit_dummy = np.zeros((2, len(spec_pitch)))
        time_merit_dummy[:, :time_merit1.shape[1]] = time_merit1[:, :]
        time_merit1 = time_merit_dummy

        time_merit_dummy = np.zeros((2, len(spec_pitch)))
        time_merit_dummy[:, :time_merit2.shape[1]] = time_merit2[:, :]
        time_merit2 = time_merit_dummy

    #---------------------------------------------------------------
    # Create pitch track candidates and their corresponding merit values.
    #---------------------------------------------------------------
    # Pitch candidates from spectrogram
    spec_pitch_cands = spec_pitch[np.newaxis, :]
    spec_merit_cands = np.ones((1, len(spec_pitch)))

    # Best candidates from Signal Domain
    best_pitch1 = time_pitch1[0, :]
    best_merit1 = time_merit1[0, :]

    best_pitch1_cands = best_pitch1[np.newaxis, :]
    best_merit1_cands = best_merit1[np.newaxis, :]

    # Second best candidates from Signal Domain
    if time_pitch1.shape[0] > 1:
        best_pitch2 = time_pitch1[1, :]
        best_merit2 = time_merit1[1, :]
        best_pitch2_cands = best_pitch2[np.newaxis, :]
        best_merit2_cands = best_merit2[np.newaxis, :]
    else:
        best_pitch2_cands = np.array([])
        best_merit2_cands = np.array([])

    # Best candidates from Non-linear Signal Domain
    best_pitchN1 = time_pitch2[0, :]
    best_meritN1 = time_merit2[0, :]

    best_pitchN1_cands = best_pitchN1[np.newaxis, :]
    best_meritN1_cands = best_meritN1[np.newaxis, :]

    # Second best candidates from Non-linear Signal Domain
    if time_pitch2.shape[0] > 1:
        best_pitchN2 = time_pitch2[1, :]
        best_meritN2 = time_merit2[1, :]
        best_pitchN2_cands = best_pitchN2[np.newaxis, :]
        best_meritN2_cands = best_meritN2[np.newaxis, :]
    else:
        best_pitchN2_cands = np.array([])
        best_meritN2_cands = np.array([])

    # Overall pitch candidates
    if len(best_pitch2_cands) > 0:
        pitch_cands = np.concatenate((spec_pitch_cands, best_pitch1_cands,
                                    best_pitch2_cands, best_pitchN1_cands,
                                    best_pitchN2_cands), axis=0)
    else:
        pitch_cands = np.concatenate((spec_pitch_cands, best_pitch1_cands,
                                    best_pitchN1_cands), axis=0)

    # Overall merit candidates
    if len(best_merit2_cands) > 0:
        merit_cands = np.concatenate((spec_merit_cands, best_merit1_cands,
                                    best_merit2_cands, best_meritN1_cands,
                                    best_meritN2_cands), axis=0)
    else:
        merit_cands = np.concatenate((spec_merit_cands, best_merit1_cands,
                                    best_meritN1_cands), axis=0)

    #---------------------------------------------------------------
    # Use dyanamic programming to find the final pitch track.
    #---------------------------------------------------------------
    main_pitch = dynamic(pitch_cands, merit_cands, pitch.vuv, parameters, 
                        use_cuda=pitch.use_cuda)

    #---------------------------------------------------------------
    # Post process the pitch track.
    #---------------------------------------------------------------
    main_pitch = postprocessing(main_pitch, pitch, parameters)

    #---------------------------------------------------------------
    # Create the pitch object.
    #---------------------------------------------------------------
    frames_pos = np.array([x-frame_size/2 for x in range(frame_size, len(
                                        signal.data), frame_jump)])
    pitch.set_frames_pos(frames_pos)
    pitch.set_values(main_pitch, len(signal.data))
    return pitch


"""
--------------------------------------------
            Auxiliary Functions.
--------------------------------------------
"""
#Cuda
"""
Normalized Low Frequency Energy Ratio function, used to determine voiced/
unvoiced frames.
"""
def nlfer(signal, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters for NLFER computation.
    #---------------------------------------------------------------
    fs = signal.fs
    data = signal.filtered
    nfft = parameters['fft_length']   #FFT length
    frame_size = pitch.frame_size     #Frame length
    frame_jump = pitch.frame_jump     #Frame jump size
    nframes = int(np.fix((len(data)-frame_size)/frame_jump+1))
    frame_energy = np.zeros((nframes))

    #---------------------------------------------------------------
    # Filter out everything outside the range of frequencies
    # that interest us.
    #---------------------------------------------------------------
    f_min = 50
    f_max = 800
    n_min = int(np.fix(nfft*f_min/fs))
    n_max = int(np.fix(nfft*f_max/fs))

    #---------------------------------------------------------------
    # Compute the nlfer.
    #---------------------------------------------------------------
    # Window data and take psd.
    window = hann(frame_size)
    data_matrix = stride_tricks.sliding_window_view(data, frame_size)[::frame_jump]
    
    # Use CUDA for FFT if available
    if pitch.use_cuda and _has_cuda:
        specData = cuda_rfft(data_matrix * window, nfft)
    else:
        specData = np.fft.rfft(data_matrix * window, nfft)
    
    frame_energy = np.zeros(nframes)
    frame_energy_low = np.zeros(nframes)
    
    # Calculate energy in different frequency bands
    magData = np.abs(specData)
    magData = magData**2
    
    # Full-band energy
    frame_energy = np.sum(magData[:,:n_max], axis=1)
    
    # Low-band energy
    frame_energy_low = np.sum(magData[:,:n_min], axis=1)

    # Add a small value to avoid division by zero
    frame_energy += 1e-10
    
    # Calculate NLFER
    nlfer = 10*np.log10(frame_energy_low/frame_energy)

    # Set threshold for voiced/unvoiced decision
    nlfer_threshold1 = parameters['nlfer_thresh1']
    nlfer_threshold2 = parameters['nlfer_thresh2']
    
    # Flag frames with voice activity
    vuv = np.zeros(nframes)
    
    for i in range(nframes):
        if nlfer[i] > nlfer_threshold1:
            vuv[i] = 1
        elif nlfer[i] > nlfer_threshold2:
            if i > 0 and i < nframes - 1:
                if vuv[i-1] == 1 and nlfer[i+1] > nlfer_threshold1:
                    vuv[i] = 1
    
    # Apply median filter to remove isolated voiced/unvoiced frames
    vuv = medfilt(vuv, 3)
    
    # Store results in pitch object
    pitch.set_energy(frame_energy, nlfer_threshold1)

"""
Calculate an approximate pitch track from the spectrum.
"""
def spec_track(nonlinear_sign, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters for the spectrum track calculation.
    #---------------------------------------------------------------
    fs = nonlinear_sign.fs
    data = nonlinear_sign.filtered
    nfft = pitch.nfft #FFT length
    frame_size = pitch.frame_size
    frame_jump = pitch.frame_jump
    nframes = int(np.fix((len(data)-frame_size)/frame_jump+1))
    f0_min = parameters['f0_min']
    f0_max = parameters['f0_max']
    nhar = parameters['shc_numharms']
    window_width = parameters['shc_window']
    max_peak = parameters['shc_maxpeaks']
    shc_threshold1 = parameters['shc_thresh1']
    shc_threshold2 = parameters['shc_thresh2']
    spec_pitch = np.zeros((nframes))
    pitch_std = np.zeros((nframes))

    #---------------------------------------------------------------
    # Window the data
    #---------------------------------------------------------------
    window = kaiser(frame_size, 0.5)
    data_matrix = stride_tricks.sliding_window_view(data, frame_size)[::frame_jump]
    
    # Use CUDA for FFT if available
    if pitch.use_cuda and _has_cuda:
        specData = cuda_rfft(data_matrix * window, nfft)
    else:
        specData = np.fft.rfft(data_matrix * window, nfft)

    # Compute SHC (Spectral Harmonic Correlation)
    halfspec = np.abs(specData)
    
    #---------------------------------------------------------------
    # Compute SHC in frequency domain and convert to lag domain
    #---------------------------------------------------------------
    N2 = int(np.fix(fs/f0_min))
    N1 = int(np.fix(fs/f0_max))
    D = N2-N1+1
    SHC = np.zeros((nframes, D))
    
    # Frequency resolution in Hz per FFT bin
    freqResolution = fs/nfft
    
    # Calculate indices for fundamental frequency range
    k_min = int(np.fix(f0_min/freqResolution))
    k_max = int(np.fix(f0_max/freqResolution))
    
    # Use CUDA for SHC computation if available
    if pitch.use_cuda and _has_cuda:
        # Prepare spectral data for SHC calculation
        specMag = np.abs(specData)
        
        # Calculate SHC using CUDA
        SHC = cuda_compute_shc(specMag, D, nhar)
    else:
        # CPU implementation
        for j in range(nframes):
            # Calculate SHC for each frame
            for k in range(N1, N2+1):
                SHC[j, k-N1] = 0
                for m in range(1, nhar+1):
                    index = m*k
                    if index < nfft//2:
                        SHC[j, k-N1] = SHC[j, k-N1] + halfspec[j, index]

    #---------------------------------------------------------------
    # Apply median filter and peak picking to extract pitch
    #---------------------------------------------------------------
    for i in range(nframes):
        # Apply median filter
        SHC_smooth = medfilt(SHC[i, :], 5)
        
        # Find peaks
        peaks = []
        for j in range(1, D-1):
            if SHC_smooth[j] > SHC_smooth[j-1] and SHC_smooth[j] > SHC_smooth[j+1]:
                if SHC_smooth[j] > shc_threshold1:
                    peaks.append((j, SHC_smooth[j]))
        
        # Sort peaks by amplitude and select the top ones
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:max_peak]
        
        # Convert peaks to frequency
        freqs = []
        for peak in peaks:
            freq = fs / (N1 + peak[0])
            freqs.append(freq)
        
        # Select the highest peak as the pitch
        if len(freqs) > 0:
            spec_pitch[i] = freqs[0]
        else:
            spec_pitch[i] = 0
            
        # Calculate standard deviation for pitch confidence
        if len(freqs) > 1:
            pitch_std[i] = np.std(freqs)
        else:
            pitch_std[i] = parameters['spec_pitch_min_std'] * spec_pitch[i]
    
    # Apply voiced/unvoiced decision
    spec_pitch = spec_pitch * pitch.vuv
    
    return spec_pitch, pitch_std

"""
Calculate the pitch track in time domain.
"""
def time_track(signal, spec_pitch, pitch_std, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters for the temporal pitch tracking
    #---------------------------------------------------------------
    fs = signal.fs
    data = signal.filtered
    nframes = len(spec_pitch)
    frame_size_ms = parameters['tda_frame_length']
    frame_size = int(np.fix(frame_size_ms*fs/1000))
    frame_jump = pitch.frame_jump
    ncc_thresh1 = parameters['nccf_thresh1']
    ncc_thresh2 = parameters['nccf_thresh2']
    ncc_maxcands = parameters['nccf_maxcands']
    ncc_pwidth = parameters['nccf_pwidth']
    merit_boost = parameters['merit_boost']
    f0_min = parameters['f0_min']
    f0_max = parameters['f0_max']
    
    #---------------------------------------------------------------
    # Calculate NCCF (Normalized Cross Correlation Function)
    #---------------------------------------------------------------
    # Array of lag values based on F0 range
    lag_min = int(np.fix(fs/f0_max))
    lag_max = int(np.fix(fs/f0_min))
    
    # NCCF calculation
    if pitch.use_cuda and _has_cuda:
        # Use CUDA to calculate NCCF
        nccf = cuda_compute_nccf(data, nframes, frame_size, frame_jump, lag_min, lag_max)
    else:
        # CPU implementation
        nccf = np.zeros((nframes, lag_max - lag_min))
        for i in range(nframes):
            start = i * frame_jump
            frame_data = data[start:start + frame_size]
            
            for lag in range(lag_min, lag_max):
                if start + frame_size + lag > len(data):
                    continue
                    
                shifted_frame = data[start + lag:start + lag + frame_size]
                
                # Calculate normalized cross-correlation
                numerator = np.sum(frame_data * shifted_frame)
                denominator = np.sqrt(np.sum(frame_data**2) * np.sum(shifted_frame**2))
                
                if denominator > 0:
                    nccf[i, lag - lag_min] = numerator / denominator
    
    #---------------------------------------------------------------
    # Extract pitch candidates from NCCF
    #---------------------------------------------------------------
    # Initialize pitch and merit arrays
    nCands = ncc_maxcands
    time_pitch = np.zeros((nCands, nframes))
    time_merit = np.zeros((nCands, nframes))
    
    for i in range(nframes):
        # Find peaks in NCCF
        peaks = []
        for lag in range(ncc_pwidth, lag_max - lag_min - ncc_pwidth):
            if (nccf[i, lag] > nccf[i, lag-1]) and (nccf[i, lag] > nccf[i, lag+1]):
                if nccf[i, lag] > ncc_thresh1:
                    peaks.append((lag + lag_min, nccf[i, lag]))
        
        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:nCands]
        
        # Convert lags to pitch values and set merit
        for j, (lag, corr) in enumerate(peaks):
            if j < nCands:
                freq = fs / lag
                time_pitch[j, i] = freq
                time_merit[j, i] = corr
        
        # Boost merit if pitch value is close to spectral pitch
        if spec_pitch[i] > 0:
            for j in range(nCands):
                if time_pitch[j, i] > 0:
                    ratio = time_pitch[j, i] / spec_pitch[i]
                    if 0.8 < ratio < 1.25:
                        time_merit[j, i] += merit_boost
    
    # Apply voiced/unvoiced decision
    for i in range(nframes):
        if pitch.vuv[i] == 0:
            time_pitch[:, i] = 0
    
    return time_pitch, time_merit

""" Use dynamic programming to find the final pitch track. """

def dynamic(pitch_cands, pitch_merit, vuv, parameters, use_cuda=False):

    #---------------------------------------------------------------
    # Initialize DP parameters
    #---------------------------------------------------------------
    f0_min = parameters['f0_min']
    f0_max = parameters['f0_max']
    dp_w1 = parameters['dp_w1']
    dp_w2 = parameters['dp_w2']
    dp_w3 = parameters['dp_w3']
    dp_w4 = parameters['dp_w4']
    
    nframes = pitch_cands.shape[1]
    ncands = pitch_cands.shape[0]
    
    #---------------------------------------------------------------
    # Calculate transition costs
    #---------------------------------------------------------------
    # Initialize cost matrix
    transition_cost = np.zeros((ncands, ncands))
    
    # Calculate transition costs between all candidate pairs
    for i in range(ncands):
        for j in range(ncands):
            # Get pitch values
            pi = pitch_cands[i, 0]
            pj = pitch_cands[j, 0]
            
            # Calculate transition cost based on voiced/unvoiced state
            if pi > 0 and pj > 0:
                # Voiced to voiced transition
                transition_cost[i, j] = dp_w1 * np.abs(np.log(pi) - np.log(pj))
            elif pi == 0 and pj == 0:
                # Unvoiced to unvoiced transition
                transition_cost[i, j] = dp_w3
            else:
                # Voiced to unvoiced or unvoiced to voiced transition
                transition_cost[i, j] = dp_w2
    
    #---------------------------------------------------------------
    # Run dynamic programming
    #---------------------------------------------------------------
    # Use CUDA for dynamic programming if available
    if use_cuda and _has_cuda:
        best_path = cuda_dynamic_programming(
            pitch_merit.T, transition_cost, dp_w4)
    else:
        # Initialize cost matrix and backtracking indices
        cost = np.zeros((nframes, ncands))
        prev = np.zeros((nframes, ncands), dtype=int)
        
        # Set costs for first frame
        cost[0, :] = (1 - pitch_merit[:, 0]) * dp_w4
        
        # Forward pass: calculate minimum cost path
        for i in range(1, nframes):
            for j in range(ncands):
                min_cost = float('inf')
                min_idx = 0
                
                # Find minimum cost path to current state
                for k in range(ncands):
                    trans_cost = transition_cost[k, j]
                    total_cost = cost[i-1, k] + trans_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        min_idx = k
                
                # Store minimum cost and path
                cost[i, j] = (1 - pitch_merit[j, i]) * dp_w4 + min_cost
                prev[i, j] = min_idx
        
        # Backtracking to find the best path
        best_path = np.zeros(nframes, dtype=int)
        best_path[-1] = np.argmin(cost[-1, :])
        
        for i in range(nframes-2, -1, -1):
            best_path[i] = prev[i+1, best_path[i+1]]
    
    # Extract pitch values from the best path
    final_pitch = np.zeros(nframes)
    for i in range(nframes):
        final_pitch[i] = pitch_cands[best_path[i], i]
    
    return final_pitch

"""
Do final post-processing to fix any remaining issues with the pitch track.
"""
def postprocessing(pitch_track, pitch_obj, parameters):
    
    """Apply median filtering and fix octave jumps"""
    # Apply median filter to smooth the pitch track
    filter_order = parameters['median_value']
    pitch_track = medfilt(pitch_track, filter_order)
    
    # Fix octave jumps
    f0_min = parameters['f0_min']
    f0_max = parameters['f0_max']
    f0_double = parameters['f0_double']
    f0_half = parameters['f0_half']
    
    # Find voiced segments
    voiced = pitch_track > 0
    voiced_indices = np.where(voiced)[0]
    
    # Apply octave jump correction
    for i in range(1, len(voiced_indices)):
        idx = voiced_indices[i]
        prev_idx = voiced_indices[i-1]
        
        # Check if consecutive voiced frames
        if prev_idx == idx - 1:
            curr_pitch = pitch_track[idx]
            prev_pitch = pitch_track[prev_idx]
            
            # Check for doubling
            if curr_pitch > prev_pitch * 1.8 and curr_pitch > f0_double:
                pitch_track[idx] = curr_pitch / 2
            
            # Check for halving
            elif curr_pitch * 1.8 < prev_pitch and prev_pitch > f0_half:
                pitch_track[idx] = curr_pitch * 2
    
    # Ensure pitch values are within valid range
    pitch_track[pitch_track < f0_min] = 0
    pitch_track[pitch_track > f0_max] = 0
    
    return pitch_track

"""
Checks CUDA status
"""
def is_cuda_available():
    """Check if CUDA is available for use with pYAAPT."""
    return CUDA_AVAILABLE

def cuda_info():
    """Return detailed information about CUDA availability and configuration."""
    if not _has_cuda:
        return {"available": False, "reason": "CUDA utilities not imported"}
    return get_cuda_info()

"""
Creates a matrix by taking advantage of the array strides. This is used
to simulate the usage of overlapping windows without having to make actual
copies of the data.
"""
def stride_matrix(x, n, noverlap=0):
    # This function was kept for backwards compatibility, but is now deprecated
    # in favor of numpy.lib.stride_tricks.sliding_window_view
    return stride_tricks.sliding_window_view(x, n)[::n-noverlap]

"""
Normalized Cross Correlation Function. Used in centers finds and nlfer.
"""
def crs_corr(frame, window_len):
    # This function was kept for backwards compatibility
    # The functionality is now incorporated directly into the time_track function
    nccf = np.zeros(len(frame) - window_len)
    
    for lag in range(len(frame) - window_len):
        x1 = frame[:window_len]
        x2 = frame[lag:lag+window_len]
        
        # Calculate normalized cross-correlation
        numerator = np.sum(x1 * x2)
        denominator = np.sqrt(np.sum(x1**2) * np.sum(x2**2))
        
        if denominator > 0:
            nccf[lag] = numerator / denominator
    
    return nccf