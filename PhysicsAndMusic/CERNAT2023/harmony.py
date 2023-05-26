#
# harmony.py - A module to research Consonance and Dissonance in Music
#
# Giuseppe Lo Presti, @glpatcern
#
# Initial version from notebooks used in the research paper
# https://link.springer.com/article/10.1140/epjp/s13360-022-03456-2
#
##########################################################################

from math import pi, sin, log, exp, ceil
import numpy as np
#import cupy as np    # to enable GPUs
import matplotlib.pyplot as plt
import scipy.signal as ss
import functools
import json
#from matplotlib.ticker import LogFormatter
import simpleaudio as sa
from scipy.io.wavfile import write as wavwrite
import librosa    # for the fast spectrogram
import librosa.display


# constants
LA4 = 440
DO4 = 261.6   # for the non-equal temperaments

fs = 24000
samples = 2**16


### Some tuning systems, cf. also http://www.microtonal-synthesis.com/scales.html

# Pythagorean tuning (including 4aug, 5dim commented out) i.e. 3-limit tuning
fn_pyt12 = [DO4 * i for i in (1, 2**8/3**5, 9/8, 32/27, 81/64, 4/3, 2**10/3**6, # 3**6/2**9,
                              3/2, 2**7/81, 27/16, 16/9, 3**5/2**7)]

# Pythagorean tuning by Fifths, from LAb to MI#
fn_pyf12 = [DO4 * (0.75**int(i/2)*1.5**int((i+1)/2)) for i in range(-5, 12) if i != 0]

# Natural/just intervals in 7-limit (NOT including 5dim at 10/7, only 4aug=tritone; a 5-limit variant for the tritone is 45/32)
nat_intervals = [(1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3), (7, 5), (3, 2), (8, 5), (5, 3), (9, 5), (15, 8)]
#                (2, 1), (32, 15), (9, 4), (12, 5), (5, 2), (8, 3), (14, 5), (3, 1), (16, 5), (10, 3), (18, 5), (15, 4)]

fn_nat12 = [DO4 * m/n for (m, n) in nat_intervals]

# Natural intervals with Chromatic Semitone in 5-limit
fn_ncs12 = [DO4 * i for i in (1, 25/24, 9/8, 6/5, 5/4, 4/3, 25/18, 3/2, 25/16, 5/3, 9/5, 15/8)]

# Quarter-Comma Meantone temperament
mtV = 5**(1/4)   # the Meantone Tempered 5th
fn_qcm12 = [DO4 * i for i in (1, 8/mtV**5, mtV**2/2, 4/mtV**3, 5/4, 2/mtV, mtV**6/8, mtV, 8/5, mtV**3/2, 4/mtV**2, mtV**5/4)]

# Werckmeister I temperament (1645)
fn_wI12 = [DO4 * i for i in (1, 256/3**5, 64*2**(1/2)/81, 32/27, 256*2**(1/4)/3**5, 4/3, 2**10/3**6, 8*8**(1/4)/9, 2**7/81, 2**10*2**(1/4)/3**6, 16/9, 128*2**(1/4)/81)]

# Werckmeister III temperament
fn_wIII12 = [DO4 * i for i in (1, 8*2**(1/4)/9, 9/8, 2**(1/4), 8*2**(1/2)/9, 9*2**(1/4)/8, 2**(1/2), 3/2, 2**7/81, 2**(3/4), 3/2**(3/4), 4*2**(1/2)/3)]

# Werckmeister IV "Septenarius" temperament
fn_wIV12 = [DO4 * i for i in (1, 98/93, 28/25, 196/165, 49/39, 4/3, 196/139, 196/131, 49/31, 196/117, 98/55, 49/26)]

# Vallotti/Young temperament (1797)
pc6 = 2**(1/6)*8/9  # 1/6th of a Pythagorean Comma
fn_vy12 = [DO4 * i for i in (1, 2**8/(pc6*3**5), 9/8*pc6**2, 2**5/(pc6*27), (pc6*3)**4/2**6, 4/(pc6*3), 2**10/(pc6*3**6), pc6*3/2, 2**7/(pc6*3**4), 27*pc6**3/16, 16/(pc6*9), (pc6*3)**5/2**7)]

# Chinese Lu scale
fn_chi12 = [DO4 * i for i in(1, 18/17, 9/8, 6/5, 54/43, 4/3, 27/19, 3/2, 27/17, 27/16, 9/5, 36/19)]

# Equal temperament (12-EDO or 12-TET)
fn_edo12 = [LA4/2 * 2**(i/12) for i in range(3, 15)]

# Natural 19 microtonal intervals in 7-limit, John Chalmers
fn_nat19 = [DO4 * i for i in (1, 21/20, 16/15, 9/8, 7/6, 6/5, 5/4, 21/16, 4/3, 7/5, 35/24,
                              3/2, 63/40, 8/5, 5/3, 7/4, 9/5, 28/15, 63/32)]

# Natural 19 microtonal intervals in 13-limit
fn_nat13l19 = [DO4 * i for i in (1, 28/27, 14/13, 10/9, 15/13, 6/5, 5/4, 9/7, 4/3, 18/13, 13/9,
                                 3/2, 14/9, 8/5, 5/3, 12/7, 9/5, 15/8, 27/14)]

# 19-EDO (1/3 Comma Meantone closely matches this)
fn_edo19 = [DO4 * 2**(i/19) for i in range(19)]

# Indian Shruti scale
fn_ind22 = [DO4 * i for i in (1, 256/3**5, 16/15, 10/9, 9/8, 32/27, 6/5, 5/4, 81/64, 4/3, 27/20, 45/32, 3**6/2**9,
                              3/2, 2**7/81, 8/5, 5/3, 27/16, 16/9, 9/5, 15/8, 3**5/2**7)]

# Natural 24 microtonal intervals in 13-limit
fn_nat24 = [DO4 * i for i in (1, 33/32, 16/15, 12/11, 9/8, 15/13, 6/5, 11/9, 5/4, 9/7, 4/3, 11/8, 7/5, 16/11,
                              3/2, 14/9, 8/5, 18/11, 5/3, 7/4, 16/9, 11/6, 15/8, 35/18)]

# 24-EDO
fn_edo24 = [LA4/2 * 2**(i/24) for i in range(6, 30)]

# Natural 31 microtonal intervals in 13-limit
fn_nat31 = [DO4 * i for i in (1, 45/44, 25/24, 15/14, 12/11, 28/25, 8/7, 7/6, 6/5, 11/9, 5/4, 32/25, 21/16,
                              4/3, 11/8, 7/5, 10/7, 19/13, 3/2, 23/15, 61/39, 8/5, 18/11, 5/3, 12/7, 7/4, 16/9,
                              11/6, 28/15, 21/11, 88/45)]

# 31-EDO
fn_edo31 = [LA4/2 * 2**(i/31) for i in range(8, 39)]

# Bohlen-Pierce scale just intervals
fn_bpnat13 = [DO4 * i for i in (1, 27/25, 25/21, 9/7, 7/5, 75/49, 5/3, 9/5, 49/25, 15/7, 7/3, 63/25, 25/9)]

# Bohlen-Pierce scale: 13-tone ET of 3/1
fn_bpedo13 = [DO4 * 3**(i/13) for i in range(13)]

# 43-EDO (Sauveur, 1653)
fn_edo43 = [DO4 * 2**(i/43) for i in range(0, 43)]

# 53-EDO (Holder, 1616)
fn_edo53 = [DO4 * 2**(i/53) for i in range(0, 53)]

temperaments = ['nat12', 'pyt12', 'ncs12', 'qcm12', 'wI12', 'wIII12', 'wIV12', 'vy12', 'chi12', 'edo12', 'nat19', 'nat13l19', 'edo19', 'ind22', 'nat24', 'edo24', 'nat31', 'edo31', 'edo43', 'edo53', 'bpnat13', 'bpedo13']

# all implemented models for amplitudes of the overtones, cf. Chord._htoamp()
h_models = ['flat', 'firstsupp', 'nodamp-string', 'harp', 'even', 'odd', 'verotta', 'harp-verotta', 'sethares']


class Chord:
    '''A class to generate clusters of notes and measure their degree of consonance'''

    def __init__(self, temperament, tonic, duration, harmonics, onlyharm, h_model, k_harp, addcb):
        self.reinit(temperament, tonic, duration, harmonics, onlyharm, h_model, k_harp, addcb)

    def reinit(self, temperament, tonic, duration, harmonics, onlyharm, h_model, k_harp, addcb):
        self.notessystem = temperament
        self.fn = globals()['fn_' + temperament]
        if '12' in temperament:
            self.nat_notessystem = 'nat12'
        elif temperament in ('edo43', 'edo53'):
            self.nat_notessystem = temperament   # we don't have "natural" ratios
        else:
            self.nat_notessystem = temperament.replace('edo', 'nat')
        self.tonic = tonic
        self.duration = duration
        self.dampingfactor = 4
        self.harmonics = harmonics
        self.onlyharm = onlyharm
        self.h_model = h_model
        self.k_harp = k_harp
        self.addcb = addcb

    def _htoamp(self, h):
        '''returns the relative amplitude of a given harmonic number for a number of models'''
        if self.h_model == 'flat':
            return 1
        if self.h_model == 'firstsupp':
            return 0.2 if h == 1 else (1 if h < 6 else 1/(h-5))
        if self.h_model == 'harp' or self.h_model == 'nodamp-string':
            return 1 / h ** 2 * sin(h * pi * self.k_harp)
        if self.h_model == 'even':
            return 1 / h ** (2 if h % 2 > 0 else 1.1)   # even harmonics decade faster
        if self.h_model == 'odd':
            return 1 / h ** (2 if h % 2 == 0 else 1.1)    # odd harmonics decade faster
        if self.h_model == 'verotta' or self.h_model == 'harp-verotta':
            return 1 / h
        if self.h_model == 'sethares':
            return 0.8**h
        return 0

    def _notetofr(self, i, fn=None):
        '''returns the frequency of the i-th note referred to the current notes system or to the given one'''
        if not fn:
            fn = self.fn
        return fn[i % len(fn)] * (2 if 'bp' not in self.notessystem else 3)**int(i / len(fn))
        #if i < 0:
        #    return int(f/2) if f/2 - int(f/2) == 0 else f/2

    def _notetoratio(self, i, fn):
        '''returns the f2_over_f1 ratio of a given interval i, taking the given rational fractions'''
        return self._notetofr(i, fn) / self._notetofr(self.tonic, fn)

    def note(self, f):
        '''Generate a numpy array for a given note index or frequency, where n = 0 means DO4.
        If f is a float, then it is directly used as a frequency.
        If onlyharm = True, generates only the harmonics.'''
        # generate array with duration*sample_rate steps, ranging between 0 and duration
        t = np.linspace(0, self.duration, int(self.duration * fs), False)
        # get the note's frequency
        if isinstance(f, int):
            f = self._notetofr(f)

        if self.onlyharm:
            # generate only the requested harmonic
            w = np.sin(2 * np.pi * f * self.harmonics * t) * self._htoamp(self.harmonics)
            return w

        # generate the fundamental wave
        w = np.sin(2 * np.pi * f * t) * self._htoamp(1) * \
            (1 if 'harp' not in self.h_model else np.exp(- t * self.dampingfactor / self.duration))
        # add the harmonics with given weights
        if self.harmonics > 1:
            for h in range(self.harmonics-1):   # h = 0..harmonics-2
                w += np.sin(2 * np.pi * f * (h+2) * t) * self._htoamp(h+2) * \
                     (1 if 'harp' not in self.h_model else np.exp(- t * (h+2) * self.dampingfactor / self.duration))
        
        # also apply some smoothing at the end, and also at the beginning for the sustained sounds
        g = ss.gaussian(4096, 500)
        if w.size > int(g.size/2):
            for i in range(int(g.size/2)):
                w[w.size-int(g.size/2)+i] *= g[int(g.size/2)+i]
        if 'harp' not in self.h_model and 'nodamp' not in self.h_model:
            for i in range(int(g.size/2)):
                w[i] *= g[i]
        return w

    def dichord(self, interval):
        s = self.note(self.tonic) + self.note(self.tonic + interval)
        return s, 0

    def tartini(self, interval):
        return self.note(self._notetofr(self.tonic + interval) - self._notetofr(self.tonic))

    # generic clusters
    def triad(self, int1, int2):
        s = self.note(self.tonic) + self.note(self.tonic + int1) + self.note(self.tonic + int2)
        print("Freqs: f1 = %.2f Hz, f2 = %.2f Hz, f3 = %.2f Hz" % (self._notetofr(0), self._notetofr(int1), self._notetofr(int2)))
        return s, 0

    def cluster(self, intlist):
        s = self.note(self.tonic)
        for i in intlist:
            s += self.note(self.tonic + i)
        return s, 0


class IChord(Chord):
    '''A specialized class for interactive usage, with play/save and plotting'''
    def __init__(self, temperament, tonic, duration, harmonics, onlyharm, h_model, k_harp, addcb):
        super().__init__(temperament, tonic, duration, harmonics, onlyharm, h_model, k_harp, addcb)

    def play(self, audio):
        '''Play a given signal to the audio card'''
        # ensure that highest value is in 16-bit range
        playable = audio * (2**15 - 1) / np.max(np.abs(audio))
        if self.onlyharm:
            # scale down so to hear the effect of the scaled single harmonic
            playable *= self._htoamp(self.harmonics)
        # stop any ongoing play
        sa.stop_all()
        # convert to 16-bit data and play
        return sa.play_buffer(playable.astype(np.int16), 1, 2, fs)

    def save(self, audio, filename):
        '''Save the given signal to an external .wav file'''
        # ensure that highest value is in 16-bit range
        playable = audio * (2**15 - 1) / np.max(np.abs(audio))
        wavwrite(filename, fs, playable.astype(np.int16))

    def plottimefreq(self, s, spectrogram=False):
        # plot the time series
        plt.subplots(figsize=(15, 5)) 
        ax = plt.subplot(1, 2, 1)
        plt.plot(s)
        plt.xlim(0)
        ax.set_xlabel('t [ms]')
        maxx = int(self.duration*10 + 1)*100
        ax.set_xticks(np.arange(0, maxx*fs/1000, maxx*fs/10000, dtype=int))
        ax.set_xticklabels(np.arange(0, maxx, maxx/10, dtype=int))
        plt.grid(which='major')
        plt.title("Wave packet")

        # also compute and plot power spectrum
        ax = plt.subplot(1, 2, 2)
        if s.size < samples:
            s = np.pad(s, (0, samples-s.size), mode='constant')
        else:
            s = s[:samples]
        W = np.abs(np.fft.fft(s) ** 2)
        f = np.fft.fftfreq(s.size, 1/fs)
        plt.plot(f, W)
        plt.xlim(30, 5000)
        #formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.1))
        #ax.get_xaxis().set_minor_formatter(formatter)
        ax.set_xlabel('f [Hz]')
        plt.ylim(1E-2)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='both')
        plt.title("Power spectrum (log/log)")

        if spectrogram:
            # play with hop_length and nfft values
            hop_length = 32
            n_fft = 1024
            bins_per_octave = 24
            fmin = 128 * (2 if self.tonic > 10 else 1)
            fmax = 4096 * (2 if self.tonic > 10 else 1)

            fig, ax = plt.subplots()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(s, hop_length=hop_length)), ref=np.max)
            img = librosa.display.specshow(D, y_axis='log', sr=fs,
                                           hop_length=hop_length, x_axis='time', ax=ax, cmap='jet',
                                           bins_per_octave=bins_per_octave, auto_aspect=False)
            ax.set_ylim([fmin, fmax])
            ax.grid(True)
            fig.colorbar(img, ax=ax, format="%+2.f dB")

        plt.show()

