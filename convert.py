import os
import struct
import sys
import mido
import math

import numpy as np
import scipy.io.wavfile as wav

class ChannelStatus():
	
	__slots__ = ["on", "freq", "key"]

	def __init__(self):
		super(ChannelStatus, self).__init__()
		self.on = False
		self.freq = 0
		self.key  = 0

tomidi = lambda n : (12*math.log(n/440,2))+69

BPM = 120
RESOLUTION = 480
DELAY = mido.bpm2tempo(BPM) / RESOLUTION * 1e-6 # AKA 0.000 000 1
SENSITIVITY = 96
STEP = 32
CHANNELS = 16

if len(sys.argv) < 3:
	sys.stderr.write("Usage: %s <in.wav> <out.mid>\n" % sys.argv[0])
	sys.exit(1)

if len(sys.argv) >= 3:
	out = sys.argv[2]

sample_rate, data = wav.read(sys.argv[1])

window_size = int(DELAY * sample_rate * STEP)

if type(data[0]) == np.ndarray:
	sys.stderr.write("Mono audio required\n")
	sys.exit(1)

samples = []

max_volume = 0

for i in range(0, len(data), window_size):
	window = data[i:i+window_size]
	n = len(window)
	freq = np.fft.fftfreq(n, 1/sample_rate)[range(int(n / 2))]
	fourier = np.fft.fft(window) / n
	fourier = fourier[range(int(n / 2))]
	velocity = np.abs(fourier)
	values = [x for x in sorted(zip(freq, velocity),
								key=lambda x: x[1],
								reverse=True)]
	samples.append(values[:min(CHANNELS, len(values))])
	# samples.append(sorted(values[:min(CHANNELS, len(values))], key = lambda x: x[0]))
	max_volume = max(max_volume, np.max(velocity))

mid = mido.MidiFile()
tracks = [mido.MidiTrack() for _ in range(CHANNELS)]
status = [ChannelStatus()  for _ in range(CHANNELS)]
# Insert GS Reset (Sadly, Doesn't work)
# tracks[0].append(mido.Message	("sysex"	, data  = [0xf0, 0x41, 0x10, 0x42, 0x12, 0x40, 0x10, 0x15, 0x00, 0x1b, 0xf7]))
tracks[0].append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(BPM)))
for i in range(CHANNELS):
	track = tracks[i]
	mid.tracks.append(track)
	# Reset
	track.append(mido.Message('control_change', channel = i, control = 121))
	# Setting up program
	track.append(mido.Message('control_change', channel = i, control = 0  , value = 8))
	track.append(mido.Message('control_change', channel = i, control = 32 , value = 0))
	track.append(mido.Message('program_change', channel = i, program = 80))
	# Setting up pitch blend sensitivity 
	track.append(mido.Message('control_change', channel = i, control = 101))
	track.append(mido.Message('control_change', channel = i, control = 100, value = 0))
	track.append(mido.Message('control_change', channel = i, control = 6  , value = SENSITIVITY))
	# track.append(mido.Message('control_change', channel = i, control = 38 , value = 0))
	# Setting up reverb
	track.append(mido.Message('control_change', channel = i, control = 91 , value = 0))

time = [0 for _ in range(CHANNELS)]

ticks = 0

while samples:
	sample = samples.pop(0)
	ticks += 1
	for ch in range(CHANNELS):
		s = sample[ch]
		velocity = min(int(s[1] / max_volume * 127), 127)
		if velocity < 1 or s[0] == 0:
			if status[ch].on:
				tracks[ch].append(mido.Message('note_off', channel = ch, note = status[ch].key))
				# tracks[ch].append(mido.Message('control_change', channel = ch, control = 11, value = 0, time = 0))
				status[ch].on = False
			time[ch] += 1 * STEP
			continue
		key_f = tomidi(s[0])
		key = int(key_f)
		if key > 127 or key < -1:
			if status[ch].on:
				tracks[ch].append(mido.Message('note_off', channel = ch, note = status[ch].key))
				# tracks[ch].append(mido.Message('control_change', channel = ch, control = 11, value = 0, time = 0))
				status[ch].on = False
			time[ch] += 1 * STEP
			continue
		tracks[ch].append(mido.Message('control_change', channel = ch, control = 11, value = velocity, time = time[ch]))
		if not status[ch].on or abs(status[ch].key - key) >= SENSITIVITY:
			if status[ch].on:
				tracks[ch].append(mido.Message('note_off', channel = ch, note = status[ch].key))
			tracks[ch].append(mido.Message('pitchwheel', channel = ch, pitch = int((key_f - key) * (8192 / SENSITIVITY))))
			tracks[ch].append(mido.Message('note_on'   , channel = ch, note = key, velocity = 127))
			status[ch].on = True
			status[ch].key = key
		else:
			tracks[ch].append(mido.Message('pitchwheel', channel = ch, pitch = int((key_f - status[ch].key) * (8192 / SENSITIVITY))))
		time[ch] = 1 * STEP

			
for i in range(CHANNELS):
	if status[ch].on:
		track = tracks[i]
		track.append(mido.Message('note_off', channel = i, note = status[ch].key))

mid.save(out)