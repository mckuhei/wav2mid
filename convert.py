import os
import struct
import sys
import mido
import math
import wave

import numpy as np

class ChannelStatus():
	
	__slots__ = ["on", "freq", "key", "key_float", "in_use"]

	def __init__(self):
		super(ChannelStatus, self).__init__()
		self.on = False
		self.freq = 0
		self.key  = 0

tomidi = lambda n : (12*math.log(n/440,2))+69

BPM = 120
RESOLUTION = 480
DELAY = mido.bpm2tempo(BPM) / RESOLUTION * 1e-6 # AKA 0.000 000 1
SENSITIVITY = 127
KEYRANGE = SENSITIVITY / 4
STEP = 10
CHANNELS = 16

if len(sys.argv) < 3:
	sys.stderr.write("Usage: %s <in.wav> <out.mid>\n" % sys.argv[0])
	sys.exit(1)

if len(sys.argv) >= 3:
	out = sys.argv[2]

def read(file):
	wav = wave.open(file, "r")
	data = wav.readframes(wav.getnframes())
	wav.close()
	wid = wav.getsampwidth()
	channels = []
	arraytype = getattr(np, "int" + str(wid * 8) if wid != 4 else "float")
	array = np.frombuffer(data, arraytype)
	for ch in range(wav.getnchannels()):
		channels.append(array[ch::wav.getnchannels()])
	return wav.getframerate(), channels if len(channels) != 1 else channels[0]

sample_rate, data = read(sys.argv[1])

window_size = int(DELAY * sample_rate * STEP)

if type(data[0]) == np.ndarray:
	#sys.stderr.write("Mono audio required\n")
	#sys.exit(1)
	channels = len(data)
	d = data[0] / channels
	for ch in range(1, channels):
		d += data[ch] / channels
	data = d

samples = []

max_volume = 0

extra = int(window_size / 2)

data = np.pad(data, (extra, extra))

for i in range(extra, len(data) - extra, window_size):
	window = data[i-extra:i+window_size+extra]
	n = len(window)
	freq = np.fft.fftfreq(n, 1/sample_rate)[range(int(n / 2))]
	fourier = np.fft.fft(window) / n
	fourier = fourier[range(int(n / 2))]
	velocity = np.abs(fourier) ** 0.5
	values = [x for x in sorted(zip(freq, velocity),
								key=lambda x: x[1],
								reverse=True)]
	sample = values[:min(CHANNELS, len(values))]
	sample += [(0, 0)] * (16 - len(sample))
	samples.append(sample)
	# samples.append(sorted(values[:min(CHANNELS, len(values))], key = lambda x: x[0]))
	max_volume = max(max_volume, np.max(velocity))

print(max_volume)

mid = mido.MidiFile()
tracks = [mido.MidiTrack() for _ in range(CHANNELS)]
status = [ChannelStatus()  for _ in range(CHANNELS)]
# Insert GS Reset
tracks[0].append(mido.Message	 ("sysex"    , data  = [0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7f, 0x00, 0x41]))
tracks[0].append(mido.Message	 ("sysex"	 , data  = [0x41, 0x10, 0x42, 0x12, 0x40, 0x10, 0x15, 0x00, 0x1b]))
tracks[0].append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(BPM)))
for i in range(CHANNELS):
	track = tracks[i]
	mid.tracks.append(track)
	# Reset
	track.append(mido.Message('control_change', channel = i, control = 121))
	track.append(mido.Message('control_change', channel = i, control = 7  , value = 127))
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

def applyToChannel(ch, s):
	velocity = min(int(s[1] / max_volume * 16383), 16383)
	if velocity < 127 or s[0] == 0:
		if status[ch].on:
			tracks[ch].append(mido.Message('note_off', channel = ch, note = status[ch].key))
			# tracks[ch].append(mido.Message('control_change', channel = ch, control = 11, value = 0, time = 0))
			status[ch].on = False
		time[ch] += 1 * STEP
		return
	key_f = tomidi(s[0])
	key = int(key_f)
	status[ch].freq = s[0]
	if key > 127 or key < -1:
		if status[ch].on:
			tracks[ch].append(mido.Message('note_off', channel = ch, note = status[ch].key))
			# tracks[ch].append(mido.Message('control_change', channel = ch, control = 11, value = 0, time = 0))
			status[ch].on = False
		time[ch] += 1 * STEP
		return
	# Expression
	tracks[ch].append(mido.Message('control_change', channel = ch, control = 11, value = velocity >> 7, time = time[ch]))
	# tracks[ch].append(mido.Message('control_change', channel = ch, control = 43, value = velocity & 0x7F, time = 0))
	if not status[ch].on or abs(status[ch].key - key) >= KEYRANGE:
		if status[ch].on:
			tracks[ch].append(mido.Message('note_off', channel = ch, note = status[ch].key))
		tracks[ch].append(mido.Message('pitchwheel', channel = ch, pitch = int((key_f - key) * (8192 / SENSITIVITY))))
		tracks[ch].append(mido.Message('note_on'   , channel = ch, note = key, velocity = 127))
		status[ch].on = True
		status[ch].key = key
	else:
		tracks[ch].append(mido.Message('pitchwheel', channel = ch, pitch = int((key_f - status[ch].key) * (8192 / SENSITIVITY))))
	time[channel] = 1 * STEP

while samples:
	sample = samples.pop(0)
	ticks += 1
	available = list(range(CHANNELS))
	for ch in range(CHANNELS):
		available.sort(key=lambda x: abs(sample[ch][0] - status[x].freq))
		channel = available.pop(0)
		applyToChannel(channel, sample[ch])

			
for i in range(CHANNELS):
	if status[i].on:
		track = tracks[i]
		track.append(mido.Message('note_off', channel = i, note = status[i].key))

mid.save(out)
