import numpy as np
from midiutil import MIDIFile
from matplotlib import pyplot as plt

'''
Drum conversion table as supplied by name of blog dude

DRUM_CONVERSION = 35:36, # acoustic bass drum -> bass drum (36)
                    37:38, 40:38, # 37:side stick, 38: acou snare, 40: electric snare
                    43:41, # 41 low floor tom, 43 ghigh floor tom
                    47:45, # 45 low tom, 47 low-mid tom
                    50:48, # 50 high tom, 48 hi mid tom
                    44:42, # 42 closed HH, 44 pedal HH
                    57:49, # 57 Crash 2, 49 Crash 1
                    59:51, 53:51, 55:51, # 59 Ride 2, 51 Ride 1, 53 Ride bell, 55 Splash
                    52:49, # 52: China cymbal
                    63:62, # synthbrasses
                    65:64, # saxes
                    66:64  # saxes

                    0 to 16383 integer represantion of
                    ALLOWED_PITCH = [36, 38, 42, 46, 41, 45, 48, 51, 49, 58, 60, 61, 62, 64]
'''


'''
	GM MIDI Drum standard:
	36=kick, 38=snare ,41=lotom, 42=closed HH,4 5=midtom, 46=open HH,  48=hitom,  49=crash, 51=ride, 58=vibraslap
	60=hi bongo, 61=lo bongo,62=synthbrass,64=saxes
'''
drum_map = [36, 38, 42, 46, 41, 45, 48, 51, 49, 58, 60, 61, 62, 64]



def flatMatrixToMidi(matrix, name = None, write = False, drum_map = [36, 38, 42, 46, 41, 45, 48, 51, 49, 58, 60, 61, 62, 64]):
	'''
	Generates a MIDI-file from a matrix representaion. Handles up to 14 instruments and 32 beats

	arguments:
	matrix : numpy array, should be 14 x 32

	kwargs:
	name : (str) name of file if MIDI file is to be written to disk
	write : (bool) if MIDI file is to be written to disk
	drum_map : mapping of numerical values to MIDI notes

	returns:
	drum_MIDI : (MIDI object)
	'''

	drum_MIDI = MIDIFile(1)
	track    = 0
	channel  = 0
	time     = 0   # In beats
	volume   = 100 # 0-127, as per the MIDI standard

	drum_MIDI.addTempo(track ,time, tempo)

	matrix = matrix.reshape(14,32)
	
	num_rows, num_cols = matrix.shape

	for t in num_cols:
		for i in num_rows:
			if matrix[i,t] == '1':

				pitch = drum_map[i]

				drum_MIDI.addNote(track, channel, pitch, time, duration, volume)

				pattern_mat[cdx - 2,idx] = 1

	if write:
		with open(name + ".mid", "wb") as output_file:
			drum_MIDI.writeFile(output_file)

	return drum_MIDI


def intSeqToMidiAndMat(int_seq, name = None, plot_pattern = False, write = False, drum_map = [36, 38, 42, 46, 41, 45, 48, 51, 49, 58, 60, 61, 62, 64]):
	'''
	Generates a MIDI-file and a matrix representaion from a drum beat enconded as integers

	arguments:
	int_seq : some sequence of integers

	kwargs:
	name : (str) name of file if MIDI file is to be written to disk
	write : (bool) if MIDI file is to be written to disk
	plot_pattern : (bool) if matrix represantion should be plotted
	drum_map : mapping of numerical values to MIDI notes

	returns:
	drum_MIDI : (MIDI object) 
	pattern_mat : (numpy matrix [num drums,num steps]) matrix repr. of pattern
	'''

	drum_MIDI = MIDIFile(1)
	track    = 0
	channel  = 0
	time     = 0   # In beats
	volume   = 100 # 0-127, as per the MIDI standard
	drum_MIDI.addTempo(track ,time, tempo)

	pattern_mat = np.zeros((14, 32))

	for idx, integer in enumerate(int_seq):
		binary = bin(integer)

		for cdx, char in enumerate(binary):
			if cdx > 1: #first two chars in binary string are 0b before actual binary number
				if char == '1':

					pitch = drum_map[cdx - 2]

					drum_MIDI.addNote(track, channel, pitch, time, duration, volume)

					pattern_mat[cdx - 2,idx] = 1
			
		time = time + 0.125 #1 = quarter note, 0.125 = 32 note

	if write:
		with open(name + ".mid", "wb") as output_file:
			drum_MIDI.writeFile(output_file)

	if plot_pattern:

		fig, ax = plt.subplots(figsize=(3, 3))
		ax.matshow(pattern_mat)
		plt.show()

	return pattern_mat

def intSeqToMat(int_seq):
	'''
	Generates a matrix representaion from a drum beat enconded as integers

	arguments:
	int_seq : some sequence of integers

	returns:
	pattern_mat : (numpy matrix [num drums,num steps]) matrix repr. of pattern
	'''

	pattern_mat = np.zeros((14, 32))

	for idx, integer in enumerate(int_seq):
		binary = bin(integer)

		for cdx, char in enumerate(binary):
			instrument_range = len(binary) - 3

			if cdx > 1: #first two chars in binary string are 0b before actual binary number
				if char == '1':

					pattern_mat[instrument_range - (cdx - 2),idx] = 1

	return pattern_mat

def flatMatToMat(flat_matrices_array, save = False, name_root = "pattern_tensor"):
	'''
	Converts flattened midi pattern matrices to non flattened matrix form

	arguments:
	flat_matrices_array : (numpy array, [ patterns x flattened_pattern ]) 
						input with flattened matrices (patterns stored as rows
	
	kwargs:
	save : (boolean) if True restored matrices are saved as name.npy
	name_root : (string) name root of saved matrices

	return:
	matrices : (numpy array, [num_patterns x 14 x 32]), tensor with pattern matrices in dim 1
	'''
	num_patterns = len(flat_matrices_array)

	#tensor to store restored matrices
	matrices = np.zeros((num_patterns,14,32))

	for idx, flat_pattern in enumerate(flat_matrices_array):

		pattern_matrix = flat_pattern.reshape(14,32)

		matrices[idx] = pattern_matrix

	if save:
		np.save(name + ".npy", matrices)

	return matrices 

def rawDataToMatrix(raw_data = 'data/dataset.tsv', print_progress = True):

	data = []

	#data in raw format is tab separated with first column containing integer sequences
	with open(raw_data) as raw:
		for row in raw:
			row_list = row.split('\t')

			sequence = row_list[0].split(',')

			for edx, element in enumerate(sequence):
				sequence[edx] = int(element)

			data.append(sequence)

	instruments = np.zeros(16383)
	for sdx, sequence in enumerate(data):
		for element in sequence:
			if element > 0:
				instruments[element - 1] += 1

	cleaned_data = []
	'''
	print("length of data before cleaning")
	print(len(data))
	for sdx, sequence in enumerate(data):
		not_only_bass = False
		
		for ndx, number in enumerate(sequence):
			print(number)
			if number > 1:
				not_only_bass = True

		if not_only_bass:
			print("hello")
			cleaned_data.append(sequence)
		input()
	
	print("length of data after cleaning")
	print(len(cleaned_data))
	'''
	first_row = True
	data_matrix = None

	data_matrix_collection = []
	data_matrices_complete = 0

	for sdx, sequence in enumerate(data):

		seq_matrix = intSeqToMat(sequence)
		seq_matrix = seq_matrix.flatten() #row major
		seq_matrix = seq_matrix.reshape(1,len(seq_matrix))

		#if first pattern in set of 5000, data matrix is initialized as the first pattern
		if first_row:
			data_matrix = seq_matrix
			first_row = False

		#if not first pattern, pattern is concatenated to data_matrix
		#different patterns storied in rows, flattened pattern across columns
		else:
			data_matrix = np.concatenate((data_matrix,seq_matrix), axis = 0)

			#every 5000 points, append data matrix to collection and reset
			if (((sdx + 1) % 5000 == 0) or (sdx == (len(data) - 1))):
				data_matrix_collection.append(data_matrix)
				first_row = True
				data_matrices_complete += 1
				print("shape of data matrix")
				print(data_matrix.shape)

		if print_progress:
			if sdx % 100 == 0:
				print(str(sdx + 1) + " out of " + str(len(data)) + " patterns have been converted to matrix form")

	return data_matrix_collection

def main():
	data_matrix_collection = rawDataToMatrix()

	for mdx, matrix in enumerate(data_matrix_collection):
		np.save('data/data_matrix_cleaned' + str(mdx) + '.npy',matrix)
'''

def main():
	dummy_seq = [1,0,3,0,1,0,3,0,1,0,3,0,1,0,3,0,1,0,3,0,1,0,3,0,1,0,3,0,1,0,3,0,]

	seq_matrix = intSeqToMat(dummy_seq)
	plt.matshow(seq_matrix)
	plt.show()
	seq_matrix = seq_matrix.flatten() #row major
	seq_matrix = seq_matrix.reshape(1,len(seq_matrix))

	full_matrix = flatMatToMat(seq_matrix)
	plt.matshow(full_matrix[0])
	plt.show()
'''

if __name__ == "__main__":
	main()
