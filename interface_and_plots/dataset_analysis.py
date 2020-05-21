import numpy as np
import matplotlib.pyplot as plt

x_train_sets = []
for i in range(6):
    x_train_sets.append( np.load('data/data_matrix' + str(i) + ".npy"))

limit = 5
correct = 0

instr = np.zeros(32)
two_note_patterns = 0
for i in range(6):

    train_set = x_train_sets[i]

    for j in range(len(train_set)):
        pattern_mat = train_set[j].reshape(14,32)

        instruments = np.sum(pattern_mat, axis = 1)

        if len(np.where(instruments >= 1)[0]) == 1:
            instr[np.where(instruments >= 1)[0]] += 1

        elif len(np.where(instruments >= 1)[0]) == 2:
            two_note_patterns +=1

print("amount of one instrument patterns per instrument")
print(instr)
print("amount of patterns with only one instrument")
print(np.sum(instr))
print("amount of patterns with only two instruments")
print(two_note_patterns)






