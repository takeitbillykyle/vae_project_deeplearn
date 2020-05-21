import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)


labels = np.load("data/barplot/LIST_NAMES.npy")
train_loss = np.load("data/barplot/LIST_TRAIN_LOSS.npy")
val_loss = np.load("data/barplot/LIST_VAL_LOSS.npy")

labels = ['64\n32\n4','64\n32\n20','128\n64\n4','128\n64\n20','256\n128\n4','256\n128\n20','128\n64\n32\n4','128\n64\n32\n20','512\n256\n4','512\n256\n20','256\n128\n128\n64\n4','256\n128\n128\n64\n20','1024\n512\n256\n20','1024\n512\n256\n128\n20','1024\n512\n20']

if True:

	x = np.arange(len(labels))  # the label locations
	width = 0.4  # the width of the bars

	fig, ax = plt.subplots()
	rects = ax.bar(x - 0.2, train_loss, width, color = 'r')
	rects1 = ax.bar(x + 0.2, val_loss, width,color = 'b')
	

	ax.set_ylabel('Loss')
	ax.set_xlabel('Architecture')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_ylim(30,55)
	def autolabel(rects):
		"""Attach a text label above each bar in *rects*, displaying its height."""
		for rect in rects:
		    height = rect.get_height()
		    ax.annotate('{}'.format(height),
		                xy=(rect.get_x() + rect.get_width() / 2, height),
		                xytext=(0, 3),  # 3 points vertical offset
		                textcoords="offset points",
		                ha='center', va='bottom')
	#autolabel(rects)
	#autolabel(rects1)


	ax.set_title('Loss achieved with different architectures')
	fig.tight_layout()
	plt.legend((rects[0], rects1[0]), ('Train Set', 'Val. Set'))
	plt.show()
