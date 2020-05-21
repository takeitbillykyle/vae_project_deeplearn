
import numpy as np
import matplotlib.pyplot as plt

arc = 4

latent_name_root = "data/latent_matrix_"

for i in range(6):
    data = np.load(latent_name_root + str(i) + ".npy")

    for j in range(0,data.shape[1],2):
        plt.scatter(data[:,j],data[:,j+1],s=0.5, alpha = 0.5)

        #plt.xlim([-1,1])
        #plt.ylim([-1,1])

    plt.show()
