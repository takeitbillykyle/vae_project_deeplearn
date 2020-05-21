import numpy as np
from matplotlib import pyplot as plt

latent_activations = np.zeros((6,5000,20))

for i in range(6):
    latent = np.load("data/latent_matrix_" + str(i) + ".npy")
    print("loaded matrix shape")
    print(latent.shape)
    latent_activations[i] = latent

latent_activations = np.array(latent_activations)
print("latent_activations shape")
print(latent_activations.shape)

latent = np.concatenate(latent_activations, axis = 0)
print("latent shape")
print(latent.shape)

latent_cov = np.dot(latent.T, latent)/latent.shape[0]
eigen_values, eigen_vectors = np.linalg.eig(latent_cov)

print("eigen_vectors shape")
print(eigen_vectors.shape)

np.save("data/latent_pca_comps.npy", eigen_vectors)

latent_pca = np.dot(latent,eigen_vectors[:2,:].T)

print("shape of latent_pca")
print(latent_pca.shape)

plt.scatter(latent_pca[:,0],latent_pca[:,1])
plt.show()