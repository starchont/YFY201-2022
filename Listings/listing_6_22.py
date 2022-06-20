from numpy import *
from numpy.linalg import *
import numpy as np

nmax = 4
H = zeros((nmax, nmax), float)

# Pauli matrices in order to calculate tensor products
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
XAXB = np.tensordot(sigma_x, sigma_x, axes=0)
YAYB = np.tensordot(sigma_y, sigma_y, axes=0)
ZAZB = np.tensordot(sigma_z, sigma_z, axes=0)
SASB = XAXB + YAYB + ZAZB - 3 * ZAZB

sasb = np.array([[-2, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, -2]])

# Hamiltonian
print("\nHamiltonian without mu^2/r^3 factor\n", sasb, "\n")

# Eigenvalues and eigenvectors
es, ev = eig(sasb)
print("Eigenvalues \n", np.round(es), "\n")
print("Eigenvectors (in columns)\n", ev, "\n")
# Eigenvectors
phi1 = (ev[0, 0], ev[1, 0], ev[2, 0], ev[3, 0])
phi4 = (ev[0, 1], ev[1, 1], ev[2, 1], ev[3, 1])
phi3 = (ev[0, 2], ev[1, 2], ev[2, 2], ev[3, 2])
phi2 = (ev[0, 3], ev[1, 3], ev[2, 3], ev[3, 3])
# List eigenvectors
basis = [phi1, phi2, phi3, phi4]
# Hamiltonian in new basis
for i in range(0, nmax):
    for j in range(0, nmax):
        term = dot(sasb, basis[i])
        H[i, j] = dot(basis[j], term)
print("Hamiltonian in Eigenvector Basis\n", np.round(H))

