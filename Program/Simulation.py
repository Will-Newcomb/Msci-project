from qiskit.quantum_info import Pauli
import DLA_Cartan_Strcture as dla
import Config
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from math import pi
import Tools

#This function currently uses the strcutre constant calculatro from DLA does it for every control operator and convertes them into a list of sparce matrices
#The order is the order of the contorl ops 

def KCreation(controlOps, pauliLiearray, index_map):
    d = len(pauliLiearray)
    K_matrices = []

    for op in controlOps:
        rows, cols, values = dla.get_structure_constant(pauliLiearray, op, index_map)
        # Build sparse matrix directly without converting to dense
        K_k_sparse = csr_matrix((values, (rows, cols)), shape=(d, d), dtype=float)
        K_matrices.append(K_k_sparse)
    return K_matrices

#Thisfunction sets the coefficenat of each bassis term in the K matrix
#Will want to seperte this into dirft and control terms for furture speed up
def Kupdate(K_matrices, coeffs):
    K_total = K_matrices[0] * coeffs[0] # first term 

    for K_k, coeff in zip(K_matrices[1:], coeffs[1:]): # other terms 
        K_total = K_total + (coeff * K_k)
    return K_total  


#Creates the intial vector we want to start with have a sepeate function for this as human input migth get the letters in the wong place
#is also useful for just populaitng it ie for different cases
#This is correct!

#calculates the time propogation of the intial operator to the finla time after d timesteps (size of the inital pulse array)
def propagate_operator_sparse(a_init, control_pulses, dt, K_matrices):

    a_t = np.copy(a_init)

    for coeffs in control_pulses:
        K_t_sparse = Kupdate(K_matrices, coeffs)

        # Computes exp(dt * K_t) @ a_t directly and efficiently
        a_t = expm_multiply(dt * K_t_sparse, a_t)
    return a_t


# analsysis tool

#THIS IS CORRECT
def compute_infidelity(a_final, a_target):

    norm_target_sq = np.dot(a_target, a_target)
    #cehck for 0 vector
    if norm_target_sq == 0:
        return 1.0

    return 1.0 - (np.dot(a_final, a_target) / norm_target_sq)


# --- Initialization and DLA Setup ---




# Generate control operators
controlAlgebra_dict = {i: Pauli(i) for i in Config.controlSets[Config.activeSystem]}
controlAlgebra_list = list(controlAlgebra_dict.values())
#generate lie algebra and keep a list of them with an index mapping for quick checks
lieAlgebradict = dla.DLA(controlAlgebra_dict)
index_map, pauliLiearray = dla.convert(lieAlgebradict)
print(index_map)
#create the basis set of the K matrix needs index map to make strcture constants as dicontary loopkups and vectorised calculations 
#outputs a list of matrices in order of the contol list and dimensional order of the pualiliearray/index map
K_matrices = KCreation(controlAlgebra_list, pauliLiearray, index_map)

#create the inital 2d guess array for the basis elemnts of the K matrix ie arrys of guessed values for d time steps
#arrays of size steps and then one value for eahc contorl element
initial_pulses = np.random.uniform(-0.1, 0.1, size=(Config.Steps, len(controlAlgebra_list)))

#creat the starting objects of the evolution takes index map to match our chosen elements to the right location
StratingVector = Tools.vectorMaker(["ZI","IZ","ZZ"],[1,1,1], pauliLiearray, index_map)


#final time propogation of everything
final = propagate_operator_sparse(StratingVector, initial_pulses, Config.Timestep, K_matrices)



TargetVector = Tools.vectorMaker(["XX"],[1],pauliLiearray, index_map)
print(compute_infidelity(final, TargetVector))
print(final)

#watch out for captional of lie elements
#seems to work for 2 qbit case