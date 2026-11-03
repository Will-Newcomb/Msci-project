from qiskit.quantum_info import Pauli
import DLA_Cartan_Strcture as dla
import Config
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from math import pi
import Tools

#note to self the index map and list ordering match

def KCreation(controlOps, pauliLiearray, index_map):
    """
    Calculates all strcture constants, stores them as sparse matrices in a list ordered by control operators/index map
    
    inputs:
        ControlOps: Needed to iterate though and calculate strcuture constants
        PauliLiearray: Need to calculate structure constants with each contorl operator
        indexMap: needed as strcure constant calculation needs both vectorised calcutions and non ordered lookups

    outputs:
        K_matrices: a list of structure constants stored as sparce matrices numbered/ordered by the indexmap/control ops 

    Note:
        Should be in DLA file, then all indexmap,paulilist and DLA are stored in a file to not be recalculated?
    """
    
    d = len(pauliLiearray)
    K_matrices = []

    for op in controlOps:
        rows, cols, values = dla.get_structure_constant(pauliLiearray, op, index_map)
        # Build sparse matrix directly without converting to dense
        K_k_sparse = csr_matrix((values, (rows, cols)), shape=(d, d), dtype=float)
        K_matrices.append(K_k_sparse)
    return K_matrices

#Will want to seperte this into dirft and control terms for furture speed up
def Kupdate(K_matrices, coeffs):
    """
    Calculates the K matrix for a set of control coefficants/pulses and their corresponding strcture constant

    Inputs:
        k_matrices: The list of strcutre constants
        Coedds: control coefficants 
    
    Output:
        K_total: the K matrix complete
    
    Note:
        Split into drift and control will improve efficeinty
    """

    #does this seperatly as avoids summing 0 dense matrix with a sparce one
    K_total = K_matrices[0] * coeffs[0] # first term 

    for K_k, coeff in zip(K_matrices[1:], coeffs[1:]): # other terms 
        K_total = K_total + (coeff * K_k)
    return K_total  



#calculates the time propogation of the intial operator to the finla time after d timesteps (size of the inital pulse array)
def propagate_operator_sparse(a_init, control_pulses, dt, K):
    """
    Forward propogates the inital vector according to a set of control coefficants.
    The length of control pulses dictates the total time.

    Inputs
        a_init: The intial vector 
        control_pulses: The set of control coefficants for each time step, length controls total control time
        dt: the timestep of each propogation
        K: the K matrix used for propogation
      
    Outputs
        forwardsStates: a list containting the propogating vector at each time step

      """

    
    a_t = np.copy(a_init)
    forwardStates = [a_t]

    for i, coeffs in enumerate(control_pulses):
        K_t_sparse = Kupdate(K, coeffs)

        # Computes exp(dt * K_t) @ a_t directly and efficiently
        a_t = expm_multiply(dt * K_t_sparse, a_t)

        forwardStates.append(a_t)
    return forwardStates


def backward_propagate_sparse(a_target, control_pulses, dt, K_matrices):
    """Same as forward propogating vector but reversed calcualtions then reversing the list at the end to match time=0 at index 0"""

    lambda_t = np.copy(a_target)
    backward_states = [lambda_t]
    
    # Iterate through pulses in reverse
    for coeffs in reversed(control_pulses):
        K_t_sparse = Kupdate(K_matrices, coeffs)
        # Propagate backwards using -dt
        lambda_t = expm_multiply(-dt * K_t_sparse, lambda_t)
        
        backward_states.append(lambda_t)

    # Reverse the list so index 0 corresponds to t=0
    backward_states.reverse()
    return backward_states



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
StratingVector = Tools.vectorMaker(Config.StartingVectorTerms,Config.StartingVectorcoeff, pauliLiearray, index_map)


#final time propogation of everything
final = propagate_operator_sparse(StratingVector, initial_pulses, Config.Timestep, K_matrices)



TargetVector = Tools.vectorMaker(Config.TargetVectorTerms,Config.TargetVectorCoeff,pauliLiearray, index_map)
print(compute_infidelity(final[-1], TargetVector))
print(final)

#watch out for captional of lie elements
#seems to work for 2 qbit case