import DLA_Cartan_Strcture as dla
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

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
def Kupdate(KBasis, coeffs):
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
    K_total = KBasis[0] * coeffs[0] # first term 

    for K_k, coeff in zip(KBasis[1:], coeffs[1:]): # other terms 
        K_total = K_total + (coeff * K_k)
    return K_total  



#calculates the time propogation of the intial operator to the finla time after d timesteps (size of the inital pulse array)
def propagate_operator_sparse(a_init, control_pulses, dt, Kdrift, kcontrol):
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
        K_t_sparse = Kupdate(kcontrol, coeffs)

        # Computes exp(dt * K_t) @ a_t directly and efficiently
        a_t = expm_multiply(dt * (Kdrift+K_t_sparse), a_t)

        forwardStates.append(a_t)
    return forwardStates


def backward_propagate_sparse(a_target, control_pulses, dt, KDrift, Kcontrol):
    """Same as forward propogating vector but reversed calcualtions then reversing the list at the end to match time=0 at index 0"""

    lambda_t = np.copy(a_target)
    backward_states = [lambda_t]
    
    # Iterate through pulses in reverse
    for coeffs in reversed(control_pulses):
        K_t_sparse = Kupdate(Kcontrol, coeffs)
        # Propagate backwards using -dt
        lambda_t = expm_multiply(-dt * (KDrift+K_t_sparse), lambda_t)
        
        backward_states.append(lambda_t)

    # Reverse the list so index 0 corresponds to t=0
    backward_states.reverse()
    return backward_states


def DriftK(K, g):
    return sum(K) * g


def gradientCalc(timestep, Kdrift ,Kcontrol, forwardprop, backwardprop, control_pulses):
    """
    Calculates the gradient using the second-order approximation.
    Reconstructs K_l on the fly by adding the drift and active controls.
    """
    #number of time steps
    steps = len(control_pulses)
    #creating 0 matrix for gradient of every control at every time step
    num_controls = len(Kcontrol)
    grad = np.zeros((steps, num_controls))
    
    for l in range(steps):

        a_l = forwardprop[l]

        # Reconstruct the total K_l matrix for this specific time step[cite: 16]
        K_t_control = Kupdate(Kcontrol, control_pulses[l])
        K_l = Kdrift + K_t_control 
        
        # Precompute K_l @ a_l once per time step for efficiency
        K_l_a = K_l.dot(a_l)
        
        for k in range(num_controls):
            K_k_c = Kcontrol[k]
            
            # 1. First order term applied to a_l: K_k^c @ a_l
            K_k_a = K_k_c.dot(a_l)
            
            # 2. Commutator applied to a_l: (K_l K_k^c - K_k^c K_l) @ a_l[cite: 22]
            commutator_a = K_l.dot(K_k_a) - K_k_c.dot(K_l_a)
            
            # 3. Combine into the full second-order derivative applied to a_l[cite: 22]
            derivative_a = (timestep * K_k_a) - (0.5 * (timestep ** 2) * commutator_a)
            
            # 4. Final gradient calculation[cite: 22]
            grad[l, k] = -np.dot(backwardprop[l], derivative_a)
    return grad

# analsysis tool

#THIS IS CORRECT
def compute_infidelity(a_final, a_target):

    norm_target_sq = np.dot(a_target, a_target)
    #cehck for 0 vector
    if norm_target_sq == 0:
        return 1.0

    return 1.0 - (np.dot(a_final, a_target) / norm_target_sq)




def optimizationloop(falt_pulses, a_init, a_target, dt, Kdrift, Kcontrol, steps, num_controls):
    #scipy only takes flat pulses so have to remake them into 2d array for our functions
    current_pulses = falt_pulses.reshape((steps, num_controls))

    #forward propogate according to our new or inital pulses
    forwardsprop = propagate_operator_sparse(a_init, current_pulses, dt, Kdrift, Kcontrol)

    #calculate the infedility of our function
    current_infidelity = compute_infidelity(forwardsprop[-1], a_target)

    #calculate the backwards propogated states
    backwardprop = backward_propagate_sparse(a_target, current_pulses, dt, Kdrift, Kcontrol)

    #calculate the gradient
    grad_2d = gradientCalc(dt, Kdrift, Kcontrol, forwardsprop, backwardprop, current_pulses)

    #have to flattern gradient again for scipy
    return current_infidelity, grad_2d.flatten()


