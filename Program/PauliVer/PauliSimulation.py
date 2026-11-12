import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp


#THESE 4 FUNCTIONS TAKE YOUR HAMTILTIAN AND OPERATOR AND CALCULATE THE TIME EVOLUTION OF THE OPERATOR ACCORDING TO HEISNBERGS EQUATION USING RUNGE KUTTA
#creates an inital operator as a pauli object

#calculates the total hamiltian taking the drift hamtilian already multipled by its interaction strength and each control term to multiply by its coeff
def ComputeHam(Hdrift, HcontrolTerms, HControlCoeff):

    H_total = Hdrift
    H_total = H_total + SparsePauliOp(HcontrolTerms,coeffs=HControlCoeff)  

    return H_total.simplify()

#calculates the time derative of the invarient operator using its EOM
def computeCommutator(Hamiltonian, InvarientOp):

    commutator = (Hamiltonian @ InvarientOp) - (InvarientOp @ Hamiltonian)
    
    return (-1j * commutator).simplify()



#This computes the transfrom propgaiton by taylor expanding the expoenential and then using the fact DA/Dt equals the commutator with H terms and then taking the coefficants
#it applied k^A by using K(KA) where KA is a new A vector
#order is how many terms deep it goes might want to check that
def propagate_step_exp(A_t, H_drift, H_control_list, control_coeffs, dt, order=8, threshold=1e-10):
    """
    Pauli-string version of expm_multiply(dt*K, a): returns exp(dt*K) A_t using the series
        sum_n (dt*K)^n A_t / n!
    where K A = -i[H, A] (computeCommutator). Each term is pruned as it is built.
    """
    H_total = ComputeHam(H_drift, H_control_list, control_coeffs)

    A_next = A_t
    term = A_t
    #looping though taylor terms and adding them up
    for n in range(1, order + 1):
        term = ((dt / n) * computeCommutator(H_total, term)).simplify()
        if np.all(np.abs(term.coeffs) <= threshold):   # nothing left to add
            break
        A_next = (A_next + term).simplify()

    return A_next


#THESE 3 CALCULATE THE GRADIETN WITH RESPECT TO EAHC CONTORL TERM LIKE THE PAPER
def forwardspropogation(A_init, H_drift, H_control_list, pulses, dt, threshold):

    A_t = A_init
    propstates = [A_t]

    for coeffs in pulses:
        A_t = propagate_step_exp(A_t, H_drift, H_control_list, coeffs, dt, threshold=threshold)
        propstates.append(A_t)

    return propstates
    
def backwardspropogatoin(A_Target, H_drift, H_control_list, pulses, dt, threshold):

    A_t = A_Target
    propstates = [A_t]

    for coeffs in reversed(pulses):
        A_t = propagate_step_exp(A_t, H_drift, H_control_list, coeffs, -dt, threshold=threshold)
        propstates.append(A_t)

    propstates.reverse()

    return propstates 

def compute_gradient_2nd_order_matrix_free(forwardprop, backwardprop, current_pulses, H_drift, H_control_list, dt, threshold=1e-10):
    """
    Calculates the full 2D gradient array (steps x num_controls) of the infidelity without matrices.
    Matches Eq (B11): -b_{l+1}^dagger * {dt*K_k^c - (dt^2/2)[K_l, K_k^c]} * a_l[cite: 22]
    """
    num_controls = len(H_control_list)
    steps = len(current_pulses)
    grad_2d = np.zeros((steps, num_controls))

    norm_target_sq = float(np.sum(np.abs(backwardprop[-1].simplify().coeffs)**2))
    for l in range(steps):
        # State entering the l-th step (matches a_{l-1} in math)[cite: 22]
        A_l = forwardprop[l]
        
        # Backward-propagated state exiting the l-th step (matches b_{l+1} in math)[cite: 22]
        B_l_plus_1 = backwardprop[l]
        
        # we make dictonaries to allow fast serach of matching terms
        #term to match for the b vector 
        dict_back = {str(pauli): coeff for pauli, coeff in zip(B_l_plus_1.paulis, B_l_plus_1.coeffs)}
        Htotal = ComputeHam(H_drift, H_control_list, current_pulses[l])

        #calculate the total term first
        K_l_A = computeCommutator(Htotal, A_l)
        
        for k in range(num_controls):
            H_control_k = SparsePauliOp(H_control_list[k])

            # 1. First order term: i[H_k^c, A_l]
            first_order_A = computeCommutator(H_control_k, A_l)
            
            # 2. The commutator: i[H_total_l, first_order_A] - i[H_control_k, K_l_A]
            commutator_A = computeCommutator(Htotal, first_order_A) - computeCommutator(H_control_k, K_l_A)
            
            # 3. Combine into the full 2nd-order derivative applied to A_l
            derivative_A = (dt * first_order_A) - (0.5 * (dt**2) * commutator_A)
            derivative_A = derivative_A.simplify()
            
            # 4. Overlap (trace) between the backward state and the derivative term
            dict_deriv = {str(pauli): coeff for pauli, coeff in zip(derivative_A.paulis, derivative_A.coeffs)}
            
            # Vectorized overlap using a fast generator expression for matching keys
            overlap = sum(np.real(coeff * np.conj(dict_back[pauli])) 
                          for pauli, coeff in dict_deriv.items() if pauli in dict_back)


            grad_2d[l, k] = -overlap / norm_target_sq
            
    return grad_2d


#THIS IS THE OPTMIZATION LOOP TO BE PUT INTO THE MINIZIATION FUNCITON
def optimization_loop_matrix_free(flat_pulses, A_init, A_target, dt, H_drift, H_control_list, steps, num_controls, threshold=1e-8):
    """
    The main objective function for SciPy minimize.
    Calculates the infidelity and the gradient entirely through Pauli algebra.
    """
    # 1. Reshape the 1D SciPy array back into our 2D (steps x controls) format
    current_pulses = flat_pulses.reshape((steps, num_controls))

    # 2. Forward Propagation and backwards propogation
    # forward_states[l] is the state BEFORE applying pulse l (matches a_{l-1} in math)[cite: 22]
    forwardprop = forwardspropogation(A_init, H_drift, H_control_list, current_pulses, dt, threshold)
    backwardprop = backwardspropogatoin(A_target, H_drift, H_control_list, current_pulses, dt, threshold)


    A_final = forwardprop[-1]
    current_infidelity = compute_infidelity(A_final, A_target)

    # 4. Gradient Calculation

    Grad = compute_gradient_2nd_order_matrix_free(forwardprop, backwardprop, current_pulses, H_drift, H_control_list, dt, threshold=1e-10)

            
    return current_infidelity, Grad.flatten()

#infdeility calcualtion
def compute_infidelity(A_final, A_target):
    """
    Calculates J = 1 - tr(I(T) * I_T) / tr(I_T^2) without full matrices[cite: 1].
    Since tr(P_i P_j) = 2^n if P_i == P_j and 0 otherwise, the trace of the product 
    is directly proportional to the dot product of their matching coefficients.
    """
    # Convert SparsePauliOps to dictionaries for easy coefficient matching
    dict_final = {str(pauli): coeff for pauli, coeff in zip(A_final.paulis, A_final.coeffs)}
    dict_target = {str(pauli): coeff for pauli, coeff in zip(A_target.paulis, A_target.coeffs)}
    
    dot_product = 0.0
    norm_target_sq = 0.0
    
    for pauli, t_coeff in dict_target.items():
        norm_target_sq += np.abs(t_coeff)**2
        
        # If the Pauli string exists in both, multiply their coefficients
        if pauli in dict_final:
            f_coeff = dict_final[pauli]
            dot_product += np.real(f_coeff * np.conj(t_coeff))
            
    # Avoid division by zero if target is empty
    if norm_target_sq == 0:
        return 1.0
        
    return 1.0 - (dot_product / norm_target_sq)


def compute_ground_state_infidelity_pauli(A_final, A_target):
    """
    Extracts the ground states of the final and target operators 
    and computes the quantum state infidelity between them.
    """
    # 1. Convert operators to dense matrices
    matrix_final = A_final.to_matrix()
    matrix_target = A_target.to_matrix()
    
    # 2. Diagonalize to find eigenvalues and eigenvectors
    # np.linalg.eigh is optimized for Hermitian/symmetric matrices
    evals_final, evecs_final = np.linalg.eigh(matrix_final)
    evals_target, evecs_target = np.linalg.eigh(matrix_target)
    
    # 3. Extract the ground state vector (eigenvector corresponding to the lowest eigenvalue)
    gs_final = evecs_final[:, 0]
    gs_target = evecs_target[:, 0]
    
    # 4. Compute state infidelity: 1 - |<psi_final | psi_target>|^2
    overlap = np.vdot(gs_final, gs_target) # Complex dot product
    state_infidelity = 1.0 - (np.abs(overlap) ** 2)
    
    return state_infidelity