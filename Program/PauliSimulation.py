
import numpy as np
import Config
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp


#THESE 4 FUNCTIONS TAKE YOUR HAMTILTIAN AND OPERATOR AND CALCULATE THE TIME EVOLUTION OF THE OPERATOR ACCORDING TO HEISNBERGS EQUATION USING RUNGE KUTTA
#creates an inital operator as a pauli object
def create_initial_operator():

    pauli_strings = Config.StartingVectorTerms
    coeffs = Config.StartingVectorcoeff

    return SparsePauliOp(pauli_strings, coeffs)

#calculates the total hamiltian taking the drift hamtilian already multipled by its interaction strength and each control term to multiply by its coeff
def ComputeHam(Hdrift, HcontrolTerms, HControlCoeff):

    H_total = Hdrift

    for H_c, coeff in zip(HcontrolTerms, HControlCoeff):
        H_total = H_total + (coeff * H_c)  

    return H_total.simplify(atol=1e-10)

#calculates the time derative of the invarient operator using its EOM
def computeOperatorGradient(Hamiltonian, InvarientOp):

    commutator = (Hamiltonian @ InvarientOp) - (InvarientOp @ Hamiltonian)
    
    return (-1j * commutator).simplify(atol=1e-10)


def propagate_step_rk4_matrix_free(A_t, H_drift, H_control_list, control_coeffs, dt, threshold=1e-10):
    """
    Evolves the operator A(t) forward by one timestep dt using 4th-order Runge-Kutta (RK4).
    Since the control pulses are piece-wise constant, H(t) is constant over this dt.
    """
    # 1. Build the total Hamiltonian for this specific timestep[cite: 1]
    H_total = ComputeHam(H_drift,H_control_list,control_coeffs)

    # 2. Compute the RK4 slopes (k1, k2, k3, k4)
    # k1 = f(t, A_t)
    k1 = computeOperatorGradient(H_total, A_t)
    
    # k2 = f(t + dt/2, A_t + dt/2 * k1)
    # We MUST simplify here to prevent combinatorial explosion of Pauli strings
    A_k2 = (A_t + (dt / 2.0) * k1).simplify(atol=threshold)
    k2 = computeOperatorGradient(H_total, A_k2)
    
    # k3 = f(t + dt/2, A_t + dt/2 * k2)
    A_k3 = (A_t + (dt / 2.0) * k2).simplify(atol=threshold)
    k3 = computeOperatorGradient(H_total, A_k3)
    
    # k4 = f(t + dt, A_t + dt * k3)
    A_k4 = (A_t + dt * k3).simplify(atol=threshold)
    k4 = computeOperatorGradient(H_total, A_k4)
    
    # 3. Combine to get the next operator state
    A_next = A_t + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
    
    # 4. Final pruning for the complete step
    return A_next.simplify(atol=threshold)




#THESE 3 CALCULATE THE GRADIETN WITH RESPECT TO EAHC CONTORL TERM LIKE THE PAPER
def forwardspropogation(A_init, H_drift, H_control_list, pulses, dt, threshold):

    A_t = A_init
    propstates = [A_t]

    for coeffs in pulses:
        A_t = propagate_step_rk4_matrix_free(A_t, H_drift, H_control_list,coeffs, dt, threshold)
        propstates.append(A_t)

    return propstates
    
def backwardspropogatoin(A_Target, H_drift, H_control_list, pulses, dt, threshold):

    A_t = A_Target
    propstates = [A_t]

    for coeffs in reversed(pulses):
        A_t = propagate_step_rk4_matrix_free(A_t, H_drift, H_control_list, coeffs, -dt, threshold)
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
        K_l_A = computeOperatorGradient(Htotal, A_l)
        
        for k in range(num_controls):
            H_control_k = H_control_list[k]

            # 1. First order term: i[H_k^c, A_l]
            first_order_A = computeOperatorGradient(H_control_k, A_l)
            
            # 2. The commutator: i[H_total_l, first_order_A] - i[H_control_k, K_l_A]
            commutator_A = computeOperatorGradient(Htotal, first_order_A) - computeOperatorGradient(H_control_k, K_l_A)
            
            # 3. Combine into the full 2nd-order derivative applied to A_l
            derivative_A = (dt * first_order_A) - (0.5 * (dt**2) * commutator_A)
            derivative_A = derivative_A.simplify(atol=threshold)
            
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
