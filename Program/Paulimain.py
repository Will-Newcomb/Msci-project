
import numpy as np
import Config
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
import PauliSimulation as ps

def main():
    # --- Problem Setup ---
    dt = Config.Timestep
    steps = Config.Steps

    # --- Define Drift and Control Hamiltonians (Matrix-Free) ---
    driftTerms = Config.driftSets[Config.activeSystem]
    HDrift = SparsePauliOp(driftTerms, coeffs=Config.driftCoeff)
    
    HcontrolTerms = Config.controlSets[Config.activeSystem]
    HcontrolList = [SparsePauliOp([i], coeffs=[1.0]) for i in HcontrolTerms]

    num_controls = len(HcontrolTerms)

    # --- Define Initial and Target Operators ---
    initialVectorTerms = Config.StartingVectorTerms
    initialVectorCoeffs = Config.StartingVectorcoeff
    A_init = SparsePauliOp(initialVectorTerms, initialVectorCoeffs)

    targetVectorTerms = Config.TargetVectorTerms
    targetVectorCoeffs = Config.TargetVectorCoeff   
    A_target = SparsePauliOp(targetVectorTerms, coeffs=targetVectorCoeffs)

    # --- Initial Guesses ---
    initial_pulses = np.random.uniform(-0.1, 0.1, size=(steps, num_controls))
    flat_initial_guess = initial_pulses.flatten()
    
    # --- Run Optimizer ---
    print("Starting matrix-free Pauli algebra optimization...")


    #first question THRESHOLDS???
    result = minimize(
        fun=ps.optimization_loop_matrix_free,
        x0=flat_initial_guess,
        args=(A_init, A_target, dt, HDrift, HcontrolList, steps, num_controls, 1e-8),
        method='L-BFGS-B',
        jac=True, 
        options={'maxiter': 50}
    )

    # --- Process and Plot Results ---
    optimized_pulses = result.x.reshape((steps, num_controls))
    final_infidelity = result.fun
    
    print(f"\nOptimization Completed.")
    print(f"Final Infidelity: {final_infidelity}")

    # Plot
    timeaxis = np.linspace(0, dt * steps, steps)
    for i in range(num_controls):
        plt.plot(timeaxis, optimized_pulses[:, i], label=f'Control {i}')
        
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Pulse Amplitude")
    plt.title("Optimized Control Pulses (Matrix-Free)")
    plt.show()

if __name__ == "__main__":
    main()
