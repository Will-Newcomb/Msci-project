
import numpy as np
import Config
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList
import PauliVer.PauliSimulation as ps



def main():
    # --- Problem Setup ---
    dt = Config.Timestep
    steps = Config.Steps
    activesystem = Config.activeSystem

    #makeing the full drift hamiltionian including the drift coefficant
    driftTerms = Config.driftSets[activesystem]
    HDrift = SparsePauliOp(driftTerms, coeffs=Config.driftCoeff)

    #making the control set for updating 
    HcontrolTerms = Config.controlSets[activesystem]
    HcontrolList = PauliList(HcontrolTerms)
    #varible needed for future
    num_controls = len(HcontrolTerms)

    #making Initial and Target Operators
    initialVectorTerms = Config.StartingVectorTerms
    initialVectorCoeffs = Config.StartingVectorcoeff
    initInvarient = SparsePauliOp(initialVectorTerms, initialVectorCoeffs)

    targetVectorTerms = Config.TargetVectorTerms
    targetVectorCoeffs = Config.TargetVectorCoeff   
    targetInvarient = SparsePauliOp(targetVectorTerms, coeffs=targetVectorCoeffs)

    #Initial Guess need to be flattered for scipy optimization
    initial_pulses = np.random.uniform(-0.1, 0.1, size=(steps, num_controls))
    flat_initial_guess = initial_pulses.flatten()

    #running the minization function
    result = minimize(
        fun=ps.optimization_loop_matrix_free,
        x0=flat_initial_guess,
        args=(initInvarient, targetInvarient, dt, HDrift, HcontrolList, steps, num_controls, 1e-8),
        method='L-BFGS-B',
        jac=True, 
        options={'maxiter': 50}
    )

    optimized_pulses = result.x.reshape((steps, num_controls))
    final_infidelity = result.fun

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
