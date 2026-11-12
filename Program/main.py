import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli

# Import your own modules
import Config
import Tools.Tools as Tools
import MatrixVer.DLA_Cartan_Strcture as dla
import MatrixVer.Simulation as Simulation  # <--- Importing your new math module



# --- Initialization and DLA Setup ---

def main():
    timestep = Config.Timestep
    steps = Config.Steps
    activeSystem = Config.activeSystem



    # Generate control excluding drift terms
    controlAlgebra_dict = {i: Pauli(i) for i in Config.controlSets[activeSystem]}
    controlAlgebra_list = list(controlAlgebra_dict.values())


    #take set of drift terms
    DriftAlgebra_dict = {i: Pauli(i) for i in Config.driftSets[activeSystem]}
    DriftAlgebra_list = list(DriftAlgebra_dict.values())

    #generate full DLA by combing the dictonarys and keep a list of them with an index mapping for quick checks
    full_algebra_dict = {**DriftAlgebra_dict, **controlAlgebra_dict}
    lieAlgebradict = dla.DLA(full_algebra_dict)
    index_map, pauliLiearray = dla.convert(lieAlgebradict)

    #create the basis set of the K matrix needs index map to make strcture constants as dicontary loopkups and vectorised calculations 
    #outputs a list of matrices in order of the contol list and dimensional order of the pualiliearray/index map

    K_drift_matrices = Simulation.KCreation(DriftAlgebra_list, pauliLiearray, index_map)
    K_control_matrices =Simulation.KCreation(controlAlgebra_list, pauliLiearray, index_map)

    #summing all the drift k matrices
    kDrift = Simulation.DriftK(K_drift_matrices,Config.driftCoeff)


    #create the inital 2d guess array for the basis elemnts of the K matrix ie arrys of guessed values for d time steps
    #arrays of size steps and then one value for eahc contorl element
    initial_pulses = np.random.uniform(-0.1, 0.1, size=(steps, len(controlAlgebra_list)))
    flat_initial_guess = initial_pulses.flatten()

    #creat the starting objects of the evolution takes index map to match our chosen elements to the right location
    StratingVector = Tools.vectorMaker(Config.StartingVectorTerms,Config.StartingVectorcoeff, pauliLiearray, index_map)
    TargetVector = Tools.vectorMaker(Config.TargetVectorTerms,Config.TargetVectorCoeff,pauliLiearray, index_map)


    #needed for the minimse funciton to recreat our 2d array apprently it just dosent take them but uses them?
    num_controls = len(controlAlgebra_list)

    result = minimize(
        fun=Simulation.optimizationloop,
        x0=flat_initial_guess,
        args=(StratingVector, TargetVector, timestep, kDrift, K_control_matrices, steps, num_controls),
        method='L-BFGS-B',
        jac=True, # Tells SciPy that our wrapper returns the gradient as the second output
        options={'maxiter': 1000}
    )


    optimized_pulses = result.x.reshape((steps, num_controls))
    final_infidelity = result.fun


    Finalop = Simulation.propagate_operator_sparse(StratingVector, optimized_pulses, timestep, kDrift,K_control_matrices)
    stateinfil = Simulation.compute_ground_state_infidelity_dla(Finalop[-1], TargetVector, pauliLiearray)
    print(stateinfil)


    timeaxis = np.linspace(0,timestep*steps,steps)
    print(len(timeaxis))
    plt.plot(timeaxis,optimized_pulses[:, 1], label = controlAlgebra_list[1].to_label())
    plt.plot(timeaxis,optimized_pulses[:, 0], label = controlAlgebra_list[0].to_label())
    plt.plot(timeaxis,optimized_pulses[:, 2], label = controlAlgebra_list[2].to_label())
    plt.plot(timeaxis,optimized_pulses[:, 3], label = controlAlgebra_list[3].to_label())
    plt.legend()
    plt.show()


    plt.plot(timeaxis,optimized_pulses[:, 4], label = controlAlgebra_list[4].to_label())
    plt.plot(timeaxis,optimized_pulses[:, 5], label = controlAlgebra_list[5].to_label())
    plt.legend()


    plt.show()
    print(final_infidelity)


if __name__ == "__main__":
    main()