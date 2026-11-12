from qiskit.quantum_info import Pauli, PauliList
import numpy as np

def DLA(lieAlgebra):
    newLiecomparisons = lieAlgebra.copy()
    #need to update to avoid redundant calculations like ab after ba or aa also apparently qiskit pauli is slow?
    while newLiecomparisons:
        newElements = {}

        for comparisonKey, comparisonElement in newLiecomparisons.items():
            for lieElementKey, lieElement in lieAlgebra.items():

                if not comparisonElement.commutes(lieElement):
                    newElement = comparisonElement@lieElement
                    newElement.phase = 0
                    label = newElement.to_label()

                    if not label in lieAlgebra and label not in newElements:
                        newElements[label] = newElement

        lieAlgebra.update(newElements)
        newLiecomparisons = newElements

    return lieAlgebra


def convert(lieAlgebra):
    """
    Takes a dictionary of {label: Qiskit Pauli}
    Returns:
      - index_map: dict mapping label string to its integer index
      - pauli_array: list containing the actual Qiskit Pauli objects
    """
    index_map = {}
    pauli_array = []
    
    for idx, (label, pauli) in enumerate(lieAlgebra.items()):
        index_map[label] = idx
        pauli_array.append(pauli)
    pauli_array = PauliList(pauli_array)
    return index_map, pauli_array


def get_structure_constant(DLA, ControlTerm, indexmap):

    # 1. Vectorized commutation check
    commuteCheck = DLA.commutes(ControlTerm)

    # 2. Get exact column indices 'j' where they anti-commute
    antiCommuteIdx= np.where(~commuteCheck)[0]

    
    # 3. Vectorized multiplication for all non-zero terms simultaneously
    CommuteProduct = DLA[antiCommuteIdx] @ ControlTerm
    cols = antiCommuteIdx # These are the 'j' indices



    #takes the phase of each and as multiplcation by another pauli we can work out the final result from the frist commutator
    phases = CommuteProduct.phase
    #sets all others to 0 follwed by setting specifc values
    vals = np.zeros(len(CommuteProduct))
    vals[phases == 1] = 2.0
    vals[phases == 3] = -2.0


    # 4. Map results to matrix coordinates
    CommuteProduct.phase = 0
    labels = CommuteProduct.to_labels()
    rows = np.array([indexmap[label] for label in labels])

    return rows, cols, vals



def cartanDecomp(lieAlgebra):
    even_y_terms = [term for term in lieAlgebra if term.count('Y') % 2 == 0] #ie set m
    odd_y_terms = [term for term in lieAlgebra if term.count('Y') % 2 != 0] # ie set l
    return even_y_terms, odd_y_terms





