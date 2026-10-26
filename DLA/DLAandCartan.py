from qiskit.quantum_info import Pauli
import Config

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



def cartanDecomp(lieAlgebra):
    even_y_terms = [term for term in lieAlgebra if term.count('Y') % 2 == 0] #ie set m
    odd_y_terms = [term for term in lieAlgebra if term.count('Y') % 2 != 0] # ie set l
    return even_y_terms, odd_y_terms


lieAlgebra = { i : Pauli(i) for i in Config.controlSets[Config.activeSystem]}
lieAlgebra = DLA(lieAlgebra)
print(len(lieAlgebra))