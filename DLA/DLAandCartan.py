from qiskit.quantum_info import Pauli, commutator


#starting control terms for a linear spin chain with 3 qbits
def controlHam():
    return ["ZII","IZI","IIZ","XXI","IXX","XII","IIX"]



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


lieAlgebra = { i : Pauli(i) for i in controlHam()}
lieAlgebra = DLA(lieAlgebra)
print(cartanDecomp(lieAlgebra))