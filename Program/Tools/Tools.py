import numpy as np


#Creates the intial vector we want to start with have a sepeate function for this as human input migth get the letters in the wong place
#is also useful for jus
# t populaitng it ie for different cases
#Tools to make vectors of ceofficants for a given order of elements
def vectorMaker(targetOps, coeffs, pauliLiearray, index_map):
    #0 vector calculation
    d = len(pauliLiearray)
    a_vector = np.zeros(d, dtype=float)

    #uses stongs of the input target to get the index
    for op, coeff in zip(targetOps, coeffs):
        op_str = str(op)
        if op_str in index_map:
            idx = index_map[op_str]
            a_vector[idx] = coeff

    return a_vector


#Tools to make the linear and interaction terms for N qubits wihtout having to write out manually
def linearTerms(N, Type):
    base = "I" * N
    return [base[:i] + Type + base[i+1:] for i in range(N)]

def interactionTerms(N, Type):
    base = "I" * N
    type2 = Type * 2
    return [base[:i] + type2 + base[i+2:] for i in range(N - 1)]

