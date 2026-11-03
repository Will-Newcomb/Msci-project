import numpy as np

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
