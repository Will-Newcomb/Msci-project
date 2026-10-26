from qiskit.quantum_info import Pauli

def get_dla():
    """Returns the Dynamical Lie Algebra (DLA) basis as a dictionary of Pauli objects."""
    pauli_strings = [
        'ZII', 'IZI', 'IIZ', 'XXI', 'IXX', 'XII', 'IIX', 'YXI', 'YII', 'XYI', 
        'IYX', 'IXY', 'IIY', 'YYI', 'ZXI', 'YZX', 'ZYI', 'XZX', 'XZY', 'IYY', 
        'IYZ', 'IXZ', 'YZY', 'YZZ', 'ZZX', 'ZZY', 'ZZZ', 'XZZ'
    ]
    # Use dictionary comprehension to avoid typing `Pauli()` 28 times
    return {p: Pauli(p) for p in pauli_strings}


def vectorMaker(targetOps,coeffs):

    # Pre-fill a dictionary with 0 for all operators in the DLA
    target_dict = {key: 0 for key in get_dla()}
    
    # Update only the target operators with the provided coefficients
    for op, coeff in zip(targetOps, coeffs):
        if op in target_dict:
            target_dict[op] = coeff
            
    return target_dict






