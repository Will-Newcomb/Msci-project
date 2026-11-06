import numpy as np

#Evolution properties
Timestep = 0.1
Steps = 70



#LinearSpinChainSetup

Type = "Z"
Type2 = "X"
N=5
activeSystem = "linearSpinChain"




def linearTerms(N, Type):
    base = "I" * N
    return [base[:i] + Type + base[i+1:] for i in range(N)]

def interactionTerms(N, Type):
    base = "I" * N
    type2 = Type * 2
    return [base[:i] + type2 + base[i+2:] for i in range(N - 1)]



driftSets = {
    "linearSpinChain": interactionTerms(N, Type2) 
}

controlSets = {
    "linearSpinChain": linearTerms(N,Type)+["I"*(N-1)+Type2]+[Type2+"I"*(N-1)]
}




#Problem setup
StartingVectorTerms = linearTerms(N,Type)
StartingVectorcoeff = np.ones(N)



TargetVectorTerms = interactionTerms(N, Type2) + ["Z"*N]
TargetVectorCoeff = -np.ones(N)

driftCoeff=1