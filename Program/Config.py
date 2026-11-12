import numpy as np
import Tools.Tools as Tools

#System properties
activeSystem = "linearSpinChain"
NQubits=4

#Evolution properties
Timestep = 0.08
Steps = 100

#LinearSpinChainProperties
singleControlType = "Z"
interactionType = "X"
endChainType = "X"
driftCoeff=1


#Problem setup
StartingVectorTerms = Tools.linearTerms(NQubits,singleControlType)
StartingVectorcoeff = np.ones(NQubits)

TargetVectorTerms = Tools.interactionTerms(NQubits, interactionType) + ["Z"*NQubits]
TargetVectorCoeff = -np.ones(NQubits)


#Sets of different system artichetures
driftSets = {
    "linearSpinChain": Tools.interactionTerms(NQubits, interactionType) 
}

controlSets = {
    "linearSpinChain": Tools.linearTerms(NQubits,singleControlType)+["I"*(NQubits-1)+endChainType]+[endChainType+"I"*(NQubits-1)]
}




