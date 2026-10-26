

#LinearSpinChainSetup

Type = "Z"
Type2 = "X"
N=3
activeSystem = "linearSpinChain"

def linearTerms(N, Type):
    base = "I" * N
    return [base[:i] + Type + base[i+1:] for i in range(N)]

def interactionTerms(N, Type):
    base = "I" * N
    type2 = Type * 2
    return [base[:i] + type2 + base[i+2:] for i in range(N - 1)]


controlSets = {
    "linearSpinChain": linearTerms(N,Type)+interactionTerms(N,Type2)+["I"*(N-1)+Type2]+[Type2+"I"*(N-1)]
}
