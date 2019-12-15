import numpy as np
def numpy_softmax(input,axis=-1):
    exp=np.exp(input)
    sum=np.repeat(np.sum(exp,axis=axis,keepdims=True),repeats=48,axis=1)
    result=np.divide(exp,sum)
    return result



if __name__=='__main__':
    r=np.random.randn(32,48)
    numpy_softmax(r)