import numpy as np
import time
from cl_1 import *
from cl_2 import *

def Init(X,L):
    size=X.size
    h=L/size
    for i in range(0,size):
        if i>size//8 and i<size//2+size//8:
            X[i]=1.-2*(i-size//8)*h/L;
        else:
            X[i]=0.0
              
def test(p,A,B,C,D,nit):
   
    niter=nit
    Init(A,1.)
    Init(B,1.)
    Init(C,1.)
    Init(D,1.)
    p(A,B,C,D,niter)
    T=0.
    while True:
        Init(A,1.)
        Init(B,1.)
        Init(C,1.)
        Init(D,1.)
        t1 = time.time()
        p(A,B,C,D,niter)
        treal=time.time() -t1
        t = treal/niter
        if treal>0.0001 and abs(t-T)/t<0.025:
            break
        else:
            T=t
            niter*=2

    return T,niter

size=16
sizemax=100000
niter=10
parsef= lambda  f: str(f).split(" ")[1] #parse function name
while size<sizemax:
    print("size: ",size)
    A= np.empty(size)
    B= np.empty(size)
    C= np.empty(size)
    D= np.empty(size)
    tbest=10.**20
    best=0
    t=0.0
    for p in  [cl_1,cl_2]:
        t,it=test(p,A,B,C,D,niter)
        if t<tbest:
            tbest=t
            best=p
        print(parsef(p)," : t= ",t," seconds ")
    nflops= 4*(size-2)
    flops=nflops/tbest
    print("\nbest: ",parsef(best))
    print("nb. flops (best): ",nflops, ", Gflops/s: ",flops/(10**9))
    print("-------")
    size*=2
    print(" ")