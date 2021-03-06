import numpy as np
from numba import jitclass,int32, float64,void
spec = [
    ('c', float64[:,:]),
    ('b0', float64),
    ('b1', float64),
    ('epsilon', float64),
    ('size',int32),
    ('dright', float64[:]),
    ('dleft', float64[:]),
    ('reconstructed', float64[:]),
    ('numflux', float64[:]),
    ('beta', float64[:]),
    ('right', float64[:]),
    ('left', float64[:]),
    ('alpharight', float64[:]),
    ('alphaleft', float64[:]),
    ('InC', float64[:]),
    ('L', float64),
    ]
@jitclass(spec)
class Weno(object):
    def __init__(self,size,L):
        self.c=np.array([11./6.,-7./6.,1./3.,1./3.,5./6.,-1./6.,
                         -1./6.,5./6.,1./3.,1./3.,-7/6.,11./6.]).reshape(4,3)
        self.b0=13./12.
        self.b1=1./4.
        self.epsilon=1.e-6
        self.size=size
        # allocate all arrays...
        self.dright=np.array([3./10.,3./5.,1./10.])
        self.dleft=np.array([1./10.,3./5.,3.10])
        self.reconstructed=np.empty(2*size+8)
        self.numflux=np.empty(size+2)
        self.beta=np.empty(3)
        self.right=np.empty(3)
        self.left=np.empty(3)
        self.alpharight=np.empty(3)
        self.alphaleft=np.empty(3)
        self.InC=np.empty(size+4)
        self.L=L

    def weno(self,F,In,Out):
        size=self.size
        h1= -1./(self.L/size)
        # build an extended array with phantom cells to deal with periodicity:
        self.InC[0]=In[size-2]
        self.InC[1]=In[size-1]
        self.InC[2:2+size]=In
        self.InC[size+2]=In[0]
        self.InC[size+3]=In[1]
        # precompute for regularity coefficients (use numflux as auxiliary
        # array).
        Reg=self.numflux
        Reg[0:size+2]=self.b0*np.power(
            self.InC[0:size+2]-2.0*self.InC[1:size+3]+self.InC[2:size+4],2)
        for vol in range(2,size+2):
            #reconstructions right & left:
 
            for r in range(0,3):
                cr=self.c[r+1]
                cl=self.c[r]
                self.right[r]=0.0
                self.left[r]=0.0
                
                for j in range(0,3):
                    sinc=self.InC[vol-r+j]
                    self.right[r]+= cr[j]*sinc#self.InC[vol-r+j]
                    self.left[r]+=cl[j]*sinc#self.InC[vol-r+j]

            # regularity coefficients
            self.beta[0]= Reg[vol]+ self.b1* \
                                      pow(3.*self.InC[vol]-4.*self.InC[vol+1]+\
                                          self.InC[vol+2],2)

            self.beta[1]= Reg[vol-1]+ self.b1*\
                                     pow(self.InC[vol-1]-self.InC[vol+1],2)
            
            self.beta[2]= Reg[vol-2]+ self.b1*\
                                     pow(self.InC[vol-2]-4.*self.InC[vol-1]+\
                                         3*self.InC[vol],2) 
            sright= 0.0
            sleft=  0.0
            for r in range(0,3):
                dv=pow(self.epsilon+self.beta[r],-2)
                self.alpharight[r]=self.dright[r]*dv
                self.alphaleft[r]=self.dleft[r]*dv
                # self.alpharight[r]=self.dright[r]/pow(self.epsilon+\
                #                                       self.beta[r],2)
                # self.alphaleft[r]=self.dleft[r]/pow(self.epsilon+\
                #                                     self.beta[r],2)
                sright+=self.alpharight[r]
                sleft+=self.alphaleft[r]
         
            recleft=np.dot(self.alphaleft,self.left)
            recright=np.dot(self.alpharight,self.right)


            #reconstructed values:
            self.reconstructed[2*vol]  = recleft/sleft
            self.reconstructed[2*vol+1]= recright/sright
        self.reconstructed[2*size+4:2*size+8]=self.reconstructed[4:8]
        #compute the numerical fluxes at boundaries:
        for vol in range(1,size+1):
            self.numflux[vol]=F(self.reconstructed[2*vol+3],
                                self.reconstructed[2*vol+4])
        self.numflux[0]=self.numflux[size]
        # #now, return RHS to solver:
        for vol  in range(0,size):
            Out[vol]=h1*(self.numflux[vol+1]-self.numflux[vol])
        #Out[0:size]=h1*self.numflux[1:size+1]-h1*self.numflux[0:size]
