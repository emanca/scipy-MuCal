from module import *
import numpy as np
import math

with np.load('unbinnedfitglobalitercorscale.npz') as f:
# with np.load('/scratchnvme/emanca/scipy-MuCal/unbinnedfitglobalitercorscale_z.npz') as f:
    d = f["xs"][...,-1]
    b = f["xs"][...,5]

etas = np.linspace(-2.4,2.4,49)

@ROOT.Numba.Declare(['float'], 'float')
def computeTrackLength(eta):
    L0=108.-4.4 #max track length in cm. tracker radius - first pixel layer
    if(abs(eta)<=1.4):
        L=L0
    else:
        tantheta = 2/(np.exp(eta)-np.exp(-eta))
        r = 267.*tantheta #267 cm: z position of the outermost disk of the TEC 
        if(eta>1.4):
            L=min(r, 108.)-4.4
        else:
            L=min(-r, 108.)-4.4
    return (L0/L)**2

@ROOT.Numba.Declare(['float'], 'float')
def invpt(pt):
    return 1./pt

@ROOT.Numba.Declare(['float','float'], 'float')
def invptsqtimesd(eta,pt2):
    bineta = np.digitize(np.array([eta]), etas)[0]-1
    return 1./(1.+d[bineta]*d[bineta]/pt2)

@ROOT.Numba.Declare(['float'], 'float')
def computesq(x):
    return x**2

@ROOT.Numba.Declare(['float','float'], 'float')
def logratio(pt1,pt2):
    return np.log(pt1/pt2)

@ROOT.Numba.Declare(['float','float'], 'float')
def cosdeltaphi(phi1,phi2):
    result = phi1 - phi2
    if result>math.pi:
        result -= float(2 * math.pi)
    if result<=math.pi:
        result += float(2 * math.pi)
    return np.cos(result)

@ROOT.Numba.Declare(['float','float','float','float','float','float','float','float','float','float'], 'RVec<float>')
def createRVec(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
    return np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

class defines(module):
   
    def __init__(self,isData=False):
        self.isData=isData
        pass

    def run(self,d):

        self.d = d.Define('Lplus','Numba::computeTrackLength(Muplus_eta)')\
            .Define('Lminus','Numba::computeTrackLength(Muminus_eta)')\
            .Define('ptplussq', 'Numba::computesq(Muplus_pt)')\
            .Define('ptminussq', 'Numba::computesq(Muminus_pt)')\
            .Define('invplus', 'Numba::invpt(Muplus_pt)')\
            .Define('invminus', 'Numba::invpt(Muminus_pt)')\
            .Define('invplussq', 'Numba::invptsqtimesd(Muplus_eta,ptplussq)')\
            .Define('invminussq', 'Numba::invptsqtimesd(Muminus_eta,ptminussq)')\
            .Define('calVariables','Numba::createRVec(Lplus,Lminus,invplus,invminus,Muplus_pt,Muminus_pt,ptplussq,ptminussq,invplussq,invminussq)')\
            .Define('cosdeltaphi', 'Numba::cosdeltaphi(Muplus_phi,Muminus_phi)')\
            .Define('logratiopt', 'Numba::logratio(Muplus_pt,Muminus_pt)')
        if not self.isData:
            self.d=self.d.Define('Lplusgen','Numba::computeTrackLength(Muplusgen_eta)')\
            .Define('Lminusgen','Numba::computeTrackLength(Muminusgen_eta)')\
            .Define('ptplussqgen', 'Numba::computesq(Muplusgen_pt)')\
            .Define('ptminussqgen', 'Numba::computesq(Muminusgen_pt)')\
            .Define('invplusgen', 'Numba::invpt(Muplusgen_pt)')\
            .Define('invminusgen', 'Numba::invpt(Muminusgen_pt)')\
            .Define('calVariablesgen','Numba::createRVec(Lplusgen,Lminusgen,invplusgen,invminusgen,Muplusgen_pt,Muminusgen_pt,ptplussqgen,ptminussqgen,invplussq,invminussq)')\
            # .Define('v1', 'ROOT::Math::PtEtaPhiMVector(Muplus_pt,Muplus_eta,Muplus_phi,0.105658)')\
            # .Define('v2', 'ROOT::Math::PtEtaPhiMVector(Muminus_pt,Muminus_eta,Muminus_phi,0.105658)')\
            # .Define('Jpsi_rap', 'float((v1+v2).Rapidity())')\
            
        return self.d

    def getTH1(self):

        return self.myTH1

    def getTH2(self):

        return self.myTH2  

    def getTH3(self):

        return self.myTH3

    def getTHN(self):

        return self.myTHN

    def getGroupTH1(self):

        return self.myTH1Group

    def getGroupTH2(self):

        return self.myTH2Group  

    def getGroupTH3(self):

        return self.myTH3Group  

    def getGroupTHN(self):

        return self.myTHNGroup

    def reset(self):

        self.myTH1 = []
        self.myTH2 = []
        self.myTH3 = [] 
        self.myTHN = [] 

        self.myTH1Group = []
        self.myTH2Group = []
        self.myTH3Group = [] 
        self.myTHNGroup = [] 
