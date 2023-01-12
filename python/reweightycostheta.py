from module import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from math import pi, sqrt
from root_numpy import hist2array
# np.set_printoptions(threshold=sys.maxsize)

class reweightycostheta(module):
   
    def __init__(self, isData=False, isJ=False):
        self.isData=isData
        self.isJ=isJ
        pass
      

    def run(self,d):
        if not self.isData:
            if self.isJ:
                fileJPsiData = h5py.File('JPsiInputData/JPsiData_mukin_rew1.hdf5', mode='r')
                fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin_rew1.hdf5', mode='r')
            else:
                fileJPsiData = h5py.File('JPsiInputData/JPsiData_mukin_Z.hdf5', mode='r')
                fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin_Z.hdf5', mode='r')
            
            hdata_pt = np.sum(fileJPsiData['Jpsi_rew_data'][:],axis=0)
            hmc_pt = np.sum(fileJPsiMC['Jpsi_rew_mc'][:],axis=0)
            Jraps = np.linspace(-2.4, 2.4, 120, dtype='float64')
            costhetas = np.array([round(-1. + 2*i/100,2) for i in range(101)])

            @ROOT.Numba.Declare(["float","float"], "float")
            def getWeighty(costheta, rap):
                bincostheta = np.digitize(np.array([costheta]), costhetas)[0]-1
                binrap = np.digitize(np.array([rap]), Jraps)[0]-1
                if hdata_pt[bincostheta, binrap]==0. or hmc_pt[bincostheta, binrap]==0.:
                    return -99.
                if abs(rap)>2.4 or abs(costheta)>1.:
                    return -99.
                else:
                    w = hdata_pt[bincostheta, binrap]/hmc_pt[bincostheta, binrap]
                    return w

            self.d = d
            self.d = self.d.Define("jrapweight", "Numba::getWeighty(CStheta, Jpsi_rap)")

        else:
            self.d = d
            self.d = self.d.Define("jrapweight", "float(1.)")

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
