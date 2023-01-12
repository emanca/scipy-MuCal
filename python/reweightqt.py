from module import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from math import pi, sqrt
from root_numpy import hist2array

class reweightqt(module):
   
    def __init__(self, isData=False, isJ=False):
        self.isData=isData
        self.isJ=isJ
        pass
      

    def run(self,d):
        if not self.isData:
            if self.isJ:
                fileJPsiData = h5py.File('JPsiInputData/JPsiData_mukin_rew2.hdf5', mode='r')
                fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin_rew2.hdf5', mode='r')
                Jpts=np.linspace(0,50,101)
            else:
                fileJPsiData = h5py.File('JPsiInputData/JPsiData_mukin_Z2.hdf5', mode='r')
                fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin_Z2.hdf5', mode='r')
                Jpts=np.linspace(0,100,101)
            
            hdata_pt = np.sum(fileJPsiData['Jpsi_rew_data'][:], axis=(1,2))/np.sum(fileJPsiData['Jpsi_rew_data'][:])
            hmc_pt = np.sum(fileJPsiMC['Jpsi_rew_mc'][:], axis=(1,2))/np.sum(fileJPsiMC['Jpsi_rew_mc'][:])

            print(hdata_pt/hmc_pt)

            @ROOT.Numba.Declare(["float"], "float")
            def getWeightqt(jpt):
                binpt = np.digitize(np.array([jpt]), Jpts)[0]-1
                if jpt>100:
                    w = 1.
                # elif jpt<4:
                #     w=0.
                else:
                    w = hdata_pt[binpt]/hmc_pt[binpt]
                return w

            self.d = d
            self.d = self.d.Define("jptweight", "Numba::getWeightqt(Jpsi_pt)")
        else:
            self.d = d
            self.d = self.d.Define("jptweight", "float(1.)")

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
