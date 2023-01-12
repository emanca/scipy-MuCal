from module import *
import h5py
import pickle
import lz4.frame
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from math import pi, sqrt
ROOT.gInterpreter.Declare('#include "helper_rew.h"')

class reweightycostheta(module):
   
    def __init__(self, isData=False, isJ=False):
        self.isData=isData
        self.isJ=isJ
        pass
    
    def array2hist(self,array, hist, err):
        if type(hist).__cpp_name__=='TH3D':
            for i in range(1,hist.GetNbinsX()+1):
                for j in range(1,hist.GetNbinsY()+1):
                    for k in range(1,hist.GetNbinsZ()+1):
                        hist.SetBinContent(i,j,k,array[i-1,j-1,k-1])
                        hist.SetBinError(i,j,k,err[i-1,j-1,k-1])
        elif type(hist).__cpp_name__=='TH2D':
            for i in range(1,hist.GetNbinsX()+1):
                for j in range(1,hist.GetNbinsY()+1):
                    hist.SetBinContent(i,array[i-1,j-1])
                    hist.SetBinError(i,err[i-1,j-1])
        elif type(hist).__cpp_name__=='TH1D':
            for i in range(1,hist.GetNbinsX()+1):
                hist.SetBinContent(i,array[i-1])
                hist.SetBinError(i,err[i-1])
        else:
            print(hist.GetName(),type(hist).__cpp_name__)
            print("type not recognized")
        return hist
    
    def run(self,d):
        if self.isJ:
            fileJPsiData = 'JPsiInputData/JPsiDatatgr.pkl.lz4'
            fileJPsiMC = 'JPsiInputData/JPsiMCtgr.pkl.lz4'
            with (lz4.frame.open(fileJPsiData, "r")) as openfile:
                resultdict_data = pickle.load(openfile)
            with (lz4.frame.open(fileJPsiMC, "r")) as openfile:
                resultdict_mc = pickle.load(openfile)
            
            hdata_pt = resultdict_data['Jpsi_data_bosonkin'].values()#/npsum(resultdict_data.values())
            hmc_pt = resultdict_mc['Jpsi_mc_bosonkin'].values()#/np.su(resultdict_mc.values())
        else:
            fileJPsiData = 'JPsiInputData/ZDatarew.pkl.lz4'
            fileJPsiMC = 'JPsiInputData/ZMCrew.pkl.lz4'
            with (lz4.frame.open(fileJPsiData, "r")) as openfile:
                resultdict_data = pickle.load(openfile)
            with (lz4.frame.open(fileJPsiMC, "r")) as openfile:
                resultdict_mc = pickle.load(openfile)
            hdata_pt = resultdict_data['Z_data_bosonkin'].values()#/np.su(resultdict_data.values())
            hmc_pt = resultdict_mc['Z_mc_bosonkin'].values()#/np.su(resultdict_mc.values())
        ratio =  np.where(hmc_pt>0.,hdata_pt/hmc_pt,1.)
        
        Jraps = resultdict_data['Z_data_bosonkin'].axes[1].edges
        costhetas = resultdict_data['Z_data_bosonkin'].axes[2].edges
        Jpts=resultdict_data['Z_data_bosonkin'].axes[0].edges

        hweights = ROOT.TH3D("rew","rew",len(Jpts)-1,np.array(Jpts),len(Jraps)-1,np.array(Jraps),len(costhetas)-1,np.array(costhetas))
        
        hweights = self.array2hist(ratio, hweights, np.zeros_like(ratio))
        self.weightshelper = ROOT.HelperRew(hweights)
        
        self.d = d
        self.d = self.d.Define("kinweight", self.weightshelper,["Jpsi_pt", "Jpsi_rap", "CStheta"]).Define('calVariables_kinweight', "Eigen::TensorFixedSize<double,Eigen::Sizes<10>> res; auto w = calVariables*kinweight; std::copy(std::begin(w), std::end(w), res.data()); return res;").Define('calVariablesgen_kinweight', "Eigen::TensorFixedSize<double,Eigen::Sizes<10>> res; auto w = calVariablesgen*kinweight; std::copy(std::begin(w), std::end(w), res.data()); return res;")

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
