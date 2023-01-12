from module import *
import numpy as np
import math

class calibrate(module):
   
    def __init__(self, calibFile):
        
        self.myTH1 = []
        self.myTH2 = []
        self.myTH3 = []
        self.myTHN = []

        self.myTH1Group = []
        self.myTH2Group = []
        self.myTH3Group = []
        self.myTHNGroup = []

        self.calibFile = calibFile
        self.fcalib = ROOT.TFile.Open(self.calibFile)
        self.A = self.fcalib.Get("A")
        self.e = self.fcalib.Get("e")
        self.M = self.fcalib.Get("M")

    def run(self,d):

        self.d = d.Define("charge1", "1").Define("charge2", "-1")\
                    .Define("pt1corr", 'ROOT.correct(Muplus_pt, Muplus_eta,charge1, self.A, self.e,self.M)')\
                    .Define("pt2corr", 'ROOT.correct(Muminus_pt, Muminus_eta,charge2, self.A, self.e,self.M)')\
                    .Define("v1corr", "ROOT::Math::PtEtaPhiMVector(pt1corr,Muplus_eta,Muplus_phi,0.105658)")\
                    .Define("v2corr", "ROOT::Math::PtEtaPhiMVector(pt2corr,Muminus_eta, Muminus_phi,0.105658)")\
                    .Define("corrMass", '(v1corr+v2corr).M()')

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