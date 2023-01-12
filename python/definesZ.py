from module import *
import numpy as np
import math

class definesZ(module):
   
    def __init__(self, isData=False):
        self.isData=isData
        pass
    def run(self,d):
        self.d = d.Define('Muplus_eta','Muon_cvhbsEta[GoodMuons][Muon_cvhbsCharge>0][0]')\
                    .Define('Muplus_pt','Muon_cvhbsPt[GoodMuons][Muon_cvhbsCharge>0][0]')\
                    .Define('Muplus_phi','Muon_cvhbsPhi[GoodMuons][Muon_cvhbsCharge>0][0]')\
                    .Define('Muminus_eta','Muon_cvhbsEta[GoodMuons][Muon_cvhbsCharge<0][0]')\
                    .Define('Muminus_pt','Muon_cvhbsPt[GoodMuons][Muon_cvhbsCharge<0][0]')\
                    .Define('Muminus_phi','Muon_cvhbsPhi[GoodMuons][Muon_cvhbsCharge<0][0]')\
                    .Define('Muplusgen_pt','float(1./(GoodMuons_kgen[GoodMuons_genCharge>0][0]))')\
                    .Define('Muminusgen_pt','float(1./(GoodMuons_kgen[GoodMuons_genCharge<0][0]))')\
                    .Define('Muplusgen_eta','float((GoodMuons_genEta[GoodMuons_genCharge>0][0]))')\
                    .Define('Muminusgen_eta','float((GoodMuons_genEta[GoodMuons_genCharge<0][0]))')\
                    .Define("resplus", "float(((1./Muplusgen_pt)-(1./Muplus_pt))/(1./Muplusgen_pt))")\
                    .Define("resminus", "float(((1./Muminusgen_pt)-(1./Muminus_pt))/(1./Muminusgen_pt))")\
                    .Define('v1', 'ROOT::Math::PtEtaPhiMVector(Muplus_pt,Muplus_eta,Muplus_phi,0.105658)')\
                    .Define('v2', 'ROOT::Math::PtEtaPhiMVector(Muminus_pt,Muminus_eta,Muminus_phi,0.105658)')\
                    .Define('Jpsi_pt', 'float((v1+v2).Pt())')\
                    .Define('Jpsi_rap', 'float((v1+v2).Rapidity())')\
                    .Define('Jpsi_mass','float((v1+v2).M())')\
                    .Define('v1gen', 'ROOT::Math::PtEtaPhiMVector(Muplusgen_pt,Muplus_eta,Muplus_phi,0.105658)')\
                    .Define('v2gen', 'ROOT::Math::PtEtaPhiMVector(Muminusgen_pt,Muminus_eta,Muminus_phi,0.105658)')\
                    .Define('Jpsigen_mass','float((v1gen+v2gen).M())')
        if not self.isData:
            self.d=self.d.Define("genweight_abs", "genWeight/std::abs(genWeight)")
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
