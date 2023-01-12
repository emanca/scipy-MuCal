from module import *
import numpy as np
import math
from root_numpy import hist2array


class smear(module):
   
    def __init__(self,isData=False):
        self.isData=isData
        pass
        
    def run(self,d):

        self.d=d
        
        etaBins = np.array([-2.4+i*0.1 for i in range(49)])

        with np.load('unbinnedfitglobalitercorscale.npz') as f:
            hdmc = f["xs"][...,-1]

        fd = ROOT.TFile.Open('calibrationDATA.root')

        had = hist2array(fd.Get('a'))
        hbd = hist2array(fd.Get('b'))
        hcd = hist2array(fd.Get('c'))

        fmc = ROOT.TFile.Open('calibrationJMC.root')

        hamc = hist2array(fmc.Get('a'))
        hbmc = hist2array(fmc.Get('b'))
        hcmc = hist2array(fmc.Get('c'))

        with np.load('unbinnedfitglobalitercorscale.npz') as f:
            hdd = f["xs"][...,-1]
            hb = f["xs"][...,5]
        
        if not self.isData:
            @ROOT.Numba.Declare(["float","float","float"], "float")
            def getSmearingFactor(eta, pt, L):
                bineta = np.digitize(np.array([eta]), etaBins)[0]-1
                ad = had[bineta]
                bd = hbd[bineta]
                cd = hcd[bineta]
                dd = hdd[bineta]*hdd[bineta]

                amc = hamc[bineta]
                bmc = hbmc[bineta]
                cmc = hcmc[bineta]
                dmc = hdmc[bineta]*hdmc[bineta]

                L2 = math.pow(L,2)
                pt2 = pt*pt
                resd = (ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2))/pt2
                resmc = (amc + cmc*pt2 + bmc*1./(1+dmc*1./pt2))/pt2

                smearFact=0.
                if resd>resmc:
                    smearFact = np.sqrt(resd-resmc)
                return smearFact
            
            @ROOT.Numba.Declare(["float","float","float"], "float")
            def getDataRes(eta, pt, L):
                bineta = np.digitize(np.array([eta]), etaBins)[0]-1

                ad = hamc[bineta]
                bd = hbmc[bineta]
                cd = hcmc[bineta]
                dd = hdd[bineta]*hdd[bineta]

                L2 = math.pow(L,2)
                pt2 = pt*pt
                resd = (ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2))/pt2
                
                return 0.8*np.sqrt(resd)

        else:
            @ROOT.Numba.Declare(["float","float","float"], "float")
            def getSmearingFactorData(eta, pt, L):
                bineta = np.digitize(np.array([eta]), etaBins)[0]-1
                ad = had[bineta]
                bd = hb[bineta]
                cd = hcd[bineta]
                dd = hdd[bineta]*hdd[bineta]

                amc = hamc[bineta]
                bmc = hb[bineta]
                cmc = hcmc[bineta]
                
                L2 = math.pow(L,2)
                pt2 = pt*pt
                resd = (ad*L + cd*pt2*L2 + bd*1./(1+dd*1./pt2))/pt2
                resmc = (amc*L + cmc*pt2*L2 + bmc*1./(1+dd*1./pt2))/pt2

                smearFact=0.
                if resmc>resd:
                    smearFact = np.sqrt(resmc-resd)
                return smearFact


        NSlots = d.GetNSlots()
        ROOT.gInterpreter.ProcessLine('''
                        std::vector<TRandom3> myRndGens({NSlots});
                        int seed = 1; // not 0 because seed 0 has a special meaning
                        for (auto &&gen : myRndGens) gen.SetSeed(seed++);
                        '''.format(NSlots = NSlots))
        if not self.isData:
            self.d = self.d.Define('Err1', 'Numba::getDataRes(Muplus_eta,Muplusgen_pt,Lplus)')\
                .Define('Err2', 'Numba::getDataRes(Muminus_eta,Muminusgen_pt,Lminus)')\
                .Define('smearedk1', 'float(invplusgen+myRndGens[rdfslot_].Gaus(0., Err1))')\
                .Define('smearedk2', 'float(invminusgen+myRndGens[rdfslot_].Gaus(0., Err2))')\
                .Define('smearedpt1','float(1./smearedk1)')\
                .Define('smearedpt2','float(1./smearedk2)')\
                .Define('v1gensm', 'ROOT::Math::PtEtaPhiMVector(smearedpt1,Muplus_eta,Muplus_phi,0.105658)')\
                .Define('v2gensm', 'ROOT::Math::PtEtaPhiMVector(smearedpt2,Muminus_eta,Muminus_phi,0.105658)')\
                .Define('smearedgenMass', 'float((v1gensm+v2gensm).M())')\
            # .Define('ptplussqsm', 'Numba::computesq(smearedpt1)')\
            # .Define('ptminussqsm', 'Numba::computesq(smearedpt2)')\
            # .Define('invplussm', 'Numba::invpt(smearedpt1)')\
            # .Define('invminussm', 'Numba::invpt(smearedpt2)')\
            # .Define('invplussqsm', 'Numba::invptsqtimesd(Muplus_eta,ptplussqsm)')\
            # .Define('invminussqsm', 'Numba::invptsqtimesd(Muminus_eta,ptminussqsm)')\
            # .Define('calVariablessm','Numba::createRVec(Lplus,Lminus,invplussm,invminussm,smearedpt1,smearedpt2 ptplussqsm,ptminussqsm,invplussqsm,invminussqsm)')
        
        if not self.isData:
            self.d=self.d.Define('smErr1','Numba::getSmearingFactor(Muplus_eta,Muplus_pt,Lplus)')\
            .Define('smErr2','Numba::getSmearingFactor(Muminus_eta,Muminus_pt,Lminus)')
        else:
            self.d=self.d.Define('smErr1','Numba::getSmearingFactorData(Muplus_eta,Muplus_pt,Lplus)')\
            .Define('smErr2','Numba::getSmearingFactorData(Muminus_eta,Muminus_pt,Lminus)')
        self.d=self.d.Define('smearedrecok1','1./pt1corr+myRndGens[rdfslot_].Gaus(0., smErr1)')\
            .Define('smearedrecok2','1./pt2corr+myRndGens[rdfslot_].Gaus(0., smErr2)')\
            .Define('smearedrecopt1','float(1./smearedrecok1)')\
            .Define('smearedrecopt2','float(1./smearedrecok2)')\
            .Define('v1corrsm', 'ROOT::Math::PtEtaPhiMVector(smearedrecopt1,Muplus_eta,Muplus_phi,0.105658)')\
            .Define('v2corrsm', 'ROOT::Math::PtEtaPhiMVector(smearedrecopt2,Muminus_eta,Muminus_phi,0.105658)')\
            .Define('smearedcorrMass', 'float((v1corrsm+v2corrsm).M())')
                
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