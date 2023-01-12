from module import *
import numpy as np
import math
import h5py


class smear(module):
   
    def __init__(self,isData=False):
        self.isData=isData
        pass
        
    def run(self,d):

        self.d=d
        
        etaBins = np.array([-2.4+i*0.2 for i in range(25)])

        fd = ROOT.TFile.Open('outClosureTruth_tgr.root')

        had = np.asarray(fd.Get('a'))[1:-1]
        hbd = np.asarray(fd.Get('b'))[1:-1]
        hcd = np.asarray(fd.Get('c'))[1:-1]
        hdd = np.asarray(fd.Get('d'))[1:-1]

        # fmc = ROOT.TFile.Open('/scratchnvme/emanca/scipy-MuCal/calibrationJMC.root')

        # hamc = hist2array(fmc.Get('a'))
        # hbmc = hist2array(fmc.Get('b'))
        # hcmc = hist2array(fmc.Get('c'))

        # with np.load('/scratchnvme/emanca/scipy-MuCal/unbinnedfitglobalitercorscale.npz') as f:
        #     hdd = f["xs"][...,-1]
        #     hb = f["xs"][...,5]
        
        if not self.isData:

            # @ROOT.Numba.Declare(["bool","float"], "float")
            # def getTriggerWeight(bit, lumiratio):
            #     if bit==True:
            #         return lumiratio
            #     else:
            #         return 1.

            # @ROOT.Numba.Declare(["float","float","float"], "float")
            # def getSmearingFactor(eta, pt, L):
            #     bineta = np.digitize(np.array([eta]), etaBins)[0]-1
            #     ad = had[bineta]
            #     bd = hbd[bineta]
            #     cd = hcd[bineta]
            #     dd = hdd[bineta]*hdd[bineta]

            #     amc = hamc[bineta]
            #     bmc = hbmc[bineta]
            #     cmc = hcmc[bineta]
            #     dmc = hdmc[bineta]*hdmc[bineta]

            #     L2 = math.pow(L,2)
            #     pt2 = pt*pt
            #     resd = (ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2))/pt2
            #     resmc = (amc*L + cmc*pt2*L2 + bmc*L*1./(1+dmc*1./pt2))/pt2

            #     smearFact=0.
            #     if resd>resmc:
            #         smearFact = np.sqrt(resd-resmc)
            #     return smearFact
            
            @ROOT.Numba.Declare(["float","float","float"], "float")
            def getDataRes(eta, pt, L):
                bineta = np.digitize(np.array([eta]), etaBins)[0]-1
                # bineta=0
                ad = had[bineta]
                bd = hbd[bineta]
                cd = hcd[bineta]
                dd = hdd[bineta]

                L2 = math.pow(L,2)
                pt2 = pt*pt
                resd = (ad*L + cd*pt2*L2 + bd*L*1./(1.+dd/pt2/L))/pt2
                # resd = (ad + cd*pt2 + bd*1./(1+dd*1./pt2))/pt2
                # resd = (ad*4.348)/pt2
                return np.sqrt(resd)
                # return 0.01/pt
            
            # etas1 =  np.array([0.2,0.4])
            # etas2 =  np.array([-2.0,-1.8])
            # pts1 = np.array([8.,25.])
            # pts2 = np.array([25.,38])
            # pts1full = np.linspace(8.,25.,9)
            # pts2full = np.linspace(25.,38,6)
            # etaBins = np.array([-2.4+i*0.2 for i in range(25)])
            # fweights = h5py.File('/scratchnvme/emanca/scipy-MuCal/weights.hdf5', mode='r')
            # weights = fweights["weights"][...]
            # print(weights)
            # @ROOT.Numba.Declare(['float','float','float','float'], 'float')
            # def ptweights(eta1,eta2,pt1,pt2):
            #     binpt1 = np.digitize(np.array([pt1]), pts1full)[0]-1
            #     binpt2 = np.digitize(np.array([pt2]), pts2full)[0]-1
            #     bineta1 = np.digitize(np.array([eta1]), etaBins)[0]-1
            #     bineta2 = np.digitize(np.array([eta2]), etaBins)[0]-1
            #     if pt1<pts1full[0] or pt1>pts1full[-1] or  pt2<pts2full[0] or pt2>pts2full[-1]:
            #         return 1.
            #     else:
            #         return weights[0,0,binpt1,binpt2]

        else:
            pass
            # @ROOT.Numba.Declare(["float","float","float"], "float")
            # def getSmearingFactorData(eta, pt, L):
            #     bineta = np.digitize(np.array([eta]), etaBins)[0]-1
            #     ad = had[bineta]
            #     bd = hb[bineta]
            #     cd = hcd[bineta]
            #     dd = hdd[bineta]*hdd[bineta]

            #     amc = hamc[bineta]
            #     bmc = hb[bineta]
            #     cmc = hcmc[bineta]
                
            #     L2 = math.pow(L,2)
            #     pt2 = pt*pt
            #     resd = (ad*L + cd*pt2*L2 + bd*1./(1+dd*1./pt2))/pt2
            #     resmc = (amc*L + cmc*pt2*L2 + bmc*1./(1+dd*1./pt2))/pt2

            #     smearFact=0.
            #     if resmc>resd:
            #         smearFact = np.sqrt(resmc-resd)
            #     return smearFact


        NSlots = d.GetNSlots()
        ROOT.gInterpreter.ProcessLine('''
                        std::vector<TRandom3> myRndGens({NSlots});
                        int seed = 1; // not 0 because seed 0 has a special meaning
                        for (auto &&gen : myRndGens) gen.SetSeed(seed++);
                        '''.format(NSlots = NSlots))
        self.d = self.d.Define('Err1', 'Numba::getDataRes(Muplus_eta,Muplus_pt,Lplus)')\
            .Define('Err2', 'Numba::getDataRes(Muminus_eta,Muminus_pt,Lminus)')\
            .Define('smearedk1', 'float(invplusgen+myRndGens[rdfslot_].Gaus(0.,Err1))')\
            .Define('smearedk2', 'float(invminusgen+myRndGens[rdfslot_].Gaus(0.,Err2))')\
            .Define('smearedpt1','float(1./smearedk1)')\
            .Define('smearedpt2','float(1./smearedk2)')\
            .Define('v1gensm', 'ROOT::Math::PtEtaPhiMVector(smearedpt1,Muplusgen_eta,Muplusgen_phi,0.105658)')\
            .Define('v2gensm', 'ROOT::Math::PtEtaPhiMVector(smearedpt2,Muminusgen_eta,Muminusgen_phi,0.105658)')\
            .Define('smearedgenMass', 'float((v1gensm+v2gensm).M())')\
            .Define('massRes', 'float(TMath::Sqrt(TMath::Power(Err1/invplusgen,2)+TMath::Power(Err2/invminusgen,2))*Jpsigen_mass/2)')\
            .Define('smearedk1red', 'float(invplusgen+myRndGens[rdfslot_].Gaus(0., 0.99*Err1))')\
            .Define('smearedk2red', 'float(invminusgen+myRndGens[rdfslot_].Gaus(0., 0.99*Err2))')\
            .Define('smearedpt1red','float(1./smearedk1red)')\
            .Define('smearedpt2red','float(1./smearedk2red)')\
            .Define('kdiffs_plus', 'float(invplus-invplusgen)')\
            .Define('kdiffs_minus', 'float(invminus-invminusgen)')\
            .Define('v1gensmred', 'ROOT::Math::PtEtaPhiMVector(smearedpt1red,Muplusgen_eta,Muplusgen_phi,0.105658)')\
            .Define('v2gensmred', 'ROOT::Math::PtEtaPhiMVector(smearedpt2red,Muminusgen_eta,Muminusgen_phi,0.105658)')\
            .Define('smearedgenMassred', 'float((v1gensmred+v2gensmred).M())')\
                
                # .Define('weights', 'Numba::ptweights(Muplus_eta,Muminus_eta,Muplusgen_pt,Muminusgen_pt)')
                # .Define('ptplussqsm', 'Numba::computesq(smearedpt1red)')\
                # .Define('ptminussqsm', 'Numba::computesq(smearedpt2red)')\
                # .Define('pull', 'float((1./smearedpt1 - 1./Muplusgen_pt)/(Err1))')
                # .Define('resplus', 'float((1./Muplus_pt - 1./Muplusgen_pt)/(1./Muplusgen_pt))')\
                # .Define('resminus', 'float((1./Muminus_pt - 1./Muminusgen_pt)/(1./Muminusgen_pt))')\
                # .Define("weight", "float(HLT_Dimuon20_Jpsi*0.1+HLT_DoubleMu4_JpsiTrk_Displaced*0.75+HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing*0.005+HLT_Mu7p5_Track2_Jpsi*0.005+HLT_Mu7p5_Track3p5_Jpsi*0.005)")\
                # .Define('smearedgenMassNorm',"float((smearedgenMass-3.09692)/massRes)")\
                # .Define('smearedgenMassSq','float(1./smearedgenMass/smearedgenMass)')\
                # .Define('genMassSq','float(1./Jpsigen_mass/Jpsigen_mass)')\
                # .Define('smearedk1', 'float(invplusgen+myRndGens[rdfslot_].Gaus(0., Err1))')\
                # .Define('smearedk2', 'float(invminusgen+myRndGens[rdfslot_].Gaus(0., Err2))')\
                # .Define('smearedgenMass', 'float(Jpsigen_mass+myRndGens[rdfslot_].Gaus(0., 0.05))')\
                # .Define('massRes', 'float(myRndGens[rdfslot_].Gaus(0.05, 0.002))')\
                
        self.d = self.d.Define('ptplussqsm', 'Numba::computesq(smearedpt1)')\
            .Define('ptminussqsm', 'Numba::computesq(smearedpt2)')\
            .Define('invplussm', 'Numba::invpt(smearedpt1)')\
            .Define('invminussm', 'Numba::invpt(smearedpt2)')\
            .Define('invplussqsm', 'Numba::invptsqtimesd(Muplus_eta,ptplussqsm,Lplus)')\
            .Define('invminussqsm', 'Numba::invptsqtimesd(Muminus_eta,ptminussqsm,Lminus)')\
            .Define('calVariablessm','Numba::createRVec(Lplus,Lminus,invplussm,invminussm,smearedpt1,smearedpt2, ptplussqsm,ptminussqsm,invplussqsm,invminussqsm)')
        
        # if not self.isData:
        #     self.d=self.d.Define('smErr1','Numba::getSmearingFactor(Muplus_eta,Muplus_pt,Lplus)')\
        #     .Define('smErr2','Numba::getSmearingFactor(Muminus_eta,Muminus_pt,Lminus)')
        # else:
        #     self.d=self.d.Define('smErr1','Numba::getSmearingFactorData(Muplus_eta,Muplus_pt,Lplus)')\
        #     .Define('smErr2','Numba::getSmearingFactorData(Muminus_eta,Muminus_pt,Lminus)')
        # self.d=self.d.Define('smearedrecok1','1./pt1corr+myRndGens[rdfslot_].Gaus(0., smErr1)')\
        #     .Define('smearedrecok2','1./pt2corr+myRndGens[rdfslot_].Gaus(0., smErr2)')\
        #     .Define('smearedrecopt1','float(1./smearedrecok1)')\
        #     .Define('smearedrecopt2','float(1./smearedrecok2)')\
        #     .Define('v1corrsm', 'ROOT::Math::PtEtaPhiMVector(smearedrecopt1,Muplus_eta,Muplus_phi,0.105658)')\
        #     .Define('v2corrsm', 'ROOT::Math::PtEtaPhiMVector(smearedrecopt2,Muminus_eta,Muminus_phi,0.105658)')\
        #     .Define('smearedcorrMass', 'float((v1corrsm+v2corrsm).M())')
                
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
