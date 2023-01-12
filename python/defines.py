from module import *
import numpy as np
import math
import h5py

# # with np.load('/scratchnvme/emanca/scipy-MuCal/unbinnedfitglobalitercorscale_single.npz') as f:
# with np.load('/scratchnvme/emanca/scipy-MuCal/unbinnedfitglobalitercorscale_zbs.npz') as f:
#     d = f["xs"][...,-1]
#     b = f["xs"][...,5]

# fd = ROOT.TFile.Open('/scratchnvme/emanca/scipy-MuCal/outClosureTruth.root')

# hd = hist2array(fd.Get('d'))

etas = np.linspace(-2.4,2.4,24)

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

@ROOT.Numba.Declare(['float','float','float'], 'float')
def invptsqtimesd(eta,pt2,L):
    bineta = np.digitize(np.array([eta]), etas)[0]-1
    # return 1./(1.+hd[bineta]/pt2/L)
    # return hd[bineta]
    return 1./pt2


@ROOT.Numba.Declare(['float'], 'float')
def computesq(x):
    return x**2

@ROOT.Numba.Declare(['float','float'], 'float')
def logratio(pt1,pt2):
    return np.log(pt1/pt2)

#@ROOT.Numba.Declare(['float','float'], 'float')
#def deltaphi(phi1,phi2):
#    result = phi1 - phi2
#    if result>math.pi:
#        result -= float(2 * math.pi)
#    if result<=math.pi:
#        result += float(2 * math.pi)
#    return result

@ROOT.Numba.Declare(['float','float','float','float','float','float','float','float','float','float'], 'RVec<float>')
def createRVec(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
    return np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

ROOT.gInterpreter.Declare("""
float getLeadMuonPt(float muplus_pt,float muminus_pt) {
    if(muplus_pt>muminus_pt) return muplus_pt;
    else return muminus_pt;
};
float getSubLeadMuonPt(float muplus_pt,float muminus_pt) {
    if(muplus_pt<muminus_pt) return muplus_pt;
    else return muminus_pt;
};
float deltaphi(float phi0, float phi1) {
    float dphi = phi1 - phi0;
    if (dphi > M_PI) {
        dphi -= 2.0*M_PI;
    }
    else if (dphi <= - M_PI) {
      dphi += 2.0*M_PI;
    }
    return dphi;
}
""")

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
            .Define('invplussq', 'Numba::invptsqtimesd(Muplus_eta,ptplussq,Lplus)')\
            .Define('invminussq', 'Numba::invptsqtimesd(Muminus_eta,ptminussq,Lminus)')\
            .Define('calVariables','Numba::createRVec(Lplus,Lminus,invplus,invminus,Muplus_pt,Muminus_pt,ptplussq,ptminussq,invplussq,invminussq)')\
            .Define('calVariables_weight', "Eigen::TensorFixedSize<double, Eigen::Sizes<10>> res; auto w = calVariables; std::copy(std::begin(w), std::end(w), res.data()); return res;")\
            .Define('logratiopt', 'Numba::logratio(Muplus_pt,Muminus_pt)')\
            .Define('Mulead_pt','getLeadMuonPt(Muplus_pt,Muminus_pt)')\
            .Define('Musublead_pt','getSubLeadMuonPt(Muplus_pt,Muminus_pt)')\
            .Define('resplusphi', 'deltaphi(Muplus_phi,Muplusgen_phi)')\
            .Define('resminusphi', 'deltaphi(Muminus_phi,Muminusgen_phi)')\
            .Define('modmass','TMath::Sqrt(Muplus_pt*Muminus_pt*TMath::Sin(deltaphi(Muplus_phi,Muminus_phi)))')\
            .Define("v1", "ROOT::Math::PtEtaPhiMVector(Muplus_pt,Muplus_eta,Muplus_phi, 0.105658)")\
            .Define("v2", "ROOT::Math::PtEtaPhiMVector(Muminus_pt,Muminus_eta,Muminus_phi, 0.105658)")\
            .Define("sintheta1","TMath::Sin(v1.Theta())")\
            .Define("sintheta2","TMath::Sin(v2.Theta())")\
            .Define("p1","Muplus_pt/sintheta1")\
            .Define("p2","Muminus_pt/sintheta2")\
            .Define("hybridpt1","p1*2./(TMath::Exp(Muplusgen_eta)+TMath::Exp(-Muplusgen_eta))")\
            .Define("hybridpt2","p2*2./(TMath::Exp(Muminusgen_eta)+TMath::Exp(-Muminusgen_eta))")\
            .Define('resplustheta', '2*TMath::ATan((TMath::Exp(-Muplusgen_eta)))-2*TMath::ATan((TMath::Exp(-Muplus_eta)))')\
            .Define('resminustheta','2*TMath::ATan((TMath::Exp(-Muminusgen_eta)))-2*TMath::ATan((TMath::Exp(-Muminus_eta)))')
            
        if not self.isData:
            self.d = self.d.Define('resplus', 'float((1./Muplus_pt - 1./Muplusgen_pt)/(1./Muplusgen_pt))')\
                .Define('resminus', 'float((1./Muminus_pt - 1./Muminusgen_pt)/(1./Muminusgen_pt))')\
            # .Define('calVariables_tgrweight', "Eigen::TensorFixedSize<double, Eigen::Sizes<10>> res; auto w = calVariables*tgrweight; std::copy(std::begin(w), std::end(w), res.data()); return res;")\
            # .Define('interpw','Numba::ptweights(Muplus_pt,Muminus_pt)')
        if not self.isData:
            self.d=self.d.Define('Lplusgen','Numba::computeTrackLength(Muplusgen_eta)')\
            .Define('Lminusgen','Numba::computeTrackLength(Muminusgen_eta)')\
            .Define('ptplussqgen', 'Numba::computesq(Muplusgen_pt)')\
            .Define('ptminussqgen', 'Numba::computesq(Muminusgen_pt)')\
            .Define('invplusgen', 'Numba::invpt(Muplusgen_pt)')\
            .Define('invminusgen', 'Numba::invpt(Muminusgen_pt)')\
            .Define('invplussqgen', 'Numba::invptsqtimesd(Muplus_eta,ptplussqgen,Lplus)')\
            .Define('invminussqgen', 'Numba::invptsqtimesd(Muminus_eta,ptminussqgen,Lminus)')\
            .Define('calVariablesgen','Numba::createRVec(Lplus,Lminus,invplus,invminus,Muplusgen_pt,Muminusgen_pt,ptplussqgen,ptminussqgen,invplussqgen,invminussqgen)')\
            .Define('calVariablesgen_weight', "Eigen::TensorFixedSize<double, Eigen::Sizes<10>> res; auto w = calVariablesgen; std::copy(std::begin(w), std::end(w), res.data()); return res;")\
            # .Define("weight", "float((6019754/241116)*(HLT_Dimuon20_Jpsi*0.1+HLT_DoubleMu4_JpsiTrk_Displaced*0.75+HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing*0.005+HLT_Mu7p5_Track2_Jpsi*0.005+HLT_Mu7p5_Track3p5_Jpsi*0.005))")\
            # .Define('v1', 'ROOT::Math::PtEtaPhiMVector(Muplus_pt,Muplus_eta,Muplus_phi,0.105658)')\
            # .Define('v2', 'ROOT::Math::PtEtaPhiMVector(Muminus_pt,Muminus_eta,Muminus_phi,0.105658)')\
            # .Define('Jpsi_rap', 'float((v1+v2).Rapidity())')\
            # .Define('angle','float(TMath::Exp(Muplusgen_eta-Muminusgen_eta)+TMath::Exp(-1*(Muplusgen_eta-Muminusgen_eta))-2*cosdeltaphi)')
            #
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
