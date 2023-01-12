import ROOT
import h5py
import sys
import time
import pickle
from root_numpy import array2hist
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append('RDFprocessor/framework')
ROOT.gSystem.Load('bin/libCalib.so')
import matplotlib.pyplot as plt
import mplhep as hep
# matplotlib stuff
plt.style.use([hep.style.ROOT])
from RDFtree import RDFtree
import numpy as np
import argparse
import itertools
sys.path.append('python')
from defines import defines
from definesZ import definesZ
from smear import smear
from reweightqt import reweightqt
from reweightycostheta import reweightycostheta
from root_numpy import array2hist, hist2array, fill_hist
from fittingFunctionsBinned import computeTrackLength

parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-dataDir', '--dataDir', default='dataPostVFP/', type=str, help='set the directory for input data')
parser.add_argument('-correct', '--correct', default=False, action='store_true', help='apply corrections')
parser.add_argument('-reweight', '--reweight', default=False, action='store_true', help='reweight mc kinematics to data')
parser.add_argument('-pretend', '--pretend', default=False, action='store_true', help='run a fake job on a few events')
parser.add_argument('-layerCorrs', '--layerCorrs', default=False, action='store_true', help='run over layer-corrected kinematics')
parser.add_argument('-ideal', '--ideal', default=False, action='store_true', help='run on ideal mc samples')
parser.add_argument('-plotdistr', '--plotdistr', default=False, action='store_true', help='get a histogram with simple distributions')


args = parser.parse_args()
isJ = args.isJ
dataDir = args.dataDir
correct = args.correct
reweight = args.reweight
pretend = args.pretend
layerCorrs = args.layerCorrs
ideal = args.ideal
plotdistr = args.plotdistr

ROOT.ROOT.EnableImplicitMT(24)

def makeData(p, genMass=False, isData=False, isJ = True):
    dataType = ""
    if isData:
        dataType = "data"
    else:
        if genMass:
            dataType = "genMC"
        else:
            dataType = "MC"
    if isJ:
        cut = 'Muplus_pt>1. && Muminus_pt>1.'
    else:
        cut = 'Muplus_pt>20. && Muminus_pt>20.'
    
    cut+= '&& fabs(Muplus_eta)<2.4 && fabs(Muminus_eta)<2.4' 

    if isJ and not isData:
        cut+= '&& Muplus_muonMedium && Muminus_muonMedium'
    
    cut_gen = "1."
    # if genMass:
    #     cut_gen= 'Muplusgen_pt>3.5 && Muminusgen_pt>3.5'
    
    print("makeData",genMass,isData,isJ)
    print(cut_gen)
    # cutFSR= 'Jpsigen_mass>3.0968'
    cutFSR="1."
    # if not isData: cutFSR= 'Jpsigen_mass>3.0968'

    print(cut)
    print(cutFSR)
    if isData:
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="isGoodLumi==true", filtername="{:20s}".format("good lumi"))
    if not isJ:
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Sum(vetoMuons)==2 && Sum(goodMuons)==2", filtername="{:20s}".format("two muons"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(HLT_IsoMu24 ||  HLT_IsoTkMu24)", filtername="{:20s}".format("Pass HLT"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(Muon_charge[goodMuons][0] + Muon_charge[goodMuons][1]) == 0", filtername="{:20s}".format("Opposite charge"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="std::abs(Muon_eta[goodMuons][0]) < 2.4 && std::abs(Muon_eta[goodMuons][1]) < 2.4", filtername="{:20s}".format("Accept"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muon_mediumId[goodMuons][0] == 1 && Muon_mediumId[goodMuons][1] == 1", filtername="{:20s}".format("MuonId"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muon_pfRelIso04_all[goodMuons][0] < 0.15 && Muon_pfRelIso04_all[goodMuons][1] < 0.15", filtername="{:20s}".format("Isolation"))
    p.EventFilter(nodeToStart='defs', nodeToEnd=dataType, evfilter=cut_gen, filtername="{:20s}".format("cut on gen pt"))
    p.EventFilter(nodeToStart=dataType, nodeToEnd=dataType, evfilter=cut, filtername="{:20s}".format("acceptance cuts"))
    p.EventFilter(nodeToStart=dataType, nodeToEnd=dataType, evfilter=cutFSR, filtername="{:20s}".format("FSR"))

    nEtaBins = 48
    if isJ:
        nEtaBins = 48
    nPtBins = 6
    if plotdistr: nPtBins = 100
    
    nMassBins = 100

    etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
    if isJ:
        # masses = np.linspace(1./3.3/3.3, 1./2.9/2.9, nMassBins+1, dtype='float64')
        # masses = np.linspace(1.06, 1.2, nMassBins+1, dtype='float64')
        masses = np.linspace(2.9, 3.3, nMassBins+1, dtype='float64')
    else:
        masses = np.linspace(75., 115., 100+1, dtype='float64')
    if not isJ:
        # pts = np.array([25.,30., 40., 50., 60.])
        pts = np.array([25.,38, 44, 48.7, 100.])
        if plotdistr: pts = np.linspace(25.,60.,nPtBins+1, dtype='float64')
    else:
        # pts = np.linspace(1.5,25.,nPtBins+1, dtype='float64')
        pts = np.array([2.6, 3.4, 4.4, 5.7, 7.4, 10.2, 13., 18.,25.])
        if plotdistr: pts = np.linspace(2.6,25.,100+1, dtype='float64')
    # pts = np.array([ 5., 6., 6.9, 7.6, 8.4, 9.3, 10.3 ,11.3 ,12.5, 13.9,15.5, 17.8, 23.], dtype='float64')
    ks = np.linspace(0.01, 1.,101)
    Jpts = np.linspace(0,100, 101)
    Jraps = np.linspace(-2.4, 2.4, 120, dtype='float64')
    costhetas = [round(-1. + 2*i/100,2) for i in range(101)]
    
    # if dataType=='data':
    #     p.displayColumn(dataType,['smErr1','smErr2','corrMass', 'smearedcorrMass'])
    if genMass:
        # p.branch(nodeToStart='defs', nodeToEnd='defs', modules=[smear()])
        # p.displayColumn(dataType,['Err1','smearedpt1','Err2','smearedpt2', 'Jpsi_mass', 'smearedMass'])
        if isJ:
            if not plotdistr:
                p.Histogram(columns = ["Muplus_eta","Muminus_eta","smearedpt1", "smearedpt2","smearedgenMass"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi" if not genMass else "Jpsi_gen"), bins = [etas,etas,pts,pts,masses])
                # p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","Jpsigen_mass"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi" if not genMass else "Jpsi_gen"), bins = [etas,etas,pts,pts,masses])

            else:
                pass
        else:
            if not plotdistr:
                # p.Histogram(columns = ["Muplus_eta","Muminus_eta","logratiopt", "cosdeltaphi","Jpsigen_mass", "genweight_abs"], types =   ['float']*6,node=dataType,histoname=ROOT.string("Jpsi" if not genMass else "Jpsi_gen"), bins = [etas,etas,logpts,cosphis, masses])
                # p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","Jpsigen_mass","genweight_abs","jrapweight","jptweight"], types = ['float']*8,node=dataType,histoname=ROOT.string("Jpsi" if not genMass else "Jpsi_gen"), bins = [etas,etas,pts,pts,masses])
                p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","Jpsigen_mass","genweight_abs","jrapweight","jptweight"], types = ['float']*8,node=dataType,histoname=ROOT.string("Jpsi_gen"), bins = [etas,etas,pts,pts,masses])
            else:
                pass
    else:
        if isJ:
            if not plotdistr:
                # p.Histogram(columns = ["Muplus_eta","Muminus_eta","logratiopt", "cosdeltaphi","corrMass"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi"), bins = [etas,etas,logpts,cosphis,masses])
                # p.Histogram(columns = ["Muplus_eta","Muminus_eta","logratiopt", "cosdeltaphi"], types = ['float']*4,node=dataType,histoname=ROOT.string("Jpsi_calVariables"), bins = [etas,etas,logpts,cosphis],sample=("calVariables",8))
                # p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","smearedMass"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi"), bins = [etas,etas,pts,pts,masses])
                p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","corrMass"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi"), bins = [etas,etas,pts,pts,masses])

                p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt"], types = ['float']*4,node=dataType,histoname=ROOT.string("Jpsi_calVariables"), bins = [etas,etas,pts,pts],sample=("calVariables",10))
            else:
                if not isData:
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt"], types = ['float']*4,node=dataType,histoname=ROOT.string("Jpsi_distr_mc"), bins = [etas,etas,pts,pts])
                    # p.Histogram(columns = ["logratiopt", "cosdeltaphi","genweight_abs"], types = ['float']*3,node=dataType,histoname=ROOT.string("Jpsi_distr2_mc"), bins = [logpts,cosphis])
                    # p.Histogram(columns = ["Jpsi_pt", "CStheta", "Jpsi_rap"], types = ['float']*3,node=dataType,histoname=ROOT.string("Jpsi_rew_mc"), bins = [Jpts, costhetas, Jraps])
                else:
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt"], types = ['float']*4,node=dataType,histoname=ROOT.string("Jpsi_distr_data"), bins = [etas,etas,pts,pts])
                    # p.Histogram(columns = ["logratiopt", "cosdeltaphi"], types = ['float']*2,node=dataType,histoname=ROOT.string("Jpsi_distr2_data"), bins = [logpts,cosphis])
                    # p.Histogram(columns = ["Jpsi_pt", "CStheta", "Jpsi_rap"], types = ['float']*3,node=dataType,histoname=ROOT.string("Jpsi_rew_data"), bins = [Jpts, costhetas, Jraps])
        else:
            if not plotdistr:
                if isData:
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","corrMass"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi_data"), bins = [etas,etas,pts,pts,masses])
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt"], types = ['float']*4,node=dataType,histoname=ROOT.string("Jpsi_calVariables_data"), bins = [etas,etas,pts,pts],sample=("calVariables",10))
                else:
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","corrMass","genweight_abs","jrapweight","jptweight"], types = ['float']*8,node=dataType,histoname=ROOT.string("Jpsi_mc"), bins = [etas,etas,pts,pts,masses])
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt", "genweight_abs"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi_calVariables_mc"), bins= [etas,etas,pts,pts],sample=("calVariables",10))
            else:
                if not isData:
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt","genweight_abs"], types = ['float']*5,node=dataType,histoname=ROOT.string("Jpsi_distr_mc"), bins = [etas,etas,pts,pts])
                    p.Histogram(columns = ["Jpsi_pt", "CStheta", "Jpsi_rap","genweight_abs","jrapweight","jptweight"], types = ['float']*6,node=dataType,histoname=ROOT.string("Jpsi_rew_mc"), bins = [Jpts, costhetas, Jraps])
                else:
                    p.Histogram(columns = ["Muplus_eta","Muminus_eta","Muplus_pt", "Muminus_pt"], types = ['float']*4,node=dataType,histoname=ROOT.string("Jpsi_distr_data"), bins = [etas,etas,pts,pts])
                    # p.Histogram(columns = ["logratiopt", "cosdeltaphi"], types = ['float']*2,node=dataType,histoname=ROOT.string("Jpsi_distr2_data"), bins = [logpts,cosphis])
                    p.Histogram(columns = ["Jpsi_pt", "CStheta", "Jpsi_rap"], types = ['float']*3,node=dataType,histoname=ROOT.string("Jpsi_rew_data"), bins = [Jpts, costhetas, Jraps])
        pass

    return p

def makepkg(histo, histotermsplus, histotermsminus, etas, pts, masses, good_idx, histoterms=True):

    edges = [etas,etas,pts,pts,masses]
    histo = histo[good_idx]

    if histoterms:
        #compute mean in each bin (integrating over mass) for pt-dependent terms
        histoden = np.sum(histo, axis=-1)
        binCenters = []
        means = []
        for histoterm in histotermsplus:
            print(histoterm.shape, histoterm[good_idx].shape)
            ret = histoterm[good_idx]/histoden
            means.append(ret)
        mean = np.stack(means,axis=-1)
        binCenters.append(mean)
        means = []
        for histoterm in histotermsminus:
            ret = histoterm[good_idx]/histoden
            means.append(ret)
        mean = np.stack(means,axis=-1)
        binCenters.append(mean)
        
    pkg = {}
    pkg['dataset'] = histo
    pkg['edges'] = edges
    if histoterms:
        pkg['binCenters1'] = binCenters[0]
        pkg['binCenters2'] = binCenters[1]
    
    pkg['good_idx'] = good_idx
    
    return pkg

if isJ:
    # inputFileMC = '%s/MuonGunUL2016_v96_RecJpsiPhotosTail0_quality_noconstraint/globalcor.root' % dataDir
    # inputFileMC = '%s/MuonGunUL2016_v123_RecJpsiPhotos_idealquality_zeromaterial_noconstraint/globalcor.root' % dataDir
    inputFileMC = '%s/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29/jpsisnapshot.root' % dataDir
    # inputFileD =['%s/MuonGunUL2016_v202a_RecDataJPsiH_quality_constraintfsr28/jpsisnapshot.root' % dataDir,'%s/MuonGunUL2016_v202_RecDataJPsiG_quality_constraintfsr28/jpsisnapshot.root' % dataDir,'%s/MuonGunUL2016_v202_RecDataJPsiFpost_quality_constraintfsr28/jpsisnapshot.root' % dataDir]
    # inputFileMC = '%s/MuonGunUL2016_v207a_RecJpsiPhotos_quality_constraintfsr29_cor/correctedTracksjpsi.root'% dataDir
    if ideal:
        inputFileMC = '%s/jpsisnapshot.root'% dataDir
    inputFileD = '%s/MuonGunUL2016_v206_RecDataJPsiFpostGH_quality_constraintfsr28/jpsisnapshot.root'% dataDir

else:
    # inputFileMC ='%s/MuonGunUL2016_v208_RecZMuMu_quality_nobs_cor/correctedTracksjpsi.root' % dataDir
    inputFileMC ='%s/ZNanoMC/*.root' % dataDir
    # inputFileD ='%s/MuonGunUL2016_v207_RecDataZMuMuFpostGH_quality_cor/correctedTracksjpsi.root' % dataDir
    inputFileD = ['%s/ZNanoData/*.root'% dataDir,'%s/ZNanoDataH/*.root'% dataDir]
outputDir = "JPsiInputData"
bMap = ROOT.TFile.Open("bFieldMap.root")

flag = ""
if plotdistr: flag="_mukin"

treeName = 'Events'
if isJ: treeName = 'tree'

p = RDFtree(outputDir = outputDir, inputFile = inputFileMC, treeName=treeName, outputFile="JPsiMC{}.root".format(flag), pretend=pretend)
calibMC = ROOT.TFile.Open("calibrationJMC_aftersm.root")
corr_flag = False
if layerCorrs: corr_flag = True
if correct:
    modules=[defines(), ROOT.applyCalibration(bMap,calibMC,False, True,corr_flag),smear()]
else:
    modules=[defines(), ROOT.applyCalibration(bMap,calibMC,False, False,corr_flag), smear(),ROOT.CSvariableProducer()]
if not isJ: 
    modules=[definesZ(),ROOT.genLeptonSelector(),ROOT.genVProducer(),defines(),ROOT.CSvariableProducer(),reweightycostheta(isJ=isJ),reweightqt(isJ=isJ)]
    if correct:
        modules.extend([ROOT.applyCalibration(bMap,calibMC,False, True,corr_flag),smear()])
    else:
        modules.extend([ROOT.applyCalibration(bMap,calibMC,False, False,corr_flag),smear()])

p.branch(nodeToStart='input', nodeToEnd='defs', modules=modules)
dataMC = makeData(p, isData=False, isJ=isJ)
if not plotdistr: dataGen = makeData(p, genMass=True, isData=False,isJ=isJ)

p2 = RDFtree(outputDir = outputDir, inputFile = inputFileD, treeName=treeName, outputFile="JPsiData{}.root".format(flag), pretend=pretend)
calibD = ROOT.TFile.Open("calibrationJDATA_aftersm.root")
datajson = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
lumi = "luminosityBlock"
if isJ:
    lumi = "lumi"

if correct: 
    modules=[ROOT.isGoodLumi(datajson,lumi), defines(),ROOT.applyCalibration(bMap,calibD,True, True, corr_flag)]
else:
    modules=[ROOT.isGoodLumi(datajson,lumi), defines(),ROOT.applyCalibration(bMap,calibD,True, False, corr_flag), smear(isData=True),ROOT.CSvariableProducer()]
if not isJ: 
    modules=[ROOT.isGoodLumi(datajson,lumi),definesZ(isData=True),defines(isData=True), ROOT.CSvariableProducer()]
    if correct:
        modules.extend([ROOT.applyCalibration(bMap,calibD,True, True, corr_flag),smear(isData=True)])
    else:
        modules.extend([ROOT.applyCalibration(bMap,calibD,True, False, corr_flag),smear(isData=True)])

p2.branch(nodeToStart='input', nodeToEnd='defs', modules=modules)
dataD = makeData(p2, isData=True, isJ=isJ)

objList = []
if not plotdistr: objList.extend(dataGen.getObjects()['genMC'])
objList.extend(dataMC.getObjects()['MC'])
objList.extend(dataD.getObjects()['data'])

start = time.time()
if not plotdistr: rGen = dataGen.getCutFlowReport('genMC')
rMC = dataMC.getCutFlowReport('MC')
rD = dataD.getCutFlowReport('data')

ROOT.RDF.RunGraphs(objList)

p.gethdf5Output()
p2.gethdf5Output()

if not plotdistr: rGen.Print()
print("------------")
rMC.Print()
print("------------")
rD.Print()

print('all samples processed in {} s'.format(time.time()-start))
if plotdistr:
    assert(0)

nEtaBins = 48
nPtBins = 6
nMassBins = 100

etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')

if isJ:
    masses = np.linspace(2.9, 3.3, nMassBins+1, dtype='float64')
    # masses = np.linspace(1./3.3/3.3, 1./2.9/2.9, nMassBins+1, dtype='float64')
    masses_gen=masses
else:
    masses = np.linspace(75., 115., 101, dtype='float64')

pts = np.linspace(20.,100.,nPtBins+1, dtype='float64')
# pts = np.array([ 5., 6., 6.9, 7.6, 8.4, 9.3, 10.3 ,11.3 ,12.5, 13.9,15.5, 17.8, 23.], dtype='float64')

fMC = h5py.File('JPsiInputData/JPsiMC.hdf5', mode='r+')
fdata = h5py.File('JPsiInputData/JPsiData.hdf5', mode='r+')
hMCgen = np.array(fMC['Jpsi_gen'][:])
if not isJ:
    hMC = np.array(fMC['Jpsi_mc'][:])
    hD = np.array(fdata['Jpsi_data'][:])
else:
    hMC = np.array(fMC['Jpsi'][:])
    hD = np.array(fdata['Jpsi'][:])

# histos for bin centers
if isJ:
    histotermsMC = np.array(fMC['Jpsi_calVariables'][:])
    hMCLplus = histotermsMC[...,0]
    hMCLminus = histotermsMC[...,1]
    hMCsplus = histotermsMC[...,2]
    hMCsminus = histotermsMC[...,3]
    hMCptplus = histotermsMC[...,4]
    hMCptminus = histotermsMC[...,5]
    hMCptplussq = histotermsMC[...,6]
    hMCptminussq = histotermsMC[...,7]
    hMCinvptplussq = histotermsMC[...,8]
    hMCinvptminussq = histotermsMC[...,9]
    histotermsMCplus = [hMCptplus, hMCsplus, hMCptplussq, hMCLplus, hMCinvptplussq]
    histotermsMCminus = [hMCptminus, hMCsminus, hMCptminussq, hMCLminus, hMCinvptminussq]

    histotermsD = np.array(fdata['Jpsi_calVariables'][:])
    hDLplus = histotermsD[...,0]
    hDLminus = histotermsD[...,1]
    hDsplus = histotermsD[...,2]
    hDsminus = histotermsD[...,3]
    hDptplus = histotermsD[...,4]
    hDptminus = histotermsD[...,5]
    hDptplussq = histotermsD[...,6]
    hDptminussq = histotermsD[...,7]
    hDinvptplussq = histotermsD[...,8]
    hDinvptminussq = histotermsD[...,9]
    histotermsDplus = [hDptplus, hDsplus, hDptplussq, hDLplus, hDinvptplussq]
    histotermsDminus = [hDptminus, hDsminus, hDptminussq, hDLminus, hDinvptminussq]
else: #something dummy which isn't used
    # histotermsMC = np.array(fMC['Jpsi_mc'][:])
    # histotermsD = np.array(fdata['Jpsi_data'][:])

    # histotermsMCplus = [histotermsMC, histotermsMC, histotermsMC, histotermsMC]
    # histotermsMCminus = [histotermsMC, histotermsMC, histotermsMC, histotermsMC]
    # histotermsDplus = histotermsMCplus
    # histotermsDminus = histotermsMCplus
    histotermsMC = np.array(fMC['Jpsi_calVariables_mc'][:])
    hMCLplus = histotermsMC[...,0]
    hMCLminus = histotermsMC[...,1]
    hMCsplus = histotermsMC[...,2]
    hMCsminus = histotermsMC[...,3]
    hMCptplus = histotermsMC[...,4]
    hMCptminus = histotermsMC[...,5]
    hMCptplussq = histotermsMC[...,6]
    hMCptminussq = histotermsMC[...,7]
    hMCinvptplussq = histotermsMC[...,8]
    hMCinvptminussq = histotermsMC[...,9]
    histotermsMCplus = [hMCptplus, hMCsplus, hMCptplussq, hMCLplus, hMCinvptplussq]
    histotermsMCminus = [hMCptminus, hMCsminus, hMCptminussq, hMCLminus, hMCinvptminussq]

    histotermsD = np.array(fdata['Jpsi_calVariables_data'][:])
    hDLplus = histotermsD[...,0]
    hDLminus = histotermsD[...,1]
    hDsplus = histotermsD[...,2]
    hDsminus = histotermsD[...,3]
    hDptplus = histotermsD[...,4]
    hDptminus = histotermsD[...,5]
    hDptplussq = histotermsD[...,6]
    hDptminussq = histotermsD[...,7]
    hDinvptplussq = histotermsD[...,8]
    hDinvptminussq = histotermsD[...,9]
    histotermsDplus = [hDptplus, hDsplus, hDptplussq, hDLplus, hDinvptplussq]
    histotermsDminus = [hDptminus, hDsminus, hDptminussq, hDLminus, hDinvptminussq]


pklfileBase = 'calInput{}'.format('J' if isJ else 'Z')
pklfileData = pklfileBase + 'DATA'
pklfileMC = pklfileBase + 'MC'

pklfileData+='_{}etaBins_{}ptBins'.format(len(etas)-1, len(pts)-1)
if correct:
    pklfileData+='_corr'
if layerCorrs:
    pklfileData+='_layerCorr'
pklfileData+='.pkl'
# pklfileData+='_noBmap.pkl'

pklfileMC+='_{}etaBins_{}ptBins'.format(len(etas)-1, len(pts)-1)
if correct:
    pklfileMC+='_corr'
if layerCorrs:
    pklfileMC+='_layerCorr'
if ideal:
    pklfileMC+='_ideal'
pklfileMC+='.pkl'

if not isJ:
    pklfileGen = 'calInputZMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(pts)-1)
    filehandler = open('calInputZMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(pts)-1), 'wb')
else:
    pklfileGen = 'calInputJMCgen_{}etaBins_{}ptBins_smeared.pkl'.format(len(etas)-1, len(pts)-1)
    filehandler = open('calInputJMCgen_{}etaBins_{}ptBins_smeared.pkl'.format(len(etas)-1, len(pts)-1), 'wb')

print(pklfileData, pklfileMC)

if isJ:
    good_idx_MC = np.nonzero(np.sum(hMC,axis=-1)>500.)
    good_idx_D = np.nonzero(np.sum(hD,axis=-1)>500.)
    print("good_idx MC size", good_idx_MC[0].shape)
    print("good_idx Data size", good_idx_D[0].shape)
    print('number of effective events', np.sum(hMC[good_idx_MC]), np.sum(hD[good_idx_D]))
else:
    good_idx_MC = np.nonzero(np.sum(hMC,axis=-1)>750.)
    good_idx_D = np.nonzero(np.sum(hD,axis=-1)>750.)
    print("good_idx MC size", good_idx_MC[0].shape)
    print("good_idx Data size", good_idx_D[0].shape)
    print('number of effective events', np.sum(hMC[good_idx_MC]), np.sum(hD[good_idx_D]))

    
# # remove eta bins with not enough pt-span
# full = np.zeros((nEtaBins,nEtaBins,20,20),dtype='float64')
# full[good_idx] = np.sum(hMCgen,axis=-1)[good_idx]
# good_pt_alongeta = np.count_nonzero(full, axis=(2,3),keepdims=True) # shape (nEtaBins,nEtaBins,1,1)
# mask = np.where(good_pt_alongeta>5, 1, 0)
# good_idx = np.nonzero((np.sum(hMCgen,axis=-1)>3000.)&(mask))
# print("good_idx:", good_idx)

with open(pklfileGen, 'wb') as filehandler:
    pickle.dump(hMCgen, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
hMCgen = None

pkgMC = makepkg(hMC, histotermsMCplus, histotermsMCminus, etas, pts, masses, good_idx_MC, True)
dataMC = None
with open(pklfileMC, 'wb') as filehandler:
    pickle.dump(pkgMC, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
pkgMC = None
pkgD = makepkg(hD, histotermsDplus, histotermsDminus,etas, pts, masses, good_idx_D,True)
dataD = None
with open(pklfileData, 'wb') as filehandler:
    pickle.dump(pkgD, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
pkgD = None
