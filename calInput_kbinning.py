import ROOT
import h5py
import sys
import time
import pickle
sys.path.append('RDFprocessor/framework')
ROOT.gSystem.Load('RDFprocessor/framework/lib/libRDFProcessorCore.so')
from RDFtree import RDFtree
import numpy as np
import argparse
import itertools
sys.path.append('python')
from defines import defines
from root_numpy import array2hist, hist2array, fill_hist
from fittingFunctionsBinned import computeTrackLength

parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-dataDir', '--dataDir', default='/scratchnvme/emanca/scipy-MuCal/dataUL/', type=str, help='set the directory for input data')

args = parser.parse_args()
isJ = args.isJ
dataDir = args.dataDir

ROOT.ROOT.EnableImplicitMT(12)

def makeData(p, genMass=False, isData=False):

    dataType = ""
    if isData:
        dataType = "data"
    else:
        if genMass:
            dataType = "genMC"
        else:
            dataType = "MC"

    if isJ:
        cut = 'Muplus_pt>3. && Muminus_pt>3. && Muplus_pt<25. && Muminus_pt<25.'
    else:
        cut = 'Muplus_pt>20.0 && Muminus_pt>20.0 && Muplus_pt<100. && Muminus_pt<100.'
    
    cut+= '&& fabs(Muplus_eta)<2.4 && fabs(Muminus_eta)<2.4' 
    
    if not isData:
        cut+= '&& Muplusgen_pt>5.5 && Muminusgen_pt>5.5 && Muplusgen_pt<100. && Muminusgen_pt<100. && Jpsi_pt<29.'

    print(cut)
    p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=cut, filtername="{:20s}".format("acceptance cuts"))

    nEtaBins = 48
    nPtBins = 12
    nMassBins = 100
    nkbins = 6

    etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
    if isJ:
        masses = np.linspace(2.9, 3.3, nMassBins+1, dtype='float64')
    else:
        masses = np.linspace(75., 115., nMassBins+1, dtype='float64')

    pts = np.linspace(5.,23.,nPtBins+1, dtype='float64')
    # pts = np.array([ 5., 6., 6.9, 7.6, 8.4, 9.3, 10.3 ,11.3 ,12.5, 13.9,15.5, 17.8, 23.], dtype='float64')
    ks = np.linspace(1./23., 1./5.5, nkbins+1, dtype=np.float64)
    
    if genMass:
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","Jpsigen_mass"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsi" if not genMass else "Jpsi_gen"), bins = [etas,etas,ks,ks,masses])
    else:
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","Jpsi_mass"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsi".format(dataType)), bins = [etas,etas,ks,ks,masses])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","Lplus"], types = ['float']*5,node='defs',histoname=ROOT.string("JpsiLplus".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","Lminus"], types = ['float']*5,node='defs',histoname=ROOT.string("JpsiLminus".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","invplus"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsisplus".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","invminus"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsisminus".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","Muplus_pt"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsiptplus".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","Muminus_pt"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsiptminus".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","ptplussq"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsiptplussq".format(dataType)), bins = [etas,etas,ks,ks])
        p.Histogram(columns = ["Muplus_eta","Muminus_eta","invplus","invminus","ptminussq"], types = ['float']*5,node='defs',histoname=ROOT.string("Jpsiptminussq".format(dataType)), bins = [etas,etas,ks,ks])

    return p

def makepkg(histo, histotermsplus, histotermsminus, etas, pts, masses, good_idx):

    edges = [etas,etas,pts,pts,masses]
    histo = histo[good_idx]

    #compute mean in each bin (integrating over mass) for pt-dependent terms
    histoden = np.sum(histo, axis=-1)
    
    binCenters = []
    means = []
    for histoterm in histotermsplus:
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
    pkg['binCenters1'] = binCenters[0]
    pkg['binCenters2'] = binCenters[1]
    pkg['good_idx'] = good_idx
    
    return pkg

if isJ:
    inputFileMC ='%s/MuonGunUL2016_v45_RecJpsiPhotos_quality_noconstraint_v3/210411_195303/0000/*.root' % dataDir
    inputFileD ='%s/MuonGunUL2016_v45_RecDataJPsiH_quality_noconstraint/210411_190954/0000/*.root' % dataDir
else:
    inputFileMC ='%s/ZJToMuMu_mWPilot.root' % dataDir
    #inputFileMC ='%s/muonTreeZMC.root' % dataDir
    inputFileD ='%s/muonTreeDataZ.root' % dataDir

outputDir = "JPsiInputData"
p = RDFtree(outputDir = outputDir, inputFile = inputFileMC, treeName="tree", outputFile="JPsiMC.root", pretend=False)
p.branch(nodeToStart='input', nodeToEnd='defs', modules=[defines()])
dataGen = makeData(p, genMass=True, isData=False)
dataMC = makeData(p, isData=False)

p2 = RDFtree(outputDir = outputDir, inputFile = inputFileD, treeName="tree", outputFile="JPsiData.root", pretend=False)
p2.branch(nodeToStart='input', nodeToEnd='defs', modules=[defines()])
dataD = makeData(p2, isData=True)

objList = []
objList.extend(dataGen.getObjects()['defs'])
objList.extend(dataMC.getObjects()['defs'])
objList.extend(dataD.getObjects()['defs'])

start = time.time()
ROOT.RDF.RunGraphs(objList)

p.gethdf5Output()
p2.gethdf5Output()


print('all samples processed in {} s'.format(time.time()-start))

nEtaBins = 48
nPtBins = 12
nMassBins = 100
nkbins = 6

etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
if isJ:
    masses = np.linspace(2.9, 3.3, nMassBins+1, dtype='float64')
else:
    masses = np.linspace(75., 115., nMassBins+1, dtype='float64')

pts = np.linspace(5.,23.,nPtBins+1, dtype='float64')
# pts = np.array([ 5., 6., 6.9, 7.6, 8.4, 9.3, 10.3 ,11.3 ,12.5, 13.9,15.5, 17.8, 23.], dtype='float64')
ks = np.linspace(1./23., 1./5.5, nkbins+1, dtype=np.float64)


fMC = h5py.File('JPsiInputData/JPsiMC.hdf5', mode='r+')
fdata = h5py.File('JPsiInputData/JPsiData.hdf5', mode='r+')
hMCgen = np.array(fMC['Jpsi_gen'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1, len(masses)-1),order='F'),order='C')
hMC = np.array(fMC['Jpsi'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1, len(masses)-1),order='F'),order='C')
hD = np.array(fdata['Jpsi'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1, len(masses)-1),order='F'),order='C')

# # retrieve quantiles
# hpt = np.sum(hMC,axis=(0,1,3,4))
# hrpt = ROOT.TH1D("pt", "pt", nPtBins, pts)
# hrpt = array2hist(hpt, hrpt)
# ptquantiles = np.linspace(0.,1.,nPtBins+1, dtype='float64')
# y=0.
# q=np.zeros([nPtBins+1])
# y=hrpt.GetQuantiles(nPtBins+1,q,ptquantiles)
# # pts = np.quantile(hpt,ptquantiles)
# print(q,y)
# assert(0)

# histos for bin centers
hMCLminus = np.array(fMC['JpsiLminus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCLplus = np.array(fMC['JpsiLplus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCsminus = np.array(fMC['Jpsisminus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCsplus = np.array(fMC['Jpsisplus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCptminus = np.array(fMC['Jpsiptminus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCptplus = np.array(fMC['Jpsiptplus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCptminussq = np.array(fMC['Jpsiptminussq'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hMCptplussq = np.array(fMC['Jpsiptplussq'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
histotermsMCplus = [hMCptplus, hMCsplus, hMCptplussq, hMCLplus]
histotermsMCminus = [hMCptminus, hMCsminus, hMCptminussq, hMCLminus]


hDLminus = np.array(fdata['JpsiLminus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDLplus = np.array(fdata['JpsiLplus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDsminus = np.array(fdata['Jpsisminus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDsplus = np.array(fdata['Jpsisplus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDptminus = np.array(fdata['Jpsiptminus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDptplus = np.array(fdata['Jpsiptplus'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDptminussq = np.array(fdata['Jpsiptminussq'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
hDptplussq = np.array(fdata['Jpsiptplussq'][:].reshape((len(etas)-1,len(etas)-1, len(ks)-1, len(ks)-1),order='F'),order='C')
histotermsDplus = [hDptplus, hDsplus, hDptplussq, hDLplus]
histotermsDminus = [hDptminus, hDsminus, hDptminussq, hDLminus]

pklfileBase = 'calInput{}'.format('J' if isJ else 'Z')
pklfileData = pklfileBase + 'DATA'
pklfileMC = pklfileBase + 'MC'

pklfileData+='_{}etaBins_{}kBins'.format(len(etas)-1, len(ks)-1)
pklfileData+='.pkl'

pklfileMC+='_{}etaBins_{}kBins'.format(len(etas)-1, len(ks)-1)
pklfileMC+='.pkl'

if not isJ:
    pklfileGen = 'calInputZMCgen_{}etaBins_{}kBins.pkl'.format(len(etas)-1, len(ks)-1)
    filehandler = open('calInputZMCgen_{}etaBins_{}kBins.pkl'.format(len(etas)-1, len(ks)-1), 'wb')
else:
    pklfileGen = 'calInputJMCgen_{}etaBins_{}kBins.pkl'.format(len(etas)-1, len(ks)-1)
    filehandler = open('calInputJMCgen_{}etaBins_{}kBins.pkl'.format(len(etas)-1, len(ks)-1), 'wb')

print(pklfileData, pklfileMC)

good_idx = np.nonzero(np.sum(hMCgen,axis=-1)>2000.)
print("good_idx size", good_idx[0].shape)
# remove eta bins with not enough pt-span
full = np.zeros((nEtaBins,nEtaBins,nPtBins,nPtBins),dtype='float64')
full[good_idx] = np.sum(hMCgen,axis=-1)[good_idx]
print(np.count_nonzero(full, axis=(2,3)))
good_pt_alongeta = np.count_nonzero(full, axis=(2,3),keepdims=True) # shape (nEtaBins,nEtaBins,1,1)
mask = np.where(good_pt_alongeta>20, 1, 0)
good_idx = np.nonzero((np.sum(hMCgen,axis=-1)>1000.)&(mask))
print("good_idx size", good_idx[0].shape)
hMCgen = hMCgen[good_idx]

with open(pklfileGen, 'wb') as filehandler:
    pickle.dump(hMCgen, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
hMCgen = None

pkgMC = makepkg(hMC, histotermsMCplus, histotermsMCminus, etas, ks, masses, good_idx)

dataMC = None
with open(pklfileMC, 'wb') as filehandler:
    pickle.dump(pkgMC, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
pkgMC = None

pkgD = makepkg(hD, histotermsDplus, histotermsDminus,etas, ks, masses, good_idx)
dataD = None
with open(pklfileData, 'wb') as filehandler:
    pickle.dump(pkgD, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
pkgD = None
