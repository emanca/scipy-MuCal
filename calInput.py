import ROOT
import pickle
import numpy as np
from header import CastToRNode
from root_numpy import array2hist, hist2array, fill_hist
import argparse
import itertools
from scipy.stats import binned_statistic_dd
from fittingFunctionsBinned import computeTrackLength

ROOT.gROOT.ProcessLine(".L src/module.cpp+")
ROOT.gROOT.ProcessLine(".L src/applyCalibration.cpp+")


parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-smearedMC', '--smearedMC', default=False, action='store_true', help='Use smeared gen mass in MC, omit for using reco mass')
parser.add_argument('-mcTruth', '--mcTruth', default=False, action='store_true', help='Use to prepare sample for MCtruth fit')
parser.add_argument('-runClosure', '--runClosure', default=False, action='store_true', help='Use to apply full calibration. If omit, rescale data for B map and leave MC as it is')
parser.add_argument('-dataDir', '--dataDir', default='/scratchssd/emanca/wproperties-analysis/muonCalibration/inputs', type=str, help='set the directory for input data')

args = parser.parse_args()
isJ = args.isJ
smearedMC = args.smearedMC
mcTruth = args.mcTruth
runClosure = args.runClosure
dataDir = args.dataDir

ROOT.ROOT.EnableImplicitMT()
RDF = ROOT.ROOT.RDataFrame

def makeData(inputFile, genMass=False, smearedMass=False, mcTruth=False, isData=False):

    if isJ:
        if mcTruth:
            cut = 'mcpt1>4.3 && mcpt2>4.3 && mcpt1<25. && mcpt2<25.'
        else:
            cut = 'pt1>4.3 && pt2>4.3 && pt1<25. && pt2<25.'
    else:
        if mcTruth:
            cut = 'mcpt1>20.0 && mcpt2>20.0 && mcpt1<100. && mcpt2<100.'
        else:
            cut = 'pt1>20.0 && pt2>20.0 && pt1<100. && pt2<100.'
    
    if restrictToBarrel:
        cut+= '&& fabs(eta1)<0.8 && fabs(eta2)<0.8'
    else:
        cut+= '&& fabs(eta1)<2.4 && fabs(eta2)<2.4' 
    if not isData:
        cut+= '&& mcpt1>0. && mcpt2>0.'

    d = RDF('tree',inputFile)

    if smearedMass:
        NSlots = d.GetNSlots()
        ROOT.gInterpreter.ProcessLine('''
                        std::vector<TRandom3> myRndGens({NSlots});
                        int seed = 1; // not 0 because seed 0 has a special meaning
                        for (auto &&gen : myRndGens) gen.SetSeed(seed++);
                        '''.format(NSlots = NSlots))

    print(cut)
    
    d = d.Filter(cut)\
        .Define('v1', 'ROOT::Math::PtEtaPhiMVector(pt1,eta1,phi1,0.105658)')\
        .Define('v2', 'ROOT::Math::PtEtaPhiMVector(pt2,eta2,phi2,0.105658)')\

    if smearedMass:
        d = d.Define('v1sm', 'ROOT::Math::PtEtaPhiMVector(mcpt1+myRndGens[rdfslot_].Gaus(0., cErr1*mcpt1),eta1,phi1,0.105658)')\
            .Define('v2sm', 'ROOT::Math::PtEtaPhiMVector(mcpt2+myRndGens[rdfslot_].Gaus(0., cErr2*mcpt2),eta2,phi2,0.105658)')\
            .Define('smearedgenMass', '(v1sm+v2sm).M()')

    if mcTruth:
        d = d.Define('res1','mcpt1/pt1')\
            .Define('res2','mcpt2/pt2')\

    f = ROOT.TFile.Open('%s/bFieldMap.root' % dataDir)
    bFieldMap = f.Get('bfieldMap')

    if runClosure: print('taking corrections from', '{}/scale_{}_80X_13TeV.root'.format(dataDir, 'DATA' if isData else 'MC'))

    f2 = ROOT.TFile.Open('{}/scale_{}_80X_13TeV.root'.format(dataDir, 'DATA' if isData else 'MC'))
    A = f2.Get('magnetic')
    e = f2.Get('e')
    M = f2.Get('B')

    module = ROOT.applyCalibration(bFieldMap, A, e, M, isData, runClosure)

    d = module.run(CastToRNode(d))
    
    mass = 'corrMass'

    if smearedMass:
        mass = 'smearedgenMass'
    
    cols=[mass,'eta1', 'pt1', 'phi1', 'eta2', 'pt2', 'phi2']
    
    if genMass:
        cols.append('genMass')

    if mcTruth:
        cols.append('mcpt1')
        cols.append('mcpt2')
        cols.append('res1')
        cols.append('res2')
    
    data = d.AsNumpy(columns=cols)

    return data


def makeGenDataset(data, etas, pts, masses):

    datasetGen = np.array([data['eta1'],data['eta2'],data['pt1'],data['pt2'],data['genMass']])
    histoGen, edges = np.histogramdd(datasetGen.T, bins = [etas,etas,pts,pts,masses])

    return histoGen

def makeMCTruthDataset(data, etas, pts, masses):

    #add charge for mc truth events
    charge1 = np.ones_like(data['eta1'])
    charge2 = -np.ones_like(data['eta1'])

    datasetMCTruth = np.array([np.concatenate((data['eta1'],data['eta2']),axis=None),np.concatenate((charge1*data['mcpt1'],charge2*data['mcpt2']),axis=None),np.concatenate((data['res1'], data['res2']),axis=None)])
    histoMCTruth, edges = np.histogramdd(datasetMCTruth.T, bins = [etas,pts,masses])

    good_idx = np.nonzero(np.sum(histoMCTruth,axis=-1)>100.)
    histoMCTruth = histoMCTruth[good_idx]

    #compute mean in each bin (integrating over mass) for pt-dependent terms
    massesfull = np.array([masses[0],masses[-1]])
    histoden = np.histogramdd(datasetMCTruth.T, bins = [etas,pts,massesfull])[0][good_idx]

    k = 1./datasetMCTruth[1]
    eta = datasetMCTruth[0]
    sEta = np.sin(2*np.arctan(np.exp(-eta)))
    L = computeTrackLength(eta)

    fIn = ROOT.TFile.Open("calibrationUnbinned.root")
    gHisto = fIn.Get("g")
    dHisto = fIn.Get("d")
    g1Histo = fIn.Get("g_1")
    g2Histo = fIn.Get("g_2")

    gterm = hist2array(gHisto)
    dterm = hist2array(dHisto)
    g1term = hist2array(g1Histo)
    g2term = hist2array(g2Histo)

    terms_etas = np.linspace(-2.4, 2.4, gterm.shape[0]+1, dtype='float64')
    findBin = np.digitize(eta, terms_etas)-1
    g = gterm[findBin]
    d = dterm[findBin]
    g1 = g1term[findBin]
    g2 = g2term[findBin]

    terms = []
    terms.append(1./k) #misalignement
    terms.append(np.abs(sEta*k)) #energy loss
    terms.append(1./np.square(k)) #hit position error
    terms.append(np.square(L)) #squared track length
    terms.append((1.+np.square(g)*np.square(k))*np.reciprocal(1.+np.square(d)*np.square(k))) # correlations resolution
    terms.append((1.+np.square(g1)*np.square(k))*np.reciprocal(1.+np.square(g2)*np.square(k))) # correlations scale
    
    means = []
    for term in terms:
        histoterm = np.histogramdd(datasetMCTruth.T, bins = [etas,pts,massesfull], weights=term)[0][good_idx]
        ret = histoterm/histoden
        #remove spurious mass dimension
        ret = np.squeeze(ret, axis=-1)
        
        means.append(ret.flatten())
            
    mean = np.stack(means,axis=-1)
    
    pkg = {}
    pkg['dataset'] = histoMCTruth
    pkg['edges'] = edges
    pkg['binCenters'] = mean
    pkg['good_idx'] = good_idx

    return pkg

def makepkg(data, etas, pts, masses, good_idx, smearedMass=False):

    mass = 'corrMass'

    if smearedMass:
        mass = 'smearedgenMass'

    dataset = np.array([data['eta1'],data['eta2'],data['pt1'],data['pt2'],data[mass]])
    
    histo, edges = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,masses])
    histo = histo[good_idx]

    #compute mean in each bin (integrating over mass) for pt-dependent terms
    massesfull = np.array([masses[0],masses[-1]])
    histoden = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,massesfull])[0][good_idx]

    fIn = ROOT.TFile.Open("resolutionMCtruth.root")
    bHisto = fIn.Get("b")
    dHisto = fIn.Get("d")

    bterm = hist2array(bHisto)
    dterm = hist2array(dHisto)

    terms_etas = np.linspace(-2.4, 2.4, bterm.shape[0]+1, dtype='float64')
    
    binCenters = []
    for ipt in range(2):
        pt = dataset[ipt+2]
        eta = dataset[ipt]
        L = computeTrackLength(eta)
        
        sEta = np.sin(2*np.arctan(np.exp(-eta)))
        findBin = np.digitize(eta, terms_etas)-1
        b = bterm[findBin]
        d = dterm[findBin]
        
        terms = []
        terms.append(pt)
        terms.append(sEta/pt)
        terms.append(np.square(pt))
        terms.append(np.square(L))
        terms.append(b*np.square(L)*np.reciprocal(1.+d/np.square(pt)/np.square(L)))

        means = []
        for term in terms:
            histoterm = np.histogramdd(dataset.T, bins = [etas,etas,pts,pts,massesfull], weights=term)[0][good_idx]
            ret = histoterm/histoden
            #remove spurious mass dimension
            ret = np.squeeze(ret, axis=-1)
            
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

restrictToBarrel = False

if isJ:
    inputFileMC ='%s/JpsiToMuMu_JpsiPt8_pythia8_2.root' % dataDir
    inputFileD ='%s/muonTreeData.root' % dataDir
else:
    inputFileMC ='%s/ZJToMuMu_mWPilot.root' % dataDir
    #inputFileMC ='%s/muonTreeZMC.root' % dataDir
    inputFileD ='%s/muonTreeDataZ.root' % dataDir


nEtaBins = 48
nPtBins = 5
nMassBins = 100

if restrictToBarrel:
    etas = np.linspace(-0.8, 0.8, nEtaBins+1, dtype='float64')
else:
    etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')

if isJ:
    masses = np.linspace(2.9, 3.3, nMassBins+1, dtype='float64')
else:
    masses = np.linspace(75., 115., nMassBins+1, dtype='float64')

ptquantiles = np.linspace(0.,1.,nPtBins+1, dtype='float64')


pklfileBase = 'calInput{}'.format('J' if isJ else 'Z')
pklfileData = pklfileBase + 'DATA'

if smearedMC:
    pklfileMC = pklfileBase + 'MCsmear'
else:
    pklfileMC = pklfileBase + 'MC'

pklfileData+='_{}etaBins_{}ptBins'.format(len(etas)-1, len(ptquantiles)-1)
pklfileData+='.pkl'

pklfileMC+='_{}etaBins_{}ptBins'.format(len(etas)-1, len(ptquantiles)-1)
pklfileMC+='.pkl'

if not isJ:
    pklfileGen = 'calInputZMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(ptquantiles)-1)
    filehandler = open('calInputZMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(ptquantiles)-1), 'wb')
    pklfileMCtruth = 'calInputZMCtruth_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, 80)
    filehandler = open('calInputZMCtruth_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, 80), 'wb')
else:
    pklfileGen = 'calInputJMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(ptquantiles)-1)
    filehandler = open('calInputJMCgen_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, len(ptquantiles)-1), 'wb')
    pklfileMCtruth = 'calInputJMCtruth_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, 20)
    filehandler = open('calInputJMCtruth_{}etaBins_{}ptBins.pkl'.format(len(etas)-1, 20), 'wb')

print(pklfileData, pklfileMC)
#print(pkgD['dataset'][0][0] - pkgMC['dataset'][0])

dataMC = makeData(inputFileMC, genMass=True, smearedMass=smearedMC, mcTruth = mcTruth)

print(ptquantiles)
if mcTruth:
    if isJ:
        pts = np.linspace(4.3,25, 20+1, dtype='float64')
    else:
        pts = np.linspace(20.,100., 80+1, dtype='float64')

    ptsNeg = np.flip(-1*pts)
    pts = np.concatenate((ptsNeg,pts), axis=None)
else:
    pts = np.quantile(np.concatenate((dataMC['pt1'],dataMC['pt2']),axis=0),ptquantiles)
print(pts)

if mcTruth:

    res = np.linspace(0.9, 1.1, nMassBins+1, dtype='float64')
    pkgTruth = makeMCTruthDataset(dataMC,etas,pts,res)

    with open(pklfileMCtruth, 'wb') as filehandler:
        pickle.dump(pkgTruth, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
    pkgTruth = None

if mcTruth: assert(0)

### end mcTruth stuff
histoGen = makeGenDataset(dataMC,etas,pts,masses)

good_idx = np.nonzero(np.sum(histoGen,axis=-1)>1000.)
print("good_idx size", good_idx[0].shape)

histoGen = histoGen[good_idx]

with open(pklfileGen, 'wb') as filehandler:
    pickle.dump(histoGen, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
histoGen = None

pkgMC = makepkg(dataMC, etas, pts, masses, good_idx, smearedMass=smearedMC)

dataMC = None
with open(pklfileMC, 'wb') as filehandler:
    pickle.dump(pkgMC, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
pkgMC = None

dataD = makeData(inputFileD, isData=True)
pkgD = makepkg(dataD, etas, pts, masses, good_idx)
dataD = None
with open(pklfileData, 'wb') as filehandler:
    pickle.dump(pkgD, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
pkgD = None
