import ROOT
import h5py
import sys
import time
import pickle
import lz4.frame
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append('RDFprocessor/framework')
import csv
import pathlib
import narf
import hist
ROOT.gSystem.Load('bin/libCalib.so')
ROOT.gInterpreter.ProcessLine(".O3")
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
from reweightycostheta import reweightycostheta
from fittingFunctionsBinned import computeTrackLength
from trigger_rew import trigger_rew


parser = argparse.ArgumentParser('')
parser.add_argument('-isJ', '--isJ', default=False, action='store_true', help='Use to run on JPsi, omit to run on Z')
parser.add_argument('-dataDir', '--dataDir', default='/scratchnvme/emanca/scipy-MuCal/dataPostVFP/', type=str, help='set the directory for input data')
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

ROOT.ROOT.EnableImplicitMT()

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
        cut = 'Muplus_pt>0 && Muminus_pt>0'
    else:
        cut = 'Muplus_pt>0. && Muminus_pt>0.'

    cut+= '&& fabs(Muplus_eta)<2.4 && fabs(Muminus_eta)<2.4'

    # if isJ and not isData:
    #     cut+= '&& Muplus_muonMedium && Muminus_muonMedium'

    cut_gen = "1."
    # if genMass:
    #     cut_gen= 'Muplusgen_pt>3.5 && Muminusgen_pt>3.5'

    print("makeData",genMass,isData,isJ)
    print(cut_gen)
    # cutFSR= 'Jpsigen_mass>3.0968'
    cutFSR="1."
    # if not isData: cutFSR= 'Jpsigen_mass>3.0968'

    cutTrigger = '1.'
    # if isJ:
    #     cutTrigger = 'HLT_Dimuon20_Jpsi||HLT_DoubleMu4_JpsiTrk_Displaced||HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing||HLT_Mu7p5_Track2_Jpsi||HLT_Mu7p5_Track3p5_Jpsi||HLT_Dimuon0_Jpsi_Muon||HLT_Dimuon0er16_Jpsi_NoVertexing||HLT_Dimuon10_Jpsi_Barrel||HLT_Dimuon16_Jpsi||HLT_DoubleMu4_3_Jpsi_Displaced||HLT_Mu7p5_Track7_Jpsi'

    print(cut)
    print(cutFSR)

    cutJPsi='Jpsi_pt>8.'
    if isData:
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="isGoodLumi==true", filtername="{:20s}".format("good lumi"))
    if not isJ:
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" Muplus_muonMedium && Muminus_muonMedium", filtername="{:20s}".format("Accept"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplus_muonIsGlobal && Muminus_muonIsGlobal", filtername="{:20s}".format("muon global"))
        p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplusgen_pt >10. && Muminusgen_pt >10.", filtername="{:20s}".format("mc matching"))
        if not dataType == "genMC":
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" corrMass>75. && corrMass<110.", filtername="{:20s}".format("Accept"))
        else:
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" Jpsigen_mass>75. && Jpsigen_mass<110.", filtername="{:20s}".format("Accept"))

        # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Sum(GoodMuons)==2", filtername="{:20s}".format("two muons"))
        # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(HLT_IsoMu24 ||  HLT_IsoTkMu24)", filtername="{:20s}".format("Pass HLT"))
        # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(Muon_charge[GoodMuons][0] + Muon_charge[GoodMuons][1]) == 0", filtername="{:20s}".format("Opposite charge"))
        # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="std::abs(Muon_eta[goodMuons][0]) < 2.4 && std::abs(Muon_eta[goodMuons][1]) < 2.4", filtername="{:20s}".format("Accept"))
        # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muon_mediumId[goodMuons][0] == 1 && Muon_mediumId[goodMuons][1] == 1", filtername="{:20s}".format("MuonId"))
        # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muon_pfRelIso04_all[goodMuons][0] < 0.15 && Muon_pfRelIso04_all[goodMuons][1] < 0.15", filtername="{:20s}".format("Isolation"))
    else:
        if dataType == "genMC":
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" smearedgenMass>2.8&& smearedgenMass<3.31", filtername="{:20s}".format("Accept"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muplus_eta)<1.2 && smearedpt1 >4.2) || (fabs(Muplus_eta)>1.2 && smearedpt1 > (-1.8*fabs(Muplus_eta)+6.3))", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muminus_eta)<1.2 && smearedpt2 >4.2) || (fabs(Muminus_eta)>1.2 && smearedpt2 > (-1.8*fabs(Muminus_eta)+6.3))", filtername="{:20s}".format("global muon boundaries mu minus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muplus_eta)<0.4 && smearedpt1 >5.2) || (fabs(Muplus_eta)>0.4 && smearedpt1 > 3.)", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muminus_eta)<0.4 && smearedpt2 >5.2) || (fabs(Muminus_eta)>0.4 && smearedpt2 > (-2.*fabs(Muminus_eta)+5.9))", filtername="{:20s}".format("global muon boundaries mu minus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" smearedgenMass>2.9&& smearedgenMass<3.31", filtername="{:20s}".format("Accept"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muplus_eta)<0.4 && Muplusgen_pt >5.2) || (fabs(Muplus_eta)>0.4 && Muplusgen_pt > (-2.*fabs(Muplus_eta)+5.9))", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muminus_eta)<0.4 && Muminusgen_pt >5.2) || (fabs(Muminus_eta)>0.4 && Muminusgen_pt > (-2.*fabs(Muminus_eta)+5.9))", filtername="{:20s}".format("global muon boundaries mu minus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" (fabs(Muplus_eta)<2. && smearedpt1 >4.2) || (fabs(Muplus_eta)>2. && smearedpt1 >2.)", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" (fabs(Muminus_eta)<2. && smearedpt2 >4.2) || (fabs(Muminus_eta)>2. && smearedpt2 >2.)", filtername="{:20s}".format("global muon boundaries mu minus"))
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="smearedpt1 >4.", filtername="{:20s}".format("global muon boundaries mu plus"))
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="smearedpt2 >4.", filtername="{:20s}".format("global muon boundaries mu minus"))
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplusgen_pt >1. && Muminusgen_pt >1", filtername="{:20s}".format("mc matching"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplus_muonLoose && Muminus_muonLoose", filtername="{:20s}".format("muon id"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplus_muonIsGlobal && Muminus_muonIsGlobal", filtername="{:20s}".format("muon global"))
            pass

        else:
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" corrMass>2.8 && corrMass<3.31", filtername="{:20s}".format("Accept"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muplus_eta)<1.2 && pt1corr >4.2) || (fabs(Muplus_eta)>1.2 && pt1corr > (-1.8*fabs(Muplus_eta)+6.3))", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muminus_eta)<1.2 && pt2corr >4.2) || (fabs(Muminus_eta)>1.2 && pt2corr > (-1.8*fabs(Muminus_eta)+6.3))", filtername="{:20s}".format("global muon boundaries mu minus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muplus_eta)<1.2 && pt1corr >4.) || (fabs(Muplus_eta)>1.2 && pt1corr >3.)", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="(fabs(Muminus_eta)<1.2 && pt2corr >4.) || (fabs(Muminus_eta)>1.2 && pt2corr > 3.)", filtername="{:20s}".format("global muon boundaries mu minus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" (fabs(Muplus_eta)<2. && pt1corr >4.2) || (fabs(Muplus_eta)>2. && pt1corr >2.)", filtername="{:20s}".format("global muon boundaries mu plus"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter=" (fabs(Muminus_eta)<2. && pt2corr >4.2) || (fabs(Muminus_eta)>2. && pt2corr >2.)", filtername="{:20s}".format("global muon boundaries mu minus"))
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="pt1corr >4.", filtername="{:20s}".format("global muon boundaries mu plus"))
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="pt2corr >4.", filtername="{:20s}".format("global muon boundaries mu minus"))
            p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplusgen_pt >1. && Muminusgen_pt >1", filtername="{:20s}".format("mc matching"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplus_muonLoose && Muminus_muonLoose", filtername="{:20s}".format("muon id"))
            # p.EventFilter(nodeToStart='defs', nodeToEnd='defs', evfilter="Muplus_muonIsGlobal && Muminus_muonIsGlobal", filtername="{:20s}".format("muon global"))
            pass
    p.EventFilter(nodeToStart='defs', nodeToEnd=dataType, evfilter=cut_gen, filtername="{:20s}".format("cut on gen pt"))
    p.EventFilter(nodeToStart=dataType, nodeToEnd=dataType, evfilter=cut, filtername="{:20s}".format("acceptance cuts"))
    p.EventFilter(nodeToStart=dataType, nodeToEnd=dataType, evfilter=cutFSR, filtername="{:20s}".format("FSR"))
    p.EventFilter(nodeToStart=dataType, nodeToEnd=dataType, evfilter=cutTrigger, filtername="{:20s}".format("trigger"))
    # p.EventFilter(nodeToStart=dataType, nodeToEnd=dataType, evfilter=cutJPsi, filtername="{:20s}".format("cut JPsi"))

    nEtaBins = 24
    axis_eta1 = hist.axis.Regular(nEtaBins, -2.4, 2.4, name = "eta1")
    axis_eta2 = hist.axis.Regular(nEtaBins, -2.4, 2.4, name = "eta2")
    if isJ:
        axis_mass =  hist.axis.Regular(100, 2.8069,3.3069, name= "mass")
    else:
        axis_mass =  hist.axis.Regular(1000, 75., 110., name= "mass")

    if not isJ:
        axis_pt1 = hist.axis.Variable([30.,35., 39., 43, 48., 70.], name = "pt1")
        axis_pt2 = hist.axis.Variable([30.,35., 39., 43, 48., 70.], name = "pt2")
        axis_phi1 = hist.axis.Variable([-3.14159265, -2.35619449, -1.57079633, -0.78539816,  0., 0.78539816,  1.57079633,  2.35619449,  3.14159265], name= "phi1")
        axis_phi2 = hist.axis.Variable([-3.14159265, -2.35619449, -1.57079633, -0.78539816,  0., 0.78539816,  1.57079633,  2.35619449,  3.14159265], name= "phi2")


    else:
        axis_pt1 = hist.axis.Variable([ 2.,4.,5.2,6.3,8.7,25.], name = "pt1")
        axis_pt2 = hist.axis.Variable([ 2.,4.,5.2,6.3,8.7,25.], name = "pt2")
        # axis_pt1 = hist.axis.Variable([5.2,6.3,8.7,25.], name = "pt1")
        # axis_pt2 = hist.axis.Variable([5.2,6.3,8.7,25.], name = "pt2")
        # axis_pt1 = hist.axis.Regular(23,2.,25., name = "pt1")
        # axis_pt2 = hist.axis.Regular(23,2.,25., name = "pt2")


    Jpts_axes = hist.axis.Regular(100, 0.,100., name = "boson_pt")
    Jraps_axes = hist.axis.Regular(120, -2.4, 2.4,name = "boson_rap")
    costhetas_axes =  hist.axis.Variable([round(-1. + 2*i/100,2) for i in range(101)], name = "costheta")

    # ress=np.linspace(-0.5,0.5,101)
    # pulls = np.linspace(-5,5,101)

    # if dataType=='data':
    # p.displayColumn(dataType,['pt1corr',"pt1corrsm", "k1corr", "k1corrsm", "Err1"])

    if genMass:
        if isJ:
            masscol = "smearedgenMass"
            # masscol = "Jpsigen_mass"
            pt1col = "smearedpt1"
            pt2col = "smearedpt2"
            # pt1col = "Muplusgen_pt"
            # pt2col = "Muminusgen_pt"
            # eta1col = "Muplusgen_eta"
            # eta2col = "Muminusgen_eta"
            eta1col = "Mupluscor_eta"
            eta2col = "Muminuscor_eta"
        else:
            masscol = "Jpsigen_mass"
            pt1col = "Muplusgen_pt"
            pt2col = "Muminusgen_pt"
            # masscol = "smearedgenMass"
            # pt1col = "smearedpt1"
            # pt2col = "smearedpt2"
            phi1col = "Muplusgen_phi"
            phi2col = "Muminusgen_phi"
            eta1col = "Muplus_eta"
            eta2col = "Muminus_eta"
    else:
        phi1col = "Muplus_phi"
        phi2col = "Muminus_phi"
        if isJ:
            if isData:
                masscol = "corrMass"
                # masscol = "modmass"
                pt1col = "pt1corr"
                pt2col = "pt2corr"
                eta1col = "Muplus_eta"
                eta2col = "Muminus_eta"
            else:
                #masscol = "corrMasssm"
                #pt1col = "pt1corrsm"
                #pt2col = "pt2corrsm"
                # masscol = "smearedgenMassred"
                # pt1col = "smearedpt1red"
                # pt2col = "smearedpt2red"
                # masscol = "corrMass"
                masscol = "Jpsicor_mass"
                # pt1col = "pt1corr"
                # pt2col = "pt2corr"
                pt1col = "Mupluscor_pt"
                pt2col = "Muminuscor_pt"
                eta1col = "Mupluscor_eta"
                eta2col = "Muminuscor_eta"
            # if isData:
            #     masscol = "corrMass"
            # else:
            #     masscol = "smearedgenMassred"
            # pt1col = "hybridpt1"
            # pt2col = "hybridpt2"
        else:
            if isData:
                masscol = "corrMass"
                pt1col = "pt1corr"
                pt2col = "pt2corr"
                eta1col = "Muplus_eta"
                eta2col = "Muminus_eta"
            else:
                #masscol = "corrMasssm"
                #pt1col = "pt1corrsm"
                #pt2col = "pt2corrsm"
                # masscol = "smearedgenMassred"
                # pt1col = "smearedpt1red"
                # pt2col = "smearedpt2red"
                masscol = "corrMass"
                pt1col = "pt1corr"
                pt2col = "pt2corr"
                eta1col = "Muplus_eta"
                eta2col = "Muminus_eta"
                # pt1col = "Muplusgen_pt"
                # pt2col = "Muminusgen_pt"


    axes = [axis_eta1,axis_eta2,axis_pt1,axis_pt2,axis_mass]
    cols = ["{}".format(eta1col),"{}".format(eta2col),"{}".format(pt1col),"{}".format(pt2col),"{}".format(masscol)]

    if not isData:
        if isJ:
            p.Histogram(dataType, "{}{}_{}".format("Jpsi" if isJ else "Z","gen" if genMass else "", "mc" if not isData else "data" ), [*cols], axes)
        else:
            p.Histogram(dataType, "{}{}_{}".format("Jpsi" if isJ else "Z","gen" if genMass else "", "mc" if not isData else "data" ), [*cols], axes)

    else:
        p.Histogram(dataType, "{}{}_{}".format("Jpsi" if isJ else "Z","gen" if genMass else "", "mc" if not isData else "data" ), cols, axes)

    axes = [axis_eta1,axis_eta2,axis_pt1,axis_pt2]
    cols = ["{}".format(eta1col),"{}".format(eta2col),"{}".format(pt1col),"{}".format(pt2col)]

    if isData or (not isJ):
        p.Histogram(dataType, "{}{}_{}_calVariables".format("Jpsi" if isJ else "Z","gen" if genMass else "", "mc" if not isData else "data" ), [*cols,'calVariables_weight'], axes)
    else:
        p.Histogram(dataType, "{}{}_{}_calVariables".format("Jpsi" if isJ else "Z","gen" if genMass else "", "mc" if not isData else "data" ), [*cols,'calVariables_weight'], axes)
    
    if not genMass:
        # axes = [Jpts_axes,Jraps_axes,costhetas_axes]
        # cols = ["Jpsi_pt","Jpsi_rap", "CStheta"]

        # if not isData and not isJ:
        #     cols.extend(["kinweight"])
        
        # # if not isData and isJ:
        # #     cols.extend(["tgrweight"])

        # p.Histogram(dataType, "{}_{}_bosonkin".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), cols, axes)

        # axis_pt1 = hist.axis.Regular(100, 0.,15., name = "pt1")

        # axes = [axis_eta1,axis_pt1]
        # cols = ["Muplus_eta","Muplus_pt"]

        # # if not isData and isJ:
        # #     cols.extend(["tgrweight"])

        # p.Histogram(dataType, "{}_{}_etapt".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), cols, axes)

        if not isData:
            if isJ:
                axis_pt1 = hist.axis.Regular(10,5.,25., name = "pt1")
                axis_pt2 = hist.axis.Regular(10,5.,25., name = "pt2")
                axis_res =  hist.axis.Regular(100,-0.05,0.05, name= "res")
                axes = [axis_pt1,axis_eta1,axis_res]
                cols = ["Muplusgen_pt","Muplusgen_eta","resplus"]

                p.Histogram(dataType, "{}_{}_resplus".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), cols, axes)

                axes = [axis_pt2,axis_eta2,axis_res]
                cols = ["Muminusgen_pt","Muminusgen_eta","resminus"]

                p.Histogram(dataType, "{}_{}_resminus".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), cols, axes)

                axes = [axis_pt1,axis_eta1]
                cols = ["Muplusgen_pt","Muplusgen_eta"]
                p.Histogram(dataType, "{}_{}_resplusmeans".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), [*cols,'calVariablesgen_weight'], axes)

                axes = [axis_pt2,axis_eta2]
                cols = ["Muminusgen_pt","Muminusgen_eta"]
                p.Histogram(dataType, "{}_{}_resminusmeans".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), [*cols,'calVariablesgen_weight'], axes)

            else:
                axis_pt1 = hist.axis.Regular(10,25.,100., name = "pt1")
                axis_pt2 = hist.axis.Regular(10,25.,100., name = "pt2")
                axis_res =  hist.axis.Regular(100,-0.05,0.05, name= "res")
                axes = [axis_pt1,axis_eta1,axis_res]
                cols = ["Muplusgen_pt","Muplusgen_eta","resplus"]

                p.Histogram(dataType, "{}_{}_resplus".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), cols, axes)

                axes = [axis_pt2,axis_eta2,axis_res]
                cols = ["Muminusgen_pt","Muminusgen_eta","resminus"]

                p.Histogram(dataType, "{}_{}_resminus".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), cols, axes)

                axes = [axis_pt1,axis_eta1]
                cols = ["Muplusgen_pt","Muplusgen_eta"]
                p.Histogram(dataType, "{}_{}_resplusmeans".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), [*cols,'calVariablesgen_weight'], axes)

                axes = [axis_pt2,axis_eta2]
                cols = ["Muminusgen_pt","Muminusgen_eta"]
                p.Histogram(dataType, "{}_{}_resminusmeans".format("Jpsi" if isJ else "Z", "mc" if not isData else "data"), [*cols,'calVariablesgen_weight'], axes)
    return p

def makepkg(histo, histotermsplus, histotermsminus, etas, pts, masses, good_idx, histoterms=True):

    edges = [etas,etas,pts,pts,masses]

    histo = histo[good_idx]
    if histoterms:
        #compute mean in each bin (integrating over mass) for pt-dependent terms
        histoden = np.sum(histo, axis=(-1))
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
    # pkg['grad1'] = np.gradient(np.sum(histodenfull,axis=(0,1,3)))
    # pkg['grad2'] = np.gradient(np.sum(histodenfull,axis=(0,1,2)))
    # print(pkg['grad1'])
    # print(pkg['grad2'])
    if histoterms:
        pkg['binCenters1'] = binCenters[0]
        pkg['binCenters2'] = binCenters[1]

    pkg['good_idx'] = good_idx

    return pkg

if isJ:
    inputFileMC = "/scratch1/ptscale/dataJan23/MuonGunUL2016_v718_RecJpsiPythiaPhotosPt8toInf_quality_novtx_noconstraint_idealgeom_corgensim/jpsicor.root"
    # inputFileMC = ["/scratch1/ptscale/dataSept22/MuonGunUL2016_v361_RecJpsiPythiaPhotosPt0to8_quality_novtx/MuonGunUL2016_v361_RecJpsiPythiaPhotosPt0to8_quality_novtx/220712_144859/0000/globalcor_0_*.root"]
    inputFileD = ["/scratch1/ptscale/dataSept22/MuonGunUL2016_v361a_RecDataJPsiFpost_quality_novtx/MuonGunUL2016_v361a_RecDataJPsiFpost_quality_novtx/220712_143056/0000/globalcor_0_*.root","/scratch1/ptscale/dataSept22/MuonGunUL2016_v361a_RecDataJPsiG_quality_novtx/MuonGunUL2016_v361a_RecDataJPsiG_quality_novtx/220712_143249/0000/globalcor_0_*.root","/scratch1/ptscale/dataSept22/MuonGunUL2016_v361a_RecDataJPsiH_quality_novtx/MuonGunUL2016_v361a_RecDataJPsiH_quality_novtx/220712_143757/0000/globalcor_0_*.root"]

else:
    # inputFileMC =['/scratch1/ptscale/dataSept22/MuonGunUL2016_v361_RecDYMuMu_quality_novtx/MuonGunUL2016_v361_RecDYMuMu_quality_novtx/220712_134831/0000/*.root','/scratch1/ptscale/dataSept22/MuonGunUL2016_v361_RecDYMuMu_quality_novtx/MuonGunUL2016_v361_RecDYMuMu_quality_novtx/220712_134831/0001/*.root']
    inputFileMC = "/scratch1/ptscale/dataJan23/MuonGunUL2016_v718_RecDYMuMu_quality_novtx_noconstraint_idealgeom_corgensim/jpsicor.root"
    inputFileD =['/scratch1/ptscale/dataSept22/MuonGunUL2016_v361_RecDataDYMuMuG_quality_novtx/MuonGunUL2016_v361_RecDataDYMuMuG_quality_novtx/220712_135305/0000/*.root','/scratch1/ptscale/dataSept22/MuonGunUL2016_v361_RecDataDYMuMuFpost_quality_novtx/MuonGunUL2016_v361_RecDataDYMuMuFpost_quality_novtx/220712_135432/0000/*.root','/scratch1/ptscale/dataSept22/MuonGunUL2016_v361_RecDataDYMuMuH_quality_novtx/MuonGunUL2016_v361_RecDataDYMuMuH_quality_novtx/220712_135125/0000/*.root']

outputDir = "JPsiInputData"
bMap = ROOT.TFile.Open("bFieldMap.root")
fsmear = ROOT.TFile.Open("outClosureTruthTheta.root")
calibZ = ROOT.TFile.Open("outClosureTruthPhi.root")

flag = ""
if plotdistr: flag="_mukin"

treeName = 'tree'
if isJ: treeName = 'tree'

p = RDFtree(outputDir = outputDir, inputFile = inputFileMC, treeName=treeName, outputFile="{}MC{}.root".format('JPsi' if isJ else 'Z',flag), pretend=pretend)
calibMC = ROOT.TFile.Open("calibrationJMC_smeared_v718.root")

corr_flag = False
if layerCorrs: corr_flag = True
if correct:
    modules=[trigger_rew(), defines(), smear(), ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, True,corr_flag),ROOT.CSvariableProducer()]
else:
    modules=[trigger_rew(), defines(), smear(), ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, False,corr_flag),ROOT.CSvariableProducer()]
if not isJ:
    # modules=[ROOT.genLeptonSelector(),definesZ(),defines()]
    modules=[defines()]
    if correct:
        modules.extend([smear(),ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, True,corr_flag),ROOT.CSvariableProducer(),reweightycostheta(isJ=isJ)])
    else:
        modules.extend([ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, False,corr_flag),smear(),ROOT.CSvariableProducer(),reweightycostheta(isJ=isJ)])

p.branch(nodeToStart='input', nodeToEnd='defs', modules=modules)
dataMC = makeData(p, isData=False, isJ=isJ)

if not plotdistr:
    pGen = RDFtree(outputDir = outputDir, inputFile = inputFileMC, treeName=treeName, outputFile="{}Gen{}.root".format('JPsi' if isJ else 'Z',flag), pretend=pretend)
    calibMC = ROOT.TFile.Open("calibrationJMC_smeared_v718.root")
    corr_flag = False
    if layerCorrs: corr_flag = True
    if correct:
        modules=[trigger_rew(), defines(),smear(isData=True),ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, True,corr_flag),ROOT.CSvariableProducer()]
    else:
        modules=[trigger_rew(), defines(),smear(isData=True),ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, False,corr_flag),ROOT.CSvariableProducer()]
    if not isJ:
        # modules=[ROOT.genLeptonSelector(),definesZ(),defines()]
        modules=[defines()]
        if correct:
            modules.extend([smear(isData=True),ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, False,corr_flag),ROOT.CSvariableProducer(),reweightycostheta(isJ=isJ,isData=True)])
        else:
            modules.extend([ROOT.applyCalibration(bMap,calibMC,calibZ,fsmear,False, False,corr_flag),ROOT.CSvariableProducer(),reweightycostheta(isJ=isJ,isData=True),smear(isData=True)])
    pGen.branch(nodeToStart='input', nodeToEnd='defs', modules=modules)
    dataGen = makeData(pGen, genMass=True, isData=False,isJ=isJ)

p2 = RDFtree(outputDir = outputDir, inputFile = inputFileD, treeName=treeName, outputFile="{}Data{}.root".format('JPsi' if isJ else 'Z',flag), pretend=pretend)
calibD = ROOT.TFile.Open("calibrationJDATA_smeared_rewtgr.root")
datajson = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
# lumi = "luminosityBlock"
# if isJ:
#     lumi = "lumi"
lumi = "lumi"

if correct:
    modules=[ROOT.isGoodLumi(datajson,lumi), trigger_rew(), defines(isData=True),ROOT.applyCalibration(bMap,calibD,calibZ,fsmear,True, True, corr_flag),ROOT.CSvariableProducer()]
else:
    modules=[ROOT.isGoodLumi(datajson,lumi), trigger_rew(), defines(isData=True),ROOT.applyCalibration(bMap,calibD,calibZ,fsmear,True, False, corr_flag),ROOT.CSvariableProducer()]
if not isJ:
    modules=[ROOT.isGoodLumi(datajson,lumi),defines(isData=True), ROOT.CSvariableProducer()]
    if correct:
        modules.extend([ROOT.applyCalibration(bMap,calibD,calibZ,fsmear,True, True, corr_flag)])
    else:
        modules.extend([ROOT.applyCalibration(bMap,calibD,calibZ,fsmear,True, False, corr_flag)])

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

p.getOutput()
p2.getOutput()

if not plotdistr:
    rGen.Print()
    pGen.getOutput()
print("------------")
rMC.Print()
print("------------")
rD.Print()

print('all samples processed in {} s'.format(time.time()-start))

nEtaBins = 24
nPtBins = 1
nMassBins = 100

etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
# etas = np.linspace(-1., 1., nEtaBins+1, dtype='float64')
if isJ:
    masses = np.linspace(2.9069,3.3069, 100+1, dtype='float64')
    masses_gen=masses
else:
    masses = np.linspace(75., 110., 1000+1, dtype='float64')
    masses_gen = np.linspace(75., 110., 1000+1, dtype='float64')
pts = np.linspace(20.,100.,nPtBins+1, dtype='float64')
# pts = np.array([ 5., 6., 6.9, 7.6, 8.4, 9.3, 10.3 ,11.3 ,12.5, 13.9,15.5, 17.8, 23.], dtype='float64')

if isJ:
    fname = "JPsiInputData/JPsiMC.pkl.lz4"
    with (lz4.frame.open(fname, "r")) as openfile:
        resultdict_mc = pickle.load(openfile)
    fname = "JPsiInputData/JPsiData.pkl.lz4"
    with (lz4.frame.open(fname, "r")) as openfile:
        resultdict_data = pickle.load(openfile)
    fname = "JPsiInputData/JPsiGen.pkl.lz4"
    with (lz4.frame.open(fname, "r")) as openfile:
        resultdict_gen = pickle.load(openfile)
else:
    fname = "JPsiInputData/ZMC.pkl.lz4"
    with (lz4.frame.open(fname, "r")) as openfile:
        resultdict_mc = pickle.load(openfile)
    fname = "JPsiInputData/ZData.pkl.lz4"
    with (lz4.frame.open(fname, "r")) as openfile:
        resultdict_data = pickle.load(openfile)
    fname = "JPsiInputData/ZGen.pkl.lz4"
    with (lz4.frame.open(fname, "r")) as openfile:
        resultdict_gen = pickle.load(openfile)

if isJ:
    hMCgen = resultdict_gen['Jpsigen_mc'].values()
    hMC = resultdict_mc['Jpsi_mc'].values()
    hD = resultdict_data['Jpsi_data'].values()
    histotermsMC = np.array(resultdict_mc['Jpsi_mc_calVariables'].values())
    histotermsD = np.array(resultdict_data['Jpsi_data_calVariables'].values())
else:
    hMCgen = resultdict_gen['Zgen_mc'].values()
    hMC = resultdict_mc['Z_mc'].values()
    hD = resultdict_data['Z_data'].values()
    histotermsMC = np.array(resultdict_mc['Z_mc_calVariables'].values())
    histotermsD = np.array(resultdict_data['Z_data_calVariables'].values())


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
    pklfileGen = 'calInputZMCgen_{}etaBins_{}ptBins_smeared.pkl'.format(len(etas)-1, len(pts)-1)
    filehandler = open('calInputZMCgen_{}etaBins_{}ptBins_smeared.pkl'.format(len(etas)-1, len(pts)-1), 'wb')
else:
    pklfileGen = 'calInputJMCgen_{}etaBins_{}ptBins_smeared.pkl'.format(len(etas)-1, len(pts)-1)
    filehandler = open('calInputJMCgen_{}etaBins_{}ptBins_smeared.pkl'.format(len(etas)-1, len(pts)-1), 'wb')

print(pklfileData, pklfileMC)

if isJ:
    good_idx_MC = np.nonzero(np.sum(hMCgen,axis=(-1))>1000.)
    good_idx_D = np.nonzero((np.sum(hMCgen,axis=(-1))>1000.) & (np.sum(hD,axis=(-1))>500.))
    print("good_idx MC size", good_idx_MC[0].shape)
    print("good_idx Data size", good_idx_D[0].shape)
    print('number of effective events', np.sum(hMC[good_idx_MC]), np.sum(hD[good_idx_D]))
else:
    good_idx_MC = np.nonzero(np.sum(hMCgen,axis=(-1))>750.)
    good_idx_D = np.nonzero((np.sum(hMCgen,axis=(-1))>750.) & (np.sum(hD,axis=(-1))>750.))
    # good_idx_D =  np.nonzero((np.sum(hD[...,th1:th2],axis=(-1))/np.sum(hD,axis=(-1))>0.9) & (np.sum(hMCgen,axis=(-1))>750.))
    # good_idx_D =  np.nonzero((np.sum(hD,axis=(-1))>750.))
    print("good_idx MC size", good_idx_MC[0].shape)
    print("good_idx Data size", good_idx_D[0].shape)
    # print("good_idx Data size", good_idx_D[0].shape)
    # print('number of effective events', np.sum(hMCgen[good_idx_MC]),np.sum(hMC[good_idx_MC]))
    # print('number of effective events', np.sum(hMCgen),np.sum(hMC))

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
