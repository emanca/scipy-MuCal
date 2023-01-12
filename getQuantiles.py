import h5py
import pickle
import ROOT
import numpy as np
import time

def array2hist(array, hist, err):
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

f = open('calInputJMCgen_24etaBins_1ptBins_smeared.pkl','rb')
h = pickle.load(f)
# fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin.hdf5', mode='r')
# h = fileJPsiMC['Jpsi_distr_mc'][:]
h=np.sum(h,axis=(0,1,3,4))
print(h.shape)
# massBins = np.linspace(2.9, 3.3, 101, dtype='float64')
massBins = np.array([ 1.1, 3.2, 5.2, 6.1, 7.8, 25.])
# # massBins = np.linspace(25.,60.,4+1, dtype='float64')
hmass = ROOT.TH1D("h","h", massBins.shape[0]-1,massBins)
hmass = array2hist(h, hmass, np.sqrt(h))
quantiles = np.array(np.linspace(0.,1.,massBins.shape[0], dtype='float64'))
y=0.
q=np.zeros([massBins.shape[0]])
y=hmass.GetQuantiles(massBins.shape[0],q,quantiles)
print(q,y)

# f = open('calInputJMC_48etaBins_6ptBins.pkl','rb')
# pkg = pickle.load(f)
# h = pkg['dataset']

# print(h.shape)
# massBins = pkg['edges'][-1]
# massWidth = massBins[1:]-massBins[:-1]
# massBinsC = 0.5*(massBins[1:]+massBins[:-1])
# print(massBinsC)
# print(massWidth.shape,np.sum(h,axis=(0)).shape)
# print(massBins.shape[0],np.sum(h,axis=(0)).shape)
# hmass = ROOT.TH1D("h","h", massBins.shape[0]-1,massBins)
# hmass = array2hist(np.sum(h,axis=(0))/massWidth, hmass)
# hmassgen = ROOT.TH1D("hgen","hgen", massBins.shape[0]-1,massBins)
# hmassgen = array2hist(np.sum(hgen,axis=(0,1,2,3))/massWidth, hmassgen)
# hmass.Draw()
# hmassgen.Draw()
# time.sleep(1000)