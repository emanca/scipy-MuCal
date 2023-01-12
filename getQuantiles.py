from root_numpy import array2hist
import h5py
import pickle
import ROOT
import numpy as np
import time

f = open('calInputZMCgen_48etaBins_6ptBins.pkl','rb')
h = pickle.load(f)

print(h.shape)
# massBins = np.linspace(2.9, 3.3, 501, dtype='float64')
# massBins = np.linspace(1.5,25.,6+1, dtype='float64')
massBins = np.linspace(25.,60.,4+1, dtype='float64')
hmass = ROOT.TH1D("h","h", massBins.shape[0]-1,massBins)
hmass = array2hist(np.sum(h,axis=(0,1,3,-1)), hmass)
quantiles = np.array(np.linspace(0.,1.,massBins.shape[0], dtype='float64'))
y=0.
q=np.zeros([massBins.shape[0]])
y=hmass.GetQuantiles(massBins.shape[0],q,quantiles)
print(q,y)

# f = open('calInputZMCgen_48etaBins_6ptBins.pkl','rb')
# h = pickle.load(f)

# print(h.shape)
# massBins = np.linspace(75., 115., 100+1, dtype='float64')
# print(massBins.shape[0],np.sum(h,axis=(0,1,2,3)).shape)
# hmass = ROOT.TH1D("h","h", massBins.shape[0]-1,massBins)
# hmass = array2hist(np.sum(h,axis=(0,1,2,3)), hmass)
# hmass.Draw()
# time.sleep(1000)