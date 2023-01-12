import numpy as np
import math
from root_numpy import hist2array
import ROOT
import matplotlib.pyplot as plt

etaBins = np.array([-2.4+i*0.1 for i in range(49)])
etasC = (etaBins[:-1] + etaBins[1:]) / 2

fd = ROOT.TFile.Open('/scratchnvme/emanca/scipy-MuCal/calibrationJDATA_aftersm.root')

had = hist2array(fd.Get('a'))
hbd = hist2array(fd.Get('b'))
hcd = hist2array(fd.Get('c'))

fmc = ROOT.TFile.Open('/scratchnvme/emanca/scipy-MuCal/calibrationJMC_aftersm.root')
hamc = hist2array(fmc.Get('a'))
hbmc = hist2array(fmc.Get('b'))
hcmc = hist2array(fmc.Get('c'))

with np.load('/scratchnvme/emanca/scipy-MuCal/unbinnedfitglobalitercorscale.npz') as f:
    hdd = f["xs"][...,-1]
    hb = f["xs"][...,5]

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

def getDataRes(eta, pt):
    bineta = np.digitize(np.array([eta]), etaBins)[0]-1
    # bineta=0 #last bin for everything
    ad = had[bineta]
    bd = hbd[bineta]
    cd = hcd[bineta]
    dd = hdd[bineta]*hdd[bineta]

    L=computeTrackLength(eta)
    L2 = math.pow(L,2)
    pt2 = pt*pt
    # resd = (ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2))
    resd = ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2)


    return np.sqrt(resd)

def getMCRes(eta, pt):
    bineta = np.digitize(np.array([eta]), etaBins)[0]-1
    # bineta=0 #last bin for everything
    ad = hamc[bineta]
    bd = hbmc[bineta]
    cd = hcmc[bineta]
    dd = hdd[bineta]*hdd[bineta]

    L=computeTrackLength(eta)
    L2 = math.pow(L,2)
    pt2 = pt*pt
    # resd = (ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2))
    resd = ad*L + cd*pt2*L2 + bd*L*1./(1+dd*1./pt2)

    
    return np.sqrt(resd)

pts = np.array([2.6, 3.4, 4.4, 5.7, 7.4, 10.2, 13., 18.,25.])
ptsC = (pts[:-1] + pts[1:]) / 2

resMC=[]
resD=[]
for i in range(ptsC.shape[0]):
    resMC.append(getMCRes(2.,ptsC[i]))
    resD.append(getDataRes(2.,ptsC[i]))

fig, ax1 = plt.subplots()
ax1.plot(ptsC, np.array(resMC))
ax1.plot(ptsC, np.array(resD))
plt.show()