#!/usr/bin/env python
# coding: utf-8

# In[1]:


import narf
import RDFtree
import ROOT
import hist
import jax
import jax.numpy as np
import numpy as onp
from scipy.interpolate import make_interp_spline, BSpline

import pickle
import lz4.frame
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])

def scale(A,e,M,pt,q):
    print(A.shape,pt.shape,q.shape)
    # return A + q*M/k - e*k
    return A + q*M*pt - e/pt

def sigmasq(a, b, c, d, k):
    return a + c/k**2 + b/(1+d*k**2)

def scalesigma(parms, qs, ks):
    A = parms[..., 0, np.newaxis, np.newaxis]
    e = parms[..., 1, np.newaxis, np.newaxis]
    M = parms[..., 2, np.newaxis, np.newaxis]
    
    qs = qs[np.newaxis, :, np.newaxis]
    ks = ks[:, np.newaxis, :]
    
    scaleout = scale(A,e,M,ks,qs)
    return scaleout

def TH2F2np(histo, overflow=False):
    binningx = onp.array(histo.GetXaxis().GetXbins())
    binningy = onp.array(histo.GetYaxis().GetXbins())
    if overflow:
        binningx = onp.append(binningx, [onp.inf])
        binningy = onp.append(binningy, [onp.inf])
    if overflow:
        content = onp.zeros((histo.GetNbinsX() + 1, histo.GetNbinsY() + 1))
    else:
        content = onp.zeros((histo.GetNbinsX(), histo.GetNbinsY()))
    for xbin in range(1, histo.GetNbinsX() + (2 if overflow else 1)):
        for ybin in range(1, histo.GetNbinsY() + (2 if overflow else 1)):
            content[xbin - 1, ybin - 1] = histo.GetBinContent(xbin, ybin)
    return content, binningx, binningy

fileMCafterZ = h5py.File('scalemctruth_afterNoAlign.hdf5', mode='r')

scalemcafterZ = fileMCafterZ['scale'][:]
scalemcErrafterZ = fileMCafterZ['scaleErr'][:]
bins = fileMCafterZ['bins'][:]

# fileMCafter = h5py.File('scalemctruth_afterZ.hdf5', mode='r')

# scalemcafter = fileMCafter['scale'][:]
# scalemcErrafter = fileMCafter['scaleErr'][:]

fileMCbefore = h5py.File('scalemctruth_before.hdf5', mode='r')

scalemcbefore = fileMCbefore['scale'][:]
scalemcErrbefore = fileMCbefore['scaleErr'][:]
bins = fileMCbefore['bins'][:]

nEtaBins = 24
etas = np.linspace(-2.4, 2.4, nEtaBins+1, dtype='float64')
etasC = (etas[:-1] + etas[1:]) / 2

fIn = ROOT.TFile.Open("outClosureTruth.root")
A = np.array(fIn.Get("A"))[1:-1]
e = np.array(fIn.Get("e"))[1:-1]
M = np.array(fIn.Get("M"))[1:-1]
Merr = np.zeros_like(M)

for i in range(1,fIn.Get("M").GetNbinsX()+1):
    Merr[i-1] = fIn.Get("M").GetBinError(i)

covmodel = TH2F2np(fIn.Get("covariance_matrix"))[0]
covmodel = covmodel.reshape((nEtaBins,7,nEtaBins,7))[:,:3,:,:3].reshape((nEtaBins*3,nEtaBins*3))

parms = np.stack((-1*A,-1*e,-1*M),axis=-1)

# binsJ = np.linspace(5.,25.,11)
# binsZ = np.linspace(25.,100.,11)
# bins = np.concatenate((binsJ,binsZ))

ks = bins

qcs = onp.array([-1.,1.], dtype=np.float64)
scaleModel = scalesigma(parms, qcs, ks)

print(bins[0,...],scalemcbefore.shape,scaleModel.shape)
print(scalemcbefore[0,0,:],scaleModel[0,0,:])

scalejac = jax.jit(jax.jacfwd(scalesigma))(parms, qcs, ks)

scalejac = np.reshape(scalejac[:,1,...], (-1,covmodel.shape[0]))
covScaleModel = np.matmul(scalejac,np.matmul(covmodel,scalejac.T))
ScaleErrsModel =  np.sqrt(np.diag(covScaleModel)).reshape((scaleModel.shape[0],scaleModel.shape[-1]))
print(ScaleErrsModel[0,...])
ScaleErrsModel = Merr[:,np.newaxis]*bins
print(ScaleErrsModel[0,...])

for i in range(24):
    fig, (ax1) = plt.subplots()
    hep.cms.text('work in progress', ax=ax1)
    ax1.text(0.95, 0.95, '$\eta$ {}'.format(round(etasC[i],2)),
        verticalalignment='top', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=12)

    ax1.errorbar(-1*bins[i,...],scalemcafterZ[i,0,...], yerr=scalemcErrafterZ[i,0,...], marker=".", color = "blue", label = '$after J/\psi$ mass fit corrections', fmt=".")
    ax1.errorbar(bins[i,...],scalemcafterZ[i,1,...], yerr=scalemcErrafterZ[i,1,...], marker=".", color = "blue", fmt=".")

    xnew = np.linspace(bins[i,...].min(), bins[i,...].max(), 100) 
    modeltot =scaleModel[i,1,...]
    errmodeltot = ScaleErrsModel[i,...]
    print(modeltot,"+/-",errmodeltot)
    spl = make_interp_spline(bins[i,...], modeltot, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    splUp = make_interp_spline(bins[i,...], modeltot+errmodeltot, k=3)  # type: BSpline
    power_smoothUp = splUp(xnew)
    splDown = make_interp_spline(bins[i,...], modeltot-errmodeltot, k=3)  # type: BSpline
    power_smoothDown = splDown(xnew)
    ax1.plot(xnew,power_smooth, color="red")
    ax1.fill_between(xnew,power_smoothDown,power_smoothUp, color="red", alpha=0.3)

    modeltot =scaleModel[i,0,...]
    errmodeltot = ScaleErrsModel[i,...]
    spl = make_interp_spline(bins[i,...], modeltot, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    splUp = make_interp_spline(bins[i,...], modeltot+errmodeltot, k=3)  # type: BSpline
    power_smoothUp = splUp(xnew)
    splDown = make_interp_spline(bins[i,...], modeltot-errmodeltot, k=3)  # type: BSpline
    power_smoothDown = splDown(xnew)
    ax1.plot(-1*xnew,power_smooth, color="red")
    ax1.fill_between(-1*xnew,power_smoothDown,power_smoothUp, color="red", alpha=0.3)

    # ax1.errorbar(-1*bins[i,...],scalemcafter[i,0,...], yerr=scalemcErrafter[i,0,...], marker=".", color = "red", label = '$after J/\psi$ mass fits corrections', fmt=".")
    # ax1.errorbar(bins[i,...],scalemcafter[i,1,...], yerr=scalemcErrafter[i,1,...], marker=".", color = "red", fmt=".")
    ax1.errorbar(-1*bins[i,...],scalemcbefore[i,0,...], yerr=scalemcErrbefore[i,0,...], marker=".", color = "green", label = 'before corrections', fmt=".")
    ax1.errorbar(bins[i,...],scalemcbefore[i,1,...], yerr=scalemcErrbefore[i,1,...], marker=".", color = "green", fmt=".")
    # ax1.fill_between(np.array([-100.,100]),-0.0001,0.0001, color="pink", alpha=0.2)
    # ax1.fill_between(-bins[i,...],-0.0001,0.0001, color="red", alpha=0.2)
    ax1.set_ylabel('scale')
    ax1.set_xlabel('$qp_T$')
    ax1.legend(loc='upper left', frameon=False)
    # ax1.set_ylim(-0.0005,0.0005)
    plt.tight_layout()
    plt.savefig('PlotsScaleTruthAfter/scale_eta{}.png'.format(round(etasC[i],2)))

assert(0)
# In[15]:


fname = "JPsiInputData/JPsiMC.pkl.lz4"
with (lz4.frame.open(fname, "r")) as openfile:
    resultdict_mc = pickle.load(openfile)

fname = "JPsiInputData/JPsiData.pkl.lz4"
with (lz4.frame.open(fname, "r")) as openfile:
    resultdict_data = pickle.load(openfile)


# In[16]:


import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])
import numpy as np

fig, ax1 = plt.subplots()
hep.histplot(np.sum(resultdict_mc['Jpsi_mc_CSdeltaphi'].values(),axis=(0,1))/np.sum(resultdict_mc['Jpsi_mc_CSdeltaphi'].values()), resultdict_mc['Jpsi_mc_CSdeltaphi'].axes[-1].edges, ax=ax1, label = ["mc"])
# hep.histplot(resultdict_data['Jpsi_data'].values()[0,0,:]/np.sum(resultdict_data['Jpsi_data'].values()[0,0,:]), resultdict_data['Jpsi_data'].axes[-1].edges, ax=ax1, label = ["data"])

ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()
hep.hist2dplot(resultdict_data['Jpsi_data_etapt'].values(),resultdict_data['Jpsi_data_etapt'].axes[0].edges, resultdict_data['Jpsi_data_etapt'].axes[1].edges, ax=ax1, label = ["mc"])
# hep.histplot(resultdict_data['Jpsi_data_etapt'].values()/resultdict_data['Jpsi_data_etapt'].values(),resultdict_data['Jpsi_data_etapt'].axes[1].edges, ax=ax1, label = ["data"])
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()
hep.histplot(np.sum(resultdict_data['Jpsi_data_etapt'].values(),axis=0)/np.sum(resultdict_data['Jpsi_data_etapt'].values()), resultdict_data['Jpsi_data_etapt'].axes[1].edges, ax=ax1, label = ["data"])
hep.histplot(np.sum(resultdict_mc['Jpsi_mc_etapt'].values(),axis=0)/np.sum(resultdict_mc['Jpsi_mc_etapt'].values()), resultdict_mc['Jpsi_mc_etapt'].axes[1].edges, ax=ax1, label = ["mc"])
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()
hep.histplot(np.sum(resultdict_data['Jpsi_data_etapt'].values(),axis=1)/np.sum(resultdict_data['Jpsi_data_etapt'].values()), resultdict_data['Jpsi_data_etapt'].axes[0].edges, ax=ax1, label = ["data"])
hep.histplot(np.sum(resultdict_mc['Jpsi_mc_etapt'].values(),axis=1)/np.sum(resultdict_mc['Jpsi_mc_etapt'].values()), resultdict_mc['Jpsi_mc_etapt'].axes[0].edges, ax=ax1, label = ["mc"])
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()
hep.histplot(np.sum(resultdict_data['Jpsi_data_bosonkin'].values(),axis=(1,2))/np.sum(resultdict_data['Jpsi_data_bosonkin'].values()), resultdict_data['Jpsi_data_bosonkin'].axes[0].edges, ax=ax1, label = ["data"])
hep.histplot(np.sum(resultdict_mc['Jpsi_mc_bosonkin'].values(),axis=(1,2))/np.sum(resultdict_mc['Jpsi_mc_bosonkin'].values()), resultdict_mc['Jpsi_mc_bosonkin'].axes[0].edges, ax=ax1, label = ["mc"])
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()


# In[ ]:




