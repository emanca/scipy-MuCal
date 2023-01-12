import h5py
import numpy as np
import ROOT
from root_numpy import array2hist
import pickle
import matplotlib.pyplot as plt
import mplhep as hep
# matplotlib stuff
plt.style.use([hep.style.CMS])

fileJPsiData = h5py.File('JPsiInputData/JPsiData_mukin.hdf5', mode='r')
fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin.hdf5', mode='r')

hdata_eta = np.sum(fileJPsiData['Jpsi_distr_data'][:],axis=(1,2,3))
hmc_eta = np.sum(fileJPsiMC['Jpsi_distr_mc'][:],axis=(1,2,3))

hdata_pt = np.sum(fileJPsiData['Jpsi_distr_data'][:],axis=(0,1,3))
hmc_pt = np.sum(fileJPsiMC['Jpsi_distr_mc'][:],axis=(0,1,3))

nEtaBins = 48
nPtBins = 100

etas = fileJPsiData['edges_Jpsi_distr_data_0']
pts = fileJPsiData['edges_Jpsi_distr_data_2']

# data over mc
#eta1
fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
hep.cms.text('work in progress', ax=ax1)
# ax1.set_title("scale", fontsize=18)
hep.histplot(hmc_eta/np.sum(fileJPsiMC['Jpsi_distr_mc'][:]),etas, ax=ax1, label = ["mc"])
hep.histplot(hdata_eta/np.sum(fileJPsiData['Jpsi_distr_data'][:]),etas, ax=ax1, label = ["data"], color='red')
ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
ax1.set_xlabel('$\eta$ of positive muon')

plt.tight_layout()
plt.savefig("FinalPlots/etadistrJ.png")
plt.savefig("FinalPlots/etadistrJ.pdf")
plt.clf()

#pt1
fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
hep.cms.text('work in progress', ax=ax1)
# ax1.set_title("scale", fontsize=18)
hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_distr_mc'][:]),pts, ax=ax1, label = ["mc"])
hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_distr_data'][:]),pts, ax=ax1, label = ["data"], color='red')
ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
ax1.set_xlabel('$p_{T}$ of positive muon')

plt.tight_layout()
plt.savefig("FinalPlots/ptdistrJ.png")
plt.savefig("FinalPlots/ptdistrJ.pdf")
plt.clf()


# hdata_pt = np.sum(fileJPsiData['Jpsi_distr_data'][:],axis=(0,1))
# hmc_pt = np.sum(fileJPsiMC['Jpsi_distr_mc'][:],axis=(0,1))
# #pt1
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.hist2dplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_distr_mc'][:]),pts, pts, ax=ax1, label = ["mc"])
# # hep.hist2dplot(hdata_pt/np.sum(fileJPsiData['Jpsi_distr_data'][:]),pts, pts, ax=ax1, label = ["data"], color='red')
# ax1.zaxis.set_scale('log')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# # ax1.set_xlabel('$p_{T}$ of positive muon')

# plt.tight_layout()
# plt.savefig("FinalPlots/ptdistr2d.png")
# plt.savefig("FinalPlots/ptdistr2d.pdf")
# plt.clf()

# hdata_pt = np.sum(fileJPsiData['Jpsi_distr2_data'][:],axis=-1)
# hmc_pt = np.sum(fileJPsiMC['Jpsi_distr2_mc'][:],axis=-1)

# # h = ROOT.TH1D("h","h", len(logpts)-1,np.array(logpts))
# # h = array2hist(hmc_pt, h)

# # quantiles = np.array(np.linspace(0.,1.,len(logpts)-1, dtype='float64'))
# # y=0.
# # q=np.zeros([len(logpts)-1])
# # y=h.GetQuantiles(len(logpts)-1,q,quantiles)
# # print(np.around(q,2),y)


# cosphis = [round(-1. + 0.2*i/5,2) for i in range(11)]
# logpts = np.linspace(-1.5,1.5,31)
# # cosphis = [round(-1. + 2.*i/20,2) for i in range(21)]
# # logpts = np.linspace(-2.5,2.5,41)


# hdata_pt = np.sum(fileJPsiData['Jpsi_distr2_data'][:],axis=-1)
# hmc_pt = np.sum(fileJPsiMC['Jpsi_distr2_mc'][:],axis=-1)

# # h = ROOT.TH1D("h","h", len(logpts)-1,np.array(logpts))
# # h = array2hist(hmc_pt, h)

# # quantiles = np.array(np.linspace(0.,1.,len(logpts)-1, dtype='float64'))
# # y=0.
# # q=np.zeros([len(logpts)-1])
# # y=h.GetQuantiles(len(logpts)-1,q,quantiles)
# # print(np.around(q,2),y)


# #logpt1/pt2
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_distr2_mc'][:]),logpts, ax=ax1, label = ["mc"])
# hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_distr2_data'][:]),logpts, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$\log{(p^+_{T}/p^-_{T})}$')

# plt.tight_layout()
# plt.savefig("FinalPlots/logptdistrJ.png")
# plt.savefig("FinalPlots/logptdistrJ.pdf")
# plt.clf()

# hdata_pt = np.sum(fileJPsiData['Jpsi_distr2_data'][:],axis=0)
# hmc_pt = np.sum(fileJPsiMC['Jpsi_distr2_mc'][:],axis=0)


# #cosphi
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_distr2_mc'][:]),cosphis, ax=ax1, label = ["mc"])
# hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_distr2_data'][:]),cosphis, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$\cos{(\Delta \phi)}$')

# plt.tight_layout()
# plt.savefig("FinalPlots/cosphidistrJ.png")
# plt.savefig("FinalPlots/cosphidistrJ.pdf")
# plt.clf()

# #Jpt
# Jpts = np.linspace(0,100,101)
# hdata_pt = np.sum(fileJPsiData['Jpsi_rew_data'][:],axis=(1,2))
# hmc_pt = np.sum(fileJPsiMC['Jpsi_rew_mc'][:],axis=(1,2))
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_rew_mc'][:]),Jpts, ax=ax1, label = ["mc"])
# hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_rew_data'][:]),Jpts, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$Z~p_T$')

# plt.tight_layout()
# plt.savefig("FinalPlots/Jpts.png")
# plt.savefig("FinalPlots/Jpts.pdf")
# plt.clf()

# #Jrapidity
# # Jpts = np.linspace(0,50,101)
# Jraps = np.linspace(-2.4, 2.4, 120, dtype='float64')
# costhetas =[round(-1. + 2*i/100,2) for i in range(101)]
# hdata_pt = np.sum(fileJPsiData['Jpsi_rew_data'][:],axis=(0,1))
# hmc_pt = np.sum(fileJPsiMC['Jpsi_rew_mc'][:],axis=(0,1))
# print(hdata_pt)
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_rew_mc'][:]),Jraps, ax=ax1, label = ["mc"])
# hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_rew_data'][:]),Jraps, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$Z~rapidity$')

# plt.tight_layout()
# plt.savefig("FinalPlots/Jrap.png")
# plt.savefig("FinalPlots/Jrap.pdf")
# plt.clf()

# #Jcostheta
# hdata_pt = np.sum(fileJPsiData['Jpsi_rew_data'][:],axis=(0,-1))
# hmc_pt = np.sum(fileJPsiMC['Jpsi_rew_mc'][:],axis=(0,-1))
# print(hdata_pt)
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_rew_mc'][:]),costhetas, ax=ax1, label = ["mc"])
# hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_rew_data'][:]),costhetas, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$\cos{\\theta^*}$')

# plt.tight_layout()
# plt.savefig("FinalPlots/costhetaJ.png")
# plt.savefig("FinalPlots/costhetaJ.pdf")
# plt.clf()

fileJPsiMC = h5py.File('JPsiInputData/JPsiMC.hdf5', mode='r')

print(list(fileJPsiMC.keys()))
massBins = fileJPsiMC['edges_Jpsi_gen_4'][:]
massBinsC = 0.5*(massBins[1:]+massBins[:-1])
hmass = np.sum(fileJPsiMC['Jpsi_gen'][:],axis=(1,2,3))
hmasssm = np.sum(fileJPsiMC['Jpsi_mc'][:],axis=(1,2,3))
bins=fileJPsiMC['edges_Jpsi_gen_0'][:]
massesPt =[]
massesPtrec =[]
for i in range(len(bins)-1):
    massPt = np.average(massBinsC, weights=hmass[i,:])
    massesPt.append(massPt)
    massPt = np.average(massBinsC, weights=hmasssm[i,:])
    massesPtrec.append(massPt)

# fileJ = open("calInputJMC_48etaBins_6ptBins.pkl", "rb")
# pkgJ = pickle.load(fileJ)
# datasetJ = pkgJ['dataset']
# massesJ = pkgJ['edges'][-1]
# massBinsC = 0.5*(massesJ[1:]+massesJ[:-1])
# good_idxJ = pkgJ['good_idx']
# filegen = open("calInputJMCgen_48etaBins_6ptBins.pkl", "rb")
# datasetgen = pickle.load(filegen)
# datasetgen = datasetgen[good_idxJ]
# massesPt =[]
# massesPtrec =[]
# for i in range(datasetgen.shape[0]):
#     massPt = np.average(massBinsC, weights=datasetgen[i,:])
#     massesPt.append(massPt)
#     massPt = np.average(massBinsC, weights=datasetJ[i,:])
#     massesPtrec.append(massPt)
# bins=np.linspace(0,datasetgen.shape[0]+1,datasetgen.shape[0]+1)

fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [3, 1]})
hep.cms.text('work in progress', ax=ax1)
hep.histplot(np.array(massesPtrec),bins, ax=ax1,histtype = 'errorbar', label="smeared gen mass", marker="^",markersize=5)
hep.histplot(np.array(massesPt),bins, ax=ax1, histtype = 'errorbar', label = "gen mass", marker="^",markersize=5)
ax1.legend(loc='upper center', frameon=True)
hep.histplot(np.array(massesPtrec)/np.array(massesPt),bins, ax=ax2,histtype = 'errorbar', marker="^",markersize=5)
# ax1.set_ylabel('gen mass')
ax2.set_xlabel('$\eta$')
ax1.set_ylim(90.4,91)
ax2.set_ylim(1-0.0005,1+0.0005)
plt.savefig('FinalPlots/Zmass.pdf')
