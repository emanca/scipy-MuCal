import h5py
import numpy as np
import scipy
import ROOT
from root_numpy import array2hist
import pickle
import matplotlib.pyplot as plt
import mplhep as hep
import time
# matplotlib stuff
plt.style.use([hep.style.CMS])

fileJPsiData = h5py.File('JPsiInputData/JPsiData_mukin.hdf5', mode='r')
fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin.hdf5', mode='r')
# print(list(fileJPsiMC.keys()))
# hdata_eta = np.sum(fileJPsiData['Jpsi_distr_data'][:],axis=(1,2,3))
# hmc_eta = np.sum(fileJPsiMC['Jpsi_distr_mc'][:],axis=(1,2,3))

hdata_pt =np.sum(fileJPsiData['test_data'][:],axis=(-1))/np.sum(fileJPsiData['test_data'][:])
hmc_pt =np.sum(fileJPsiMC['test_mc'][:],axis=(-1))/np.sum(fileJPsiMC['test_mc'][:])
print(np.sum(fileJPsiMC['test_mc'][:]))
# hmc_pt_w =fileJPsiMC['Jpsi_distr_mcplus_means_w'][:]
# print(np.sum(hmc_pt_w[...,6],axis=-1)/np.sum(hmc_pt,axis=-1))
# hmc_ptgen =fileJPsiMC['Jpsi_distr_mcgen'][:]

pts = fileJPsiMC['edges_test_mc_0']
etas = fileJPsiMC['edges_test_mc_1']
ptsS = pts[1:]-pts[:-1]
# pts2 = fileJPsiMC['edges_Jpsi_distr_mc_1']
# pts1 = fileJPsiMC['edges_Jpsi_distr_mc_2']
# pts1C =  0.5*(pts1[:-1]+pts1[1:])
# pts2 = fileJPsiMC['edges_Jpsi_distr_mc_3']
# pts2C =  0.5*(pts2[:-1]+pts2[1:])

# print(hmc_pt.shape,pts1C.shape,pts2C.shape)

# interp = scipy.interpolate.interp2d(x=pts1C,y=pts2C,z=hmc_pt.T,kind="linear")
# print(interp(pts1C,pts2C).T.shape)
# print(hmc_pt/(interp(pts1C,pts2C).T))
# hf = h5py.File('interp.hdf5', 'w')
# hf.create_dataset('interp', data=(1./interp(pts1C,pts2C).T))
# hf.close()
# print(hmc_pt[...,0,0]/hmc_ptgen[...,0,0])
# hf = h5py.File('weights.hdf5', 'w')
# hf.create_dataset('weights', data=np.where(hmc_ptgen[...,0,0]>0,hmc_pt[...,0,0]/hmc_ptgen[...,0,0],0.))
# hf.close()

# pts1C = 0.5*(pts1[1:]+pts1[:-1])
# mean=np.average(pts1C, weights=hmc_pt[0,0,:,0,0])
# print(mean)
fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
hep.cms.text('work in progress', ax=ax1)
# ax1.set_title("scale", fontsize=18)
hep.histplot(hmc_pt,pts, ax=ax1, label = ["mc"])
hep.histplot(hdata_pt,pts, ax=ax1, label = ["data"])
plt.show()


# ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
# ax1.set_xlabel('$\eta$ of positive muon')

# hpull = ROOT.TH1D("mass_pull","mass_pull", 100, -5.,5.)
# hpull = array2hist(hmc_pt, hpull,np.sqrt(hmc_pt))
# hpull.Fit('gaus')
# hpull.Draw()
# import time
# time.sleep(1000)

# # fig, (ax1) = plt.subplots()
# # # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # hep.cms.text('work in progress', ax=ax1)
# # # ax1.set_title("scale", fontsize=18)
# # hep.hist2dplot(hmc_eta,pts,pts, ax=ax1, label = ["mc"])
# # ax1.legend(loc='upper right', frameon=True)
# # # ax1.set_ylabel('number of events')
# # # ax1.set_xlabel('$\eta$ of positive muon')

# # plt.tight_layout()
# # plt.show()

# # data over mc
# #eta1
# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_eta/np.sum(fileJPsiMC['Jpsi_distr_mc'][:]),etas, ax=ax1, label = ["mc"])
# hep.histplot(hdata_eta/np.sum(fileJPsiData['Jpsi_distr_data'][:]),etas, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$\eta$ of positive muon')

# plt.tight_layout()
# plt.savefig("FinalPlots/etadistrJ.png")
# plt.savefig("FinalPlots/etadistrJ.pdf")
# plt.clf()

#pt1
# fig, (ax1) = plt.subplots()
# # hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# hep.cms.text('work in progress', ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# # hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_distr_mc'][:]),pts, ax=ax1, label = ["mc"])
# hep.histplot(hmc_pt,pts, ax=ax1, label = ["mc"])
# # hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_distr_data'][:]),pts, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# # ax1.set_xlabel('$p_{T}$ of positive muon')
# plt.show()
# # plt.tight_layout()
# plt.savefig("FinalPlots/ptdistrJ.png")
# plt.savefig("FinalPlots/ptdistrJ.pdf")
# plt.clf()


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
# Jpts = np.linspace(0,50,101)
# hdata_pt = np.sum(fileJPsiData['Jpsi_rew_data'][:],axis=(1,2))
# hmc_pt = np.sum(fileJPsiMC['Jpsi_rew_mc'][:],axis=(1,2))
# fig, (ax1) = plt.subplots()
# hep.cms.label(loc=1,year=2016, data=True, ax=ax1)
# # ax1.set_title("scale", fontsize=18)
# hep.histplot(hmc_pt/np.sum(fileJPsiMC['Jpsi_rew_mc'][:]),Jpts, ax=ax1, label = ["mc"])
# hep.histplot(hdata_pt/np.sum(fileJPsiData['Jpsi_rew_data'][:]),Jpts, ax=ax1, label = ["data"], color='red')
# # ratio = np.where(hdata_pt/hmc_pt<50, hdata_pt/hmc_pt, 1)
# # hep.histplot(ratio,Jpts, ax=ax1, label = ["data"], color='red')
# ax1.legend(loc='upper right', frameon=True)
# # ax1.set_ylabel('number of events')
# ax1.set_xlabel('$J~p_T$')

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
# ax1.set_xlabel('$J~rapidity$')

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

# fileJPsiMC = h5py.File('JPsiInputData/JPsiMC.hdf5', mode='r')

# print(list(fileJPsiMC.keys()))
# massBins = fileJPsiMC['edges_Jpsi_gen_4'][:]
# massBinsC = 0.5*(massBins[1:]+massBins[:-1])
# hmass = np.sum(fileJPsiMC['Jpsi_gen'][:],axis=(0,1,3))
# hmasssm = np.sum(fileJPsiMC['Jpsi_mc'][:],axis=(0,1,3))
# bins=fileJPsiMC['edges_Jpsi_gen_2'][:]
# massesPt =[]
# massesPtrec =[]
# print(massBinsC)
# for i in range(len(bins)-1):
#     massPt = np.average(massBinsC, weights=hmass[i,:])
#     massesPt.append(massPt)
#     massPt = np.average(massBinsC, weights=hmasssm[i,:])
#     massesPtrec.append(massPt)

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

# fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [3, 1]})
# hep.cms.text('work in progress', ax=ax1)
# hep.histplot(np.array(massesPtrec),bins, ax=ax1,histtype = 'errorbar', label="smeared gen mass", marker="^",markersize=5)
# hep.histplot(np.array(massesPt),bins, ax=ax1, histtype = 'errorbar', label = "gen mass", marker="^",markersize=5)
# ax1.legend(loc='upper center', frameon=True)
# hep.histplot(np.array(massesPtrec)/np.array(massesPt),bins, ax=ax2,histtype = 'errorbar', marker="^",markersize=5)
# ax1.set_ylabel('gen mass')
# ax2.set_xlabel('$\eta$')
# # ax1.set_ylim(90.4,91)
# # ax2.set_ylim(1-0.0005,1+0.0005)
# plt.savefig('FinalPlots/Jmass.pdf')

# import pickle
# fileJ = open("calInputJMC_1etaBins_1ptBins.pkl","rb")
# pkgJ = pickle.load(fileJ)
# datasetJ = pkgJ['dataset']
# import numpy as np
# sigmas = np.linspace(0.02, 0.06, 9, dtype='float64')
# masses = np.linspace(2.9069,3.3069, 100+1, dtype='float64')
# import matplotlib.pyplot as plt
# import mplhep as hep
# import time

# h = ROOT.TH1D("sigma","sigma", 100,0.01, 0.07)
# # h = ROOT.TH1D("mass","mass", 100, 2.9069,3.3069)
# h = array2hist(np.sum(datasetJ[0,...],axis=0),h)
# print(np.sum(datasetJ[0,...],axis=0))
# # h.Fit('landau')
# h.Draw("colz")
# time.sleep(1000)
# fig, (ax1) = plt.subplots()
# hep.hist2dplot(datasetJ[0,...],masses,sigmas, ax=ax1, label = ["mc"])
# sigmasC = 0.5*(sigmas[1:]+sigmas[:-1])
# mean=np.average(sigmasC, weights=np.sum(datasetJ,axis=(0,1)))
# var = np.average((sigmasC - mean)**2, weights=np.sum(datasetJ,axis=(0,1)))
# print(mean,np.sqrt(var))
# bins = np.linspace(0, datasetJ[0,:,:].ravel("F").shape[0]+1,datasetJ[0,:,:].ravel("F").shape[0]+1)
# hep.histplot(datasetJ[0,:,:].ravel("F"),bins, ax=ax1, label = ["mc"])
# plt.show()
# kdiffs = np.linspace(-0.01, 0.01,101)
# fileJPsiMC = h5py.File('JPsiInputData/JPsiMC_mukin.hdf5', mode='r')
# print(list(fileJPsiMC.keys()))
# hmc_plus = fileJPsiMC['Jpsi_kplus'][:]/np.sum(fileJPsiMC['Jpsi_kplus'][:])
# hmc_minus = fileJPsiMC['Jpsi_kminus'][:]
# hmc = hmc_plus+hmc_minus
# h = ROOT.TH1D("h","h", 100, -0.02, 0.02)
# # h = array2hist(hmc[0,0,0,1],h,errors=np.sqrt(hmc[0,0,0,1]))
# h = array2hist(hmc[0,0,5,4],h,errors=np.sqrt(hmc[0,0,5,4]))
# f1 = ROOT.TF1("f1","gaus",-0.002,0.002)
# f2 = ROOT.TF1("f2","gaus", -0.02, 0.02)

# h.Fit(f1,"R0")
# h.Fit(f2,"R0+")

# f = ROOT.TF1("f","gaus(0)+gaus(3)", -0.02, 0.02)
# f.SetParameter(0,f1.GetParameter(0))
# f.SetParameter(1,f1.GetParameter(1))
# f.SetParameter(2,f1.GetParameter(2))
# f.SetParameter(3,f2.GetParameter(0))
# f.SetParameter(4,f2.GetParameter(1))
# f.SetParameter(5,f2.GetParameter(2))

# h.Fit(f,"R+")
# h.Draw("")
# time.sleep(1000)
# fig, (ax1) = plt.subplots()
# pts = np.array([1.1, 3.4, 4.4, 5.7, 7.4, 10.2, 13., 18.,25.])
# ptsS = pts[1:]-pts[:-1]
# hep.histplot(np.sum(hmc_plus,axis=0)/np.sum(hmc_plus,axis=1),pts, label = ["smeared pt"])
# print(np.sum(hmc_plus,axis=0)/np.sum(hmc_plus,axis=1))
# hep.histplot(,pts, label = ["reco pt"])

# plt.show()
# fig, (ax1) = plt.subplots()
# files=[]
# for i in range(1,5):
#     files.append("JPsiInputData/ZMC_{}.hdf5".format(i))

# for i,file in enumerate(files):
#     fileZMC = h5py.File(file, mode='r')
#     hmc_eta =fileZMC['Jpsi_distr_mc'][:]
#     pts = fileZMC['edges_Jpsi_distr_mc_0']

#     hep.cms.text('work in progress', ax=ax1)
#     hep.histplot(hmc_eta/np.sum(hmc_eta),pts, ax=ax1, label = ["bin_{}".format(i)])
# ax1.legend(loc='upper right', frameon=True)
# plt.show()
