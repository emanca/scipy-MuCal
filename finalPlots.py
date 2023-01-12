import h5py
import numpy as np
import ROOT
from root_numpy import array2hist, hist2array
import matplotlib.pyplot as plt
import mplhep as hep
from fittingFunctionsBinned import bin_ndarray
# matplotlib stuff
plt.style.use([hep.style.CMS])
ROOT.gROOT.SetBatch(True)

# parameters fit
bins = np.linspace(-2.4,2.4, 48+1)
binsC = (bins[:-1] + bins[1:]) / 2
fmc = ROOT.TFile.Open("calibrationJMC_aftersm.root")
fmc_layer = ROOT.TFile.Open("calibrationMC_layercorr_after.root")
fd = ROOT.TFile.Open("calibrationJDATA_aftersm.root")
fd_layer = ROOT.TFile.Open("calibrationDATA_layerCorr.root")
# unbinned fit
with np.load('unbinnedfitglobalitercorscale_ztwotracks.npz') as f:
      xs = f["xs"]
      xerrs = f["xerrs"]
with np.load('unbinnedfitglobalitercorscale.npz') as f:
      xs_j = f["xs"]
      xerrs_j = f["xerrs"]
with np.load('unbinnedfitglobalitercorscale_single.npz') as f:
      xs_single = f["xs"]
      xerrs_single = f["xerrs"]

# unbinned fit w layer corr
with np.load('mctruthresults_v202_jpsi_cor/unbinnedfitglobalitercorscale.npz') as f:
      xs_corr = f["xs"]
      xerrs_corr = f["xerrs"]
with np.load('mctruthresults_v202_single_cor/unbinnedfitglobalitercorscale.npz') as f:
      xs_single_corr = f["xs"]
      xerrs_single_corr = f["xerrs"]

hAmc = np.array(fmc.Get('A'))[1:-1]
hemc = np.array(fmc.Get('e'))[1:-1]
hMmc = np.array(fmc.Get('M'))[1:-1]
hamc = np.array(fmc.Get('a'))[1:-1]
hcmc = np.array(fmc.Get('c'))[1:-1]
hbmc = np.array(fmc.Get('b'))[1:-1]

Aerrmc = np.sqrt(np.array([fmc.Get('A').GetSumw2()[i] for i in range(fmc.Get('A').GetSumw2().GetSize())])[1:-1])
eerrmc = np.sqrt(np.array([fmc.Get('e').GetSumw2()[i] for i in range(fmc.Get('e').GetSumw2().GetSize())])[1:-1])
Merrmc = np.sqrt(np.array([fmc.Get('M').GetSumw2()[i] for i in range(fmc.Get('M').GetSumw2().GetSize())])[1:-1])
aerrmc = np.sqrt(np.array([fmc.Get('a').GetSumw2()[i] for i in range(fmc.Get('a').GetSumw2().GetSize())])[1:-1])
berrmc = np.sqrt(np.array([fmc.Get('b').GetSumw2()[i] for i in range(fmc.Get('b').GetSumw2().GetSize())])[1:-1])
cerrmc = np.sqrt(np.array([fmc.Get('c').GetSumw2()[i] for i in range(fmc.Get('c').GetSumw2().GetSize())])[1:-1])

hAmc_layer = np.array(fmc_layer.Get('A'))[1:-1]
hemc_layer = np.array(fmc_layer.Get('e'))[1:-1]
hMmc_layer = np.array(fmc_layer.Get('M'))[1:-1]
hamc_layer = np.array(fmc_layer.Get('a'))[1:-1]
hcmc_layer = np.array(fmc_layer.Get('c'))[1:-1]
hbmc_layer = np.array(fmc_layer.Get('b'))[1:-1]

Aerrmc_layer = np.sqrt(np.array([fmc_layer.Get('A').GetSumw2()[i] for i in range(fmc_layer.Get('A').GetSumw2().GetSize())])[1:-1])
eerrmc_layer = np.sqrt(np.array([fmc_layer.Get('e').GetSumw2()[i] for i in range(fmc_layer.Get('e').GetSumw2().GetSize())])[1:-1])
Merrmc_layer = np.sqrt(np.array([fmc_layer.Get('M').GetSumw2()[i] for i in range(fmc_layer.Get('M').GetSumw2().GetSize())])[1:-1])
aerrmc_layer = np.sqrt(np.array([fmc_layer.Get('a').GetSumw2()[i] for i in range(fmc_layer.Get('a').GetSumw2().GetSize())])[1:-1])
berrmc_layer = np.sqrt(np.array([fmc_layer.Get('b').GetSumw2()[i] for i in range(fmc_layer.Get('b').GetSumw2().GetSize())])[1:-1])
cerrmc_layer = np.sqrt(np.array([fmc_layer.Get('c').GetSumw2()[i] for i in range(fmc_layer.Get('c').GetSumw2().GetSize())])[1:-1])

hAd = np.array(fd.Get('A'))[1:-1]
hed = np.array(fd.Get('e'))[1:-1]
hMd = np.array(fd.Get('M'))[1:-1]
had = np.array(fd.Get('a'))[1:-1]
hcd = np.array(fd.Get('c'))[1:-1]
hbd = np.array(fd.Get('b'))[1:-1]

Aerrd = np.sqrt(np.array([fd.Get('A').GetSumw2()[i] for i in range(fd.Get('A').GetSumw2().GetSize())])[1:-1])
eerrd = np.sqrt(np.array([fd.Get('e').GetSumw2()[i] for i in range(fd.Get('e').GetSumw2().GetSize())])[1:-1])
Merrd = np.sqrt(np.array([fd.Get('M').GetSumw2()[i] for i in range(fd.Get('M').GetSumw2().GetSize())])[1:-1])
aerrd = np.sqrt(np.array([fd.Get('a').GetSumw2()[i] for i in range(fd.Get('a').GetSumw2().GetSize())])[1:-1])
berrd = np.sqrt(np.array([fd.Get('b').GetSumw2()[i] for i in range(fd.Get('b').GetSumw2().GetSize())])[1:-1])
cerrd = np.sqrt(np.array([fd.Get('c').GetSumw2()[i] for i in range(fd.Get('c').GetSumw2().GetSize())])[1:-1])

hAd_layer = np.array(fd_layer.Get('A'))[1:-1]
hed_layer = np.array(fd_layer.Get('e'))[1:-1]
hMd_layer = np.array(fd_layer.Get('M'))[1:-1]
had_layer = np.array(fd_layer.Get('a'))[1:-1]
hcd_layer = np.array(fd_layer.Get('c'))[1:-1]
hbd_layer = np.array(fd_layer.Get('b'))[1:-1]

Aerrd_layer = np.sqrt(np.array([fd_layer.Get('A').GetSumw2()[i] for i in range(fd_layer.Get('A').GetSumw2().GetSize())])[1:-1])
eerrd_layer = np.sqrt(np.array([fd_layer.Get('e').GetSumw2()[i] for i in range(fd_layer.Get('e').GetSumw2().GetSize())])[1:-1])
Merrd_layer = np.sqrt(np.array([fd_layer.Get('M').GetSumw2()[i] for i in range(fd_layer.Get('M').GetSumw2().GetSize())])[1:-1])
aerrd_layer = np.sqrt(np.array([fd_layer.Get('a').GetSumw2()[i] for i in range(fd_layer.Get('a').GetSumw2().GetSize())])[1:-1])
berrd_layer = np.sqrt(np.array([fd_layer.Get('b').GetSumw2()[i] for i in range(fd_layer.Get('b').GetSumw2().GetSize())])[1:-1])
cerrd_layer = np.sqrt(np.array([fd_layer.Get('c').GetSumw2()[i] for i in range(fd_layer.Get('c').GetSumw2().GetSize())])[1:-1])

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
ax1.set_title("A", fontsize=18)
hep.histplot(hAmc,bins,histtype = 'errorbar', yerr=Aerrmc, ax=ax1, label = ["$J/\psi$ mc"])
# hep.histplot(hAmc_layer,bins,histtype = 'errorbar', yerr=Aerrmc_layer, ax=ax1, label = ["mc w layer corr"],color='orange')
hep.histplot(hAd,bins,histtype = 'errorbar', yerr=Aerrd, ax=ax1, label = ["$J/\psi$ data"], color='red')
# hep.histplot(hAd_layer,bins,histtype = 'errorbar', yerr=Aerrd_layer, ax=ax1, label = ["data w layer corr"], color='green')
# hep.histplot(-xs_j[...,0],bins,histtype = 'errorbar', yerr=xerrs[...,0], ax=ax1, label = ["mc truth"], color='green')
# hep.histplot(-xs_single[...,0],bins,histtype = 'errorbar', yerr=xerrs[...,0], ax=ax1, label = ["mc truth (single muon)"], color='magenta')
# hep.histplot(-xs_corr[...,0],bins,histtype = 'errorbar', yerr=xerrs_corr[...,0], ax=ax1, label = ["mc truth (jpsi) w layer corr"], color='yellow')
# hep.histplot(-xs_single_corr[...,0],bins,histtype = 'errorbar', yerr=xerrs_corr[...,0], ax=ax1, label = ["mc truth (single muon) w layer corr"], color='grey')
# ax1.fill_between(binsC, hAmc-Aerrmc, hAmc+Aerrmc, color=['blue'], alpha=0.4, label = "mc after corrections")
# ax1.fill_between(binsC, hAd-Aerrd, hAd+Aerrd, color=['red'], alpha=0.4, label = "data after corrections")
ax1.legend(loc='upper right', frameon=True)
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/A.png')
plt.savefig('FinalPlots/A.pdf')
plt.cla()

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
ax1.set_title("e", fontsize=18)
hep.histplot(hemc,bins,histtype = 'errorbar', yerr=eerrmc, ax=ax1, label = ["$J/\psi$ mc"])
# hep.histplot(hemc_layer,bins,histtype = 'errorbar', yerr=eerrmc_layer, ax=ax1, label = ["mc w layer corr"],color='orange')
hep.histplot(hed,bins,histtype = 'errorbar', yerr=eerrd, ax=ax1, label = ["$J/\psi$ data"], color='red')
# hep.histplot(hed_layer,bins,histtype = 'errorbar', yerr=eerrd_layer, ax=ax1, label = ["data w layer corr"], color='green')
# hep.histplot(-xs[...,1],bins,histtype = 'errorbar', yerr=xerrs[...,1], ax=ax1, label = ["mc truth (z)"], color='green')
# hep.histplot(-xs_single[...,1],bins,histtype = 'errorbar', yerr=xerrs[...,1], ax=ax1, label = ["mc truth (single muon)"], color='magenta')
# hep.histplot(-xs_corr[...,1],bins,histtype = 'errorbar', yerr=xerrs_corr[...,1], ax=ax1, label = ["mc truth (jpsi) w layer corr"], color='yellow')
# hep.histplot(-xs_single_corr[...,1],bins,histtype = 'errorbar', yerr=xerrs_corr[...,1], ax=ax1, label = ["mc truth (single muon) w layer corr"], color='grey')
# ax1.fill_between(binsC, hemc-eerrmc, hemc+eerrmc, color=['blue'], alpha=0.4, label = "mc after corrections")
# ax1.fill_between(binsC, hed-eerrd, hed+eerrd, color=['red'], alpha=0.4, label = "data after corrections")
# ax1.set_ylim([np.min(hmc)-0.0005, np.max(hmc)+0.0005])
ax1.legend(loc='upper right', frameon=True)
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/e.png')
plt.savefig('FinalPlots/e.pdf')
plt.cla()

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
ax1.set_title("M", fontsize=18)
hep.histplot(hMmc,bins,histtype = 'errorbar', yerr=Merrmc, ax=ax1, label = ["$J/\psi$ mc"])
# hep.histplot(hMmc_layer,bins,histtype = 'errorbar', yerr=Merrmc_layer, ax=ax1, label = ["mc w layer corr"],color='orange')
hep.histplot(hMd,bins,histtype = 'errorbar', yerr=Merrd, ax=ax1, label = ["$J/\psi$ data"], color='red')
# hep.histplot(hMd_layer,bins,histtype = 'errorbar', yerr=Merrd_layer, ax=ax1, label = ["data w layer corr"], color='green')
# hep.histplot(-xs[...,2],bins,histtype = 'errorbar', yerr=xerrs[...,2], ax=ax1, label = ["mc truth (z)"], color='green')
# hep.histplot(-xs_single[...,2],bins,histtype = 'errorbar', yerr=xerrs[...,2], ax=ax1, label = ["mc truth (single muon)"], color='magenta')
# hep.histplot(-xs_corr[...,2],bins,histtype = 'errorbar', yerr=xerrs_corr[...,2], ax=ax1, label = ["mc truth (jpsi) w layer corr"], color='yellow')
# hep.histplot(-xs_single_corr[...,2],bins,histtype = 'errorbar', yerr=xerrs_corr[...,2], ax=ax1, label = ["mc truth (single muon) w layer corr"], color='grey')
# ax1.fill_between(binsC, hMmc-Merrmc, hMmc+Merrmc, color=['blue'], alpha=0.4, label = "mc after corrections")
# ax1.fill_between(binsC, hMd-Merrd, hMd+Merrd, color=['red'], alpha=0.4, label = "data after corrections")
ax1.set_ylim(-1e-5,9e-5)
ax1.legend(loc='upper right', frameon=True)
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/M.png')
plt.savefig('FinalPlots/M.pdf')
plt.cla()

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
# ax1.set_title("a", fontsize=18)
hep.histplot(hamc,bins,histtype = 'errorbar', yerr=aerrmc, ax=ax1, label = ["$J/\psi$ mc"])
hep.histplot(had,bins,histtype = 'errorbar', yerr=aerrd, ax=ax1, label = ["$J/\psi$ + $Z$ data"], color='red')
# hep.histplot(xs[...,3],bins,histtype = 'errorbar', yerr=xerrs[...,3], ax=ax1, label = ["mc truth (z)"], color='green')
# hep.histplot(xs_j[...,3],bins,histtype = 'errorbar', yerr=xerrs_j[...,3], ax=ax1, label = ["single muon"], color='orange')
# hep.histplot(xs_single[...,3],bins,histtype = 'errorbar', yerr=xerrs[...,3], ax=ax1, label = ["mc truth"], color='magenta')
# ax1.set_ylim([np.min(hmc)-0.0005, np.max(hmc)+0.0005])
ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
ax1.set_ylim(0,4e-4)
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/a.png')
plt.savefig('FinalPlots/a.pdf')
plt.cla()

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
# ax1.set_title("b", fontsize=18)
hep.histplot(hbmc,bins,histtype = 'errorbar', yerr=berrmc, ax=ax1, label = ["$J/\psi$ mc"])
hep.histplot(hbd,bins,histtype = 'errorbar', yerr=berrd, ax=ax1, label = ["$J/\psi$ + $Z$ data"], color='red')
# hep.histplot(xs[...,5],bins,histtype = 'errorbar', yerr=xerrs[...,5], ax=ax1, label = ["mc truth (z)"], color='green')
# hep.histplot(xs_j[...,5],bins,histtype = 'errorbar', yerr=xerrs_j[...,5], ax=ax1, label = ["single muon"], color='orange')
# hep.histplot(xs_single[...,5],bins,histtype = 'errorbar', yerr=xerrs[...,5], ax=ax1, label = ["mc truth"], color='magenta')
# ax1.set_ylim([np.min(hmc)-0.0005, np.max(hmc)+0.0005])
ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/b.png')
plt.savefig('FinalPlots/b.pdf')

plt.cla()

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
# ax1.set_title("c", fontsize=18)
hep.histplot(hcmc,bins,histtype = 'errorbar', yerr=cerrmc, ax=ax1, label = ["$J/\psi$ mc"])
hep.histplot(hcd,bins,histtype = 'errorbar', yerr=cerrd, ax=ax1, label = ["$J/\psi$ + $Z$ data"], color='red')
# hep.histplot(xs[...,4],bins,histtype = 'errorbar', yerr=xerrs[...,4], ax=ax1, label = ["mc truth (z)"], color='green')
# hep.histplot(xs_j[...,4],bins,histtype = 'errorbar', yerr=xerrs_j[...,4], ax=ax1, label = ["single muon"], color='orange')
# hep.histplot(xs_single[...,4],bins,histtype = 'errorbar', yerr=xerrs[...,4], ax=ax1, label = ["mc truth"], color='magenta')
# ax1.set_ylim([0., 0.2e-6])
ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/c.png')
plt.savefig('FinalPlots/c.pdf')

plt.cla()

fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
ax1.set_title("d", fontsize=18)
# hep.histplot(xs[...,6],bins,histtype = 'errorbar', yerr=xerrs[...,6], ax=ax1, label = ["mc truth (z)"], color='green')
hep.histplot(xs_j[...,6],bins,histtype = 'errorbar', yerr=xerrs_j[...,6], ax=ax1, label = ["single muon"], color='green')
# hep.histplot(xs_single[...,6],bins,histtype = 'errorbar', yerr=xerrs_single[...,6], ax=ax1, label = ["mc truth (single)"], color='magenta')
ax1.set_ylim([0, 30])
ax1.legend(loc='upper right', frameon=True)
# ax1.set_ylabel('number of events')
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.savefig('FinalPlots/d.png')
plt.savefig('FinalPlots/d.pdf')
plt.cla()

# hAmctruth = ROOT.TH1D('A','',48,-2.4,2.4)
# array2hist(-1*xs[...,0],hAmctruth, errors=xerrs[...,0])
# hemctruth = ROOT.TH1D('e','',48,-2.4,2.4)
# array2hist(-1*xs[...,1],hemctruth, errors=xerrs[...,1])
# hMmctruth = ROOT.TH1D('M','',48,-2.4,2.4)
# array2hist(-1*xs[...,2],hMmctruth, errors=xerrs[...,2])

# fmctruth = ROOT.TFile('calibrationMCtruth.root','recreate')
# fmctruth.cd()
# hAmctruth.Write()
# hemctruth.Write()
# hMmctruth.Write()

# compute the combinations of the bins of the grid

filesSigma = [h5py.File('fitsZMC_corr.hdf5', mode='r'), h5py.File('fitsZDATA_corr.hdf5', mode='r')]

# closure of sigma 

hmc = filesSigma[0]['sigma'][:]
hmcErr = filesSigma[0]['sigmaErr'][:]
hdata = filesSigma[1]['sigma'][:]
hdataErr = filesSigma[1]['sigmaErr'][:]

good_idx = tuple(np.array(filesSigma[0]['good_idx'][:], dtype="int"))
good_idxd = tuple(np.array(filesSigma[1]['good_idx'][:], dtype="int"))
etas = filesSigma[0]['etas'][:]
etasS = etas[1:]-etas[:-1]
etasC = (bins[:-1] + bins[1:]) / 2

# pullmc,_ = np.histogram((1-hmc)/hmcErr, bins=100, range=(-5,5))
# pulldata,_ = np.histogram((1-hdata)/hdataErr, bins=100, range=(-5,5))
# hpullmc = ROOT.TH1D("pull_mc_all","", 100, -5,5)
# hpulld = ROOT.TH1D("pull_d_all","", 100, -5,5)
# hpullmc = array2hist(pullmc,hpullmc)
# hpulld = array2hist(pulldata,hpulld)
# canvmc = ROOT.TCanvas('cpull_mc_all')
# canvd = ROOT.TCanvas('cpull_data_all')
# canvmc.cd()
# hpullmc.Fit('gaus')
# hpullmc.Draw()
# canvmc.SaveAs('FinalPlots/{}.png'.format(hpullmc.GetName()))

# separate plots per eta bins
full = np.zeros((48,48,4,4),dtype='float64')
full[good_idx] = hmc
sigmamcPatched = full
print("sigmaZmc",np.nonzero(sigmamcPatched[:,:,0,:]))
full = np.zeros((48,48,4,4),dtype='float64')
full[good_idx] = hmcErr
sigmamcErrorPatched = full

full = np.zeros((48,48,4,4),dtype='float64')
full[good_idxd] = hdata
sigmadataPatched = full
full = np.zeros((48,48,4,4),dtype='float64')
full[good_idxd] = hdataErr
sigmadataErrorPatched = full

sigmasEtamc = []
sigmasErrEtamc = []
sigmasEtadata = []
sigmasErrEtadata = []
for ieta1 in range(48):
      sigmasEtadata.append(np.average(sigmadataPatched[ieta1,...][(sigmadataPatched[ieta1,...]!=0)&(sigmadataPatched[ieta1,...]>0.001)],weights=np.reciprocal(np.square(sigmadataErrorPatched[ieta1][(sigmadataPatched[ieta1,...]!=0)&(sigmadataPatched[ieta1,...]>0.001)]))))
      sigmasErrEtadata.append(np.sqrt(1/np.sum(np.reciprocal(np.square(sigmadataErrorPatched[ieta1][(sigmadataPatched[ieta1,...]!=0)&(sigmadataPatched[ieta1,...]>0.001)])))))
      sigmasEtamc.append(np.average(sigmamcPatched[ieta1,...][sigmamcPatched[ieta1,...]!=0],weights=np.reciprocal(np.square(sigmamcErrorPatched[ieta1][sigmamcPatched[ieta1,...]!=0]))))
      sigmasErrEtamc.append(np.sqrt(1/np.sum(np.reciprocal(np.square(sigmamcErrorPatched[ieta1][sigmamcPatched[ieta1,...]!=0])))))
sigmasEtamc_red = bin_ndarray(np.array(sigmasEtamc),(12,),'mean')
sigmasEtadata_red = bin_ndarray(np.array(sigmasEtadata),(12,),'mean')
sigmasErrEtamc_red = np.sqrt(1./bin_ndarray(np.reciprocal(np.square(np.array(sigmasErrEtamc))),(12,),'sum'))
sigmasErrEtadata_red = np.sqrt(1./bin_ndarray(np.reciprocal(np.square(np.array(sigmasErrEtadata))),(12,),'sum'))

etas_red = np.linspace(-2.4, 2.4, 13)
fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [3, 1]})
hep.cms.text('work in progress', loc=1, ax=ax1)
hep.histplot(np.array(sigmasEtamc_red),etas_red,yerr=sigmasErrEtamc_red, histtype = 'errorbar', ax=ax1, label="$Z$ mc")
# hep.histplot(np.array(sigmasEtamc_smeared),etas,histtype = 'errorbar', yerr=np.array(sigmasErrEtamc_smeared), ax=ax1, label="$J/\psi$ mc after smearing")
hep.histplot(np.array(sigmasEtadata_red),etas_red,yerr=sigmasErrEtadata_red,histtype = 'errorbar', ax=ax1,color="red", label="$Z$ data")
hep.histplot(np.array(sigmasEtadata_red)/np.array(sigmasEtamc_red),etas_red,histtype = 'errorbar', color = "k", stack = False, ax=ax2)
# hep.histplot(np.array(sigmasEtadata)/np.array(sigmasEtamc_smeared),etas,histtype = 'errorbar', color = "b", stack = False, ax=ax2)
ax2.set_ylabel('data/mc')
ax2.set_ylim(0.8,1.2)
ax2.set_xlabel('$\eta$')
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('FinalPlots/sigma_clos_eta.png')
plt.savefig('FinalPlots/sigma_clos_eta.pdf')


sigmasEtamc = []
sigmasErrEtamc = []
sigmasEtadata = []
sigmasErrEtadata = []
for ipt1 in range(4):
      try:
            sigmasEtadata.append(np.average(sigmadataPatched[:,:,ipt1,...][(sigmadataPatched[:,:,ipt1,...]!=0) & (sigmadataPatched[:,:,ipt1,:]>0.001)],weights=np.reciprocal(np.square(sigmadataErrorPatched[:,:,ipt1][(sigmadataPatched[:,:,ipt1,...]!=0) & (sigmadataPatched[:,:,ipt1,:]>0.001)]))))
      except ZeroDivisionError:
            sigmasEtadata.append(0)
      try:
            sigmasErrEtadata.append(np.sqrt(1/np.sum(np.reciprocal(np.square(sigmadataErrorPatched[:,:,ipt1][(sigmadataPatched[:,:,ipt1,...]!=0) & (sigmadataPatched[:,:,ipt1,:]>0.001)])))))
      except ZeroDivisionError:
            sigmasErrEtadata.append(0)
      try:
            sigmasEtamc.append(np.average(sigmamcPatched[:,:,ipt1,...][sigmamcPatched[:,:,ipt1,...]!=0],weights=np.reciprocal(np.square(sigmamcErrorPatched[:,:,ipt1][sigmamcPatched[:,:,ipt1,...]!=0]))))
      except ZeroDivisionError:
            sigmasEtamc.append(0)
      try:
            sigmasErrEtamc.append(np.sqrt(1/np.sum(np.reciprocal(np.square(sigmamcErrorPatched[:,:,ipt1][sigmamcPatched[:,:,ipt1,...]!=0])))))
      except ZeroDivisionError:
            sigmasErrEtamc.append(0)

# pts = np.array([2.6, 3.4, 4.4, 5.7, 7.4, 10.2, 13., 18.,25.])
pts = np.array([25.,30., 40., 50., 60.])
fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [3, 1]})
hep.cms.text('work in progress', loc=1, ax=ax1)
hep.histplot(np.array(sigmasEtamc),pts,histtype = 'errorbar', yerr=np.array(sigmasErrEtamc), ax=ax1, label="$Z$ mc")
# hep.histplot(np.array(sigmasEtamc_smeared),pts,histtype = 'errorbar', yerr=np.array(sigmasErrEtamc_smeared), ax=ax1, label="$J/\psi$ mc after smearing")
hep.histplot(np.array(sigmasEtadata),pts,histtype = 'errorbar', yerr=np.array(sigmasErrEtadata), ax=ax1,color="red", label="$Z$ data")
# ax1.set_ylim(0.01,0.012)
hep.histplot(np.array(sigmasEtadata)/np.array(sigmasEtamc),pts,histtype = 'errorbar', color = "k", stack = False, ax=ax2)
print(np.array(sigmasEtadata)/np.array(sigmasEtamc))
# hep.histplot(np.array(sigmasEtadata)/np.array(sigmasEtamc_smeared),pts,histtype = 'errorbar', color = "b", stack = False, ax=ax2)
ax2.set_ylabel('data/mc')
ax2.set_xlabel('$p_T$ (GeV)')
ax2.set_ylim(0.8,1.2)
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('FinalPlots/sigma_clos_pt.png')
plt.savefig('FinalPlots/sigma_clos_pt.pdf')


# closure of scale
filesJ = [h5py.File('fitsJMC_corr.hdf5', mode='r'), h5py.File('fitsJDATA_corr.hdf5', mode='r')]
filesZ = [h5py.File('fitsZMC_corr.hdf5', mode='r'), h5py.File('fitsZDATA_corr.hdf5', mode='r')]

hZmc = filesZ[0]['scale'][:]
hZmcErr = filesZ[0]['scaleErr'][:]
hZd = filesZ[1]['scale'][:]
hZdErr = filesZ[1]['scaleErr'][:]
good_idxZ = tuple(np.array(filesZ[0]['good_idx'][:], dtype="int"))
good_idxZd = tuple(np.array(filesZ[1]['good_idx'][:], dtype="int"))

hJmc = filesJ[0]['scale'][:]
hJmcErr = filesJ[0]['scaleErr'][:]
hJd = filesJ[1]['scale'][:]
hJdErr = filesJ[1]['scaleErr'][:]
good_idxJ = tuple(np.array(filesJ[0]['good_idx'][:], dtype="int"))
good_idxJd = tuple(np.array(filesJ[1]['good_idx'][:], dtype="int"))

# J
full = np.zeros((48,48,8,8),dtype='float64')
full[good_idxJ] = hJmc
scaleJmcPatched = full
full = np.zeros((48,48,8,8),dtype='float64')
full[good_idxJ] = hJmcErr
scaleJmcErrorPatched = full

full = np.zeros((48,48,8,8),dtype='float64')
full[good_idxJd] = hJd
scaleJdPatched = full
full = np.zeros((48,48,8,8),dtype='float64')
full[good_idxJd] = hJdErr
scaleJdErrorPatched = full

# Z
full = np.zeros((48,48,4,4),dtype='float64')
full[good_idxZ] = hZmc
scaleZmcPatched = full
full = np.zeros((48,48,4,4),dtype='float64')
full[good_idxZ] = hZmcErr
scaleZmcErrorPatched = full

full = np.zeros((48,48,4,4),dtype='float64')
full[good_idxZd] = hZd
scaleZdPatched = full
full = np.zeros((48,48,4,4),dtype='float64')
full[good_idxZd] = hZdErr
scaleZdErrorPatched = full

scalesEtaZmc=[]
scalesEtaZdata=[]
scalesErrEtaZmc=[]
scalesErrEtaZdata=[]

scalesEtaJmc=[]
scalesEtaJdata=[]
scalesErrEtaJmc=[]
scalesErrEtaJdata=[]

for ieta1 in range(48):

      scalesEtaJmc.append(np.average(scaleJmcPatched[:,ieta1,...][scaleJmcPatched[:,ieta1,...]!=0],weights=np.reciprocal(np.square(scaleJmcErrorPatched[:,ieta1,...][scaleJmcPatched[:,ieta1,...]!=0]))))
      scalesErrEtaJmc.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleJmcErrorPatched[:,ieta1,...][scaleJmcPatched[:,ieta1,...]!=0])))))
      scalesEtaJdata.append(np.average(scaleJdPatched[:,ieta1,...][scaleJdPatched[:,ieta1,...]!=0],weights=np.reciprocal(np.square(scaleJdErrorPatched[:,ieta1,...][scaleJdPatched[:,ieta1,...]!=0]))))
      scalesErrEtaJdata.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleJdErrorPatched[:,ieta1,...][scaleJdPatched[:,ieta1,...]!=0])))))

      scalesEtaZmc.append(np.average(scaleZmcPatched[:,ieta1,...][scaleZmcPatched[:,ieta1,...]!=0],weights=np.reciprocal(np.square(scaleZmcErrorPatched[:,ieta1,...][scaleZmcPatched[:,ieta1,...]!=0]))))
      scalesErrEtaZmc.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleZmcErrorPatched[:,ieta1,...][scaleZmcPatched[:,ieta1,...]!=0])))))
      scalesEtaZdata.append(np.average(scaleZdPatched[:,ieta1,...][scaleZdPatched[:,ieta1,...]!=0],weights=np.reciprocal(np.square(scaleZdErrorPatched[:,ieta1,...][scaleZdPatched[:,ieta1,...]!=0]))))
      scalesErrEtaZdata.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleZdErrorPatched[:,ieta1,...][scaleZdPatched[ieta1,...]!=0])))))

etas = np.linspace(-2.4, 2.4, 13, dtype='float64')
etasC = (etas[:-1] + etas[1:]) / 2

etasZ = np.linspace(-2.4, 2.4, 13, dtype='float64')
etasZC = (etasZ[:-1] + etasZ[1:]) / 2

scalesEtaJmc_red = bin_ndarray(np.array(scalesEtaJmc),(12,),'mean')
scalesEtaJdata_red = bin_ndarray(np.array(scalesEtaJdata),(12,),'mean')
scalesEtaZmc_red = bin_ndarray(np.array(scalesEtaZmc),(12,),'mean')
scalesEtaZdata_red = bin_ndarray(np.array(scalesEtaZdata),(12,),'mean')

scalesErrEtaJmc_red = np.sqrt(1./bin_ndarray(np.reciprocal(np.square(np.array(scalesErrEtaJmc))),(12,),'sum'))
scalesErrEtaJdata_red = np.sqrt(1./bin_ndarray(np.reciprocal(np.square(np.array(scalesErrEtaJdata))),(12,),'sum'))
scalesErrEtaZmc_red = np.sqrt(1./bin_ndarray(np.reciprocal(np.square(np.array(scalesErrEtaZmc))),(12,),'sum'))
scalesErrEtaZdata_red = np.sqrt(1./bin_ndarray(np.reciprocal(np.square(np.array(scalesErrEtaZdata))),(12,),'sum'))

# fig, (ax1, ax2) = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [3, 1]})
fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
# hep.histplot(np.array(scalesEtaZmc),etas,histtype = 'errorbar', yerr=np.array(scalesErrEtaZmc), ax=ax1, label="$Z$ mc", color="green")
# hep.histplot(np.array(scalesEtaZdata),etas,histtype = 'errorbar', yerr=np.array(scalesErrEtaZdata), ax=ax1, label="$Z$ data", color="red")
# ax1.set_ylim([1-0.001, 1+0.001])
ax1.errorbar(etasC, np.array(scalesEtaJdata_red)-np.array(scalesEtaJmc_red),yerr=np.sqrt(np.square(scalesErrEtaJmc_red)+np.square(scalesErrEtaJdata_red)), c='b', marker="v", label = '$J/\psi$',fmt='bv')
ax1.errorbar(etasZC,np.array(scalesEtaZdata_red)-np.array(scalesEtaZmc_red),yerr=np.sqrt(np.square(scalesErrEtaZmc_red)+np.square(scalesErrEtaZdata_red)),c='r', marker="o", label = '$Z$',fmt='or')
ax1.set_ylim([-0.0008, 0.0008])
ax1.plot(etasC, 2*(10**-4)*np.ones_like(etasC), c='grey', linestyle='dashed')
ax1.plot(etasC, -2*(10**-4)*np.ones_like(etasC), c='grey', linestyle='dashed')
# ax2.fill_between(etasC, -0.0002, 0.0002, color="green", alpha=0.2)
# ax2.text(-2.0, 1.e-4, '{}'.format(1e-4), style='italic', fontsize=12)
# ax2.text(-2.0, -1.7e-4, '{}'.format(-1e-4), style='italic', fontsize=12)
ax1.set_ylabel('data-mc')
ax1.set_xlabel('$\eta$')
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('FinalPlots/clos_eta.png')
plt.savefig('FinalPlots/clos_eta.pdf')


scalesZmc=[]
scalesZdata=[]
scalesErrZmc=[]
scalesErrZdata=[]

scalesJmc=[]
scalesJdata=[]
scalesErrJmc=[]
scalesErrJdata=[]

for ipt1 in range(4):
      try:
            scalesZdata.append(np.average(scaleZdPatched[:,:,:,ipt1,...][scaleZdPatched[:,:,:,ipt1,...]!=0],weights=np.reciprocal(np.square(scaleZdErrorPatched[:,:,:,ipt1][scaleZdPatched[:,:,:,ipt1,...]!=0]))))
      except ZeroDivisionError:
            scalesZdata.append(0)
      try:
            scalesErrZdata.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleZdErrorPatched[:,:,:,ipt1][scaleZdPatched[:,:,:,ipt1,...]!=0])))))
      except ZeroDivisionError:
            scalesErrZdata.append(0)
      try:
            scalesZmc.append(np.average(scaleZmcPatched[:,:,:,ipt1,...][scaleZmcPatched[:,:,:,ipt1,...]!=0],weights=np.reciprocal(np.square(scaleZmcErrorPatched[:,:,:,ipt1][scaleZmcPatched[:,:,:,ipt1,...]!=0]))))
      except ZeroDivisionError:
            scalesZmc.append(0)
      try:
            scalesErrZmc.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleZmcErrorPatched[:,:,:,ipt1][scaleZmcPatched[:,:,:,ipt1,...]!=0])))))
      except ZeroDivisionError:
            scalesErrZmc.append(0)


for ipt1 in range(8):
      try:
            scalesJdata.append(np.average(scaleJdPatched[:,:,:,ipt1,...][scaleJdPatched[:,:,:,ipt1,...]!=0],weights=np.reciprocal(np.square(scaleJdErrorPatched[:,:,:,ipt1][scaleJdPatched[:,:,:,ipt1,...]!=0]))))
      except ZeroDivisionError:
            scalesJdata.append(0)
      try:
            scalesErrJdata.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleJdErrorPatched[:,:,:,ipt1][scaleJdPatched[:,:,:,ipt1,...]!=0])))))
      except ZeroDivisionError:
            scalesErrJdata.append(0)
      try:
            scalesJmc.append(np.average(scaleJmcPatched[:,:,:,ipt1,...][scaleJmcPatched[:,:,:,ipt1,...]!=0],weights=np.reciprocal(np.square(scaleJmcErrorPatched[:,:,:,ipt1][scaleJmcPatched[:,:,:,ipt1,...]!=0]))))
      except ZeroDivisionError:
            scalesJmc.append(0)
      try:
            scalesErrJmc.append(np.sqrt(1/np.sum(np.reciprocal(np.square(scaleJmcErrorPatched[:,:,:,ipt1][scaleJmcPatched[:,:,:,ipt1,...]!=0])))))
      except ZeroDivisionError:
            scalesErrJmc.append(0)
ptsJ = np.array([2.6, 3.4, 4.4, 5.7, 7.4, 10.2, 13., 18.])
# ptsZ = np.array([25.,30., 40., 50., 60.])
ptsZ = np.array([25.,38, 44, 48.7, 60.])
pts=np.concatenate((ptsJ,ptsZ))
closureZ = np.array(scalesZdata)-np.array(scalesZmc)
closureJ = np.array(scalesJdata)-np.array(scalesJmc)
closure=np.concatenate((closureJ,closureZ))
closureZErr = np.sqrt(np.square(np.array(scalesErrZdata))+np.square(np.array(scalesErrZmc)))
closureJErr = np.sqrt(np.square(np.array(scalesErrJdata))+np.square(np.array(scalesErrJmc)))
closureErr=np.concatenate((closureJErr,closureZErr))

ptsC = (pts[:-1] + pts[1:]) / 2
fig, ax1 = plt.subplots()
hep.cms.text('work in progress', loc=1, ax=ax1)
ax1.errorbar(ptsC[:8], closure[:8],yerr=closureErr[:8], c='b', marker="v", label = '$J/\psi$',fmt='bv')
ax1.errorbar(ptsC[8:],closure[8:],yerr=closureErr[8:],c='r', marker="o", label = '$Z$',fmt='or')
# ax1.errorbar(ptsC[8:],scalesZmc, marker="o", label = '$Z mc$',fmt='o')
# ax1.errorbar(ptsC[8:],scalesZdata, marker="o", label = '$Z data$',fmt='o')
# hep.histplot(closure,pts,histtype = 'errorbar', color = "g", stack = False, ax=ax1, label = '$Z$')
# hep.histplot(np.array(scalesJdata)-np.array(scalesJmc),pts,histtype = 'errorbar', color = "b", stack = False, ax=ax1, label = '$J/\psi$')
ax1.set_ylabel('data-mc')
ax1.set_xlabel('$p_T$ (GeV)')
ax1.set_ylim(-0.0005,0.0005)
ax1.plot(ptsC, 2*(10**-4)*np.ones_like(ptsC),c='grey', linestyle='dashed')
ax1.plot(ptsC, -2*(10**-4)*np.ones_like(ptsC),c='grey', linestyle='dashed')
ax1.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('FinalPlots/clos_pt.png')
plt.savefig('FinalPlots/clos_pt.pdf')


av_jpsi = np.average(hJmc,weights=np.reciprocal(np.square(hJmcErr)))-np.average(hJd,weights=np.reciprocal(np.square(hJdErr)))
av_jpsi_err = np.sqrt(1/np.sum(np.reciprocal(np.square(hJmcErr)))+1/np.sum(np.reciprocal(np.square(hJdErr))))

av_z = np.average(hZmc,weights=np.reciprocal(np.square(hZmcErr)))-np.average(hZd,weights=np.reciprocal(np.square(hZdErr)))
av_z_err = np.sqrt(1/np.sum(np.reciprocal(np.square(hZmcErr)))+1/np.sum(np.reciprocal(np.square(hZdErr))))

print("JPsi:",av_jpsi, "+/-", av_jpsi_err)
print("Z:",av_z, "+/-", av_z_err)

# bins = files[0]['etas'][:]
# binsC = (bins[:-1] + bins[1:]) / 2

# hcov = fd.Get('covariance_matrix')
# cov = hist2array(hcov)[:,:]

# forms = []
# form_errs = []
# for i in range(48):
#       redcov = cov[i*6:i*6+3,i*6:i*6+3]

#       form = hAd[i] + hed[i]/5. + hMd[i] *5.
#       jac = np.array([1.,-1./5,5.])
#       form_err = np.sqrt(np.matmul(jac,np.matmul(redcov,jac.T)))
#       forms.append(form)
#       form_errs.append(form_err)
#       print(i,redcov)

# forms = np.array(forms)
# form_errs = np.array(form_errs)

# fig, ax1 = plt.subplots()
# hep.histplot(forms,bins)
# ax1.fill_between(binsC, forms-form_errs, forms+form_errs, color="magenta", alpha=0.2)
# ax1.set_xlabel('$\eta$')
# plt.tight_layout()
# plt.savefig('FinalPlots/prop40.png')
