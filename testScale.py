import ROOT
import time
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import root_numpy

fd = ROOT.TFile.Open('/scratchnvme/emanca/scipy-MuCal/calibrationJDATA.root')
fmc = ROOT.TFile.Open('/scratchnvme/emanca/scipy-MuCal/calibrationJMC.root')
with np.load('unbinnedfitglobalitercorscale.npz') as f:
      xs_j = f["xs"]
      xerrs_j = f["xerrs"]

hAmc = np.array(fmc.Get('A'))[1:-1]
hAd = np.array(fd.Get('A'))[1:-1]
Aerrmc = np.sqrt(np.array([fmc.Get('A').GetSumw2()[i] for i in range(fmc.Get('A').GetSumw2().GetSize())])[1:-1])
Aerrd = np.sqrt(np.array([fd.Get('A').GetSumw2()[i] for i in range(fd.Get('A').GetSumw2().GetSize())])[1:-1])

hemc = np.array(fmc.Get('e'))[1:-1]
hed = np.array(fd.Get('e'))[1:-1]
eerrmc = np.sqrt(np.array([fmc.Get('e').GetSumw2()[i] for i in range(fmc.Get('e').GetSumw2().GetSize())])[1:-1])
eerrd = np.sqrt(np.array([fd.Get('e').GetSumw2()[i] for i in range(fd.Get('e').GetSumw2().GetSize())])[1:-1])


corr_A = hAmc - xs_j[...,0]
corr_e = hAmc - xs_j[...,1]

bins = np.linspace(-2.4, 2.4, 48+1)

fig, ax1 = plt.subplots()
hep.histplot(hAmc-corr_A,bins,histtype = 'errorbar', yerr=Aerrmc, ax=ax1, label = ["mc"])
hep.histplot(hAd-corr_A,bins,histtype = 'errorbar', yerr=Aerrd, ax=ax1, label = ["data"], color='red')
ax1.legend(loc='upper right', frameon=True)
ax1.set_xlabel('$\eta$')
plt.tight_layout()
plt.show()
