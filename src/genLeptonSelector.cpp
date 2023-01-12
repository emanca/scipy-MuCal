#include "genLeptonSelector.hpp"
#include <algorithm>
#include <utility>

RNode genLeptonSelector::run(RNode d)
{
  //Logic taken from https://github.com/WMass/nanoAOD-tools/blob/master/python/postprocessing/wmass/genLepSelection.py
  //have to think of refinement
  auto getGenLeptonIdx = [](const RVec<int> &pdgId, const RVec<int> &status, const RVec<int> &genPartIdxMother, const RVec<int> &statusFlag, const RVec<float> &pt)
  {
    RVec<std::pair<int, float>> status746;

    for (unsigned int i = 0; i < pdgId.size(); i++)
    {
      if (status[i] == 1 && (statusFlag[i] & (1 << 0)) && std::abs(pdgId[i])==13) //is prompt muon after FSR
        status746.emplace_back(std::make_pair(i, pt[i]));
      }
    std::pair<int, int> prefsrlepidx(-1, -1);
    if (status746.size() == 2)
    {
      prefsrlepidx.first = status746[0].second > status746[1].second ? status746[0].first : status746[1].first;
      prefsrlepidx.second = status746[0].second > status746[1].second ? status746[1].first : status746[0].first;
    }
    //swap indices to save +ve lepton as first index
    std::pair<int, int> genLep;
    genLep.first = pdgId[prefsrlepidx.first] < 0 ? prefsrlepidx.second : prefsrlepidx.first;
    genLep.second = pdgId[prefsrlepidx.first] < 0 ? prefsrlepidx.first : prefsrlepidx.second;

    //return prefsrlepidx;
    return genLep;
  }; //function

  auto getVtype = [](const int idx1, const int idx2, const RVec<int> &pdg)
  {
    int vtype = -1;
    if (idx1 != -1 && idx2 == -1) //only 1 lepton
      vtype = pdg[idx1];
    else if (idx1 == -1 && idx2 == -1)
    { //https://github.com/WMass/nanoAOD-tools/blob/master/python/postprocessing/wmass/genLepSelection.py#L78-L86
      // int vcharge = pdg[idx1] % 2 ? -1 * pdg[idx1] / std::abs(pdg[idx1]) : -1 * pdg[idx2] / std::abs(pdg[idx2]);
      // vtype = vcharge * int((abs(pdg[idx1]) + abs(pdg[idx1])) / 2.);
      vtype = -99;
    }
    else if (idx1 != -1 && idx2 != -1)
    {
      vtype = pdg[idx1] % 2 ? -1 * pdg[idx1] : -1 * pdg[idx2];
    }
    std::cout << vtype<< std::endl;
    return vtype;
  };

  auto d1 = d.Define("SelectedGenPartIdxs", getGenLeptonIdx, {"GenPart_pdgId", "GenPart_status", "GenPart_genPartIdxMother", "GenPart_statusFlags", "GenPart_pt"})
                .Define("GenPart_postFSRLepIdx1", [](const std::pair<int, int> &gpIdxs)
                        { return gpIdxs.first; },
                        {"SelectedGenPartIdxs"})
                .Define("GenPart_postFSRLepIdx2", [](const std::pair<int, int> &gpIdxs)
                        { return gpIdxs.second; },
                        {"SelectedGenPartIdxs"})
                .Define("genVtype", getVtype, {"GenPart_postFSRLepIdx1", "GenPart_postFSRLepIdx2", "GenPart_pdgId"});

  return d1;
}
