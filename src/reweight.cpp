#include "interface/reweight.hpp"

RNode reweight::run(RNode d)
{

  auto applyReweight = [this](float pt1, float pt2) -> float {
    // auto bin = hrew->FindBin(pt1,pt2);
    int xbin = std::max(1, std::min(hrew->GetNbinsX(), hrew->GetXaxis()->FindFixBin(pt1)));
    int ybin = std::max(1, std::min(hrew->GetNbinsY(), hrew->GetYaxis()->FindFixBin(pt2)));
    float weight = hrew->GetBinContent(xbin,ybin);
    return weight;
  };

  auto d1 = d.Define("weightMC", applyReweight, {"Muplus_pt", "Muminus_pt"});

  return d1;
}
