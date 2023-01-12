#include <unordered_map>
#include <memory>
#include <algorithm>
#include <stdexcept>

class HelperRew {
 
public:
  
  HelperRew(TH3D *weights) {
    hweights_= weights;
  }
  
    
  double operator () (float var1, float var2, float var3) const {

    int xbin = std::max(1, std::min(hweights_->GetNbinsX(), hweights_->GetXaxis()->FindFixBin(var1)));
    int ybin = std::max(1, std::min(hweights_->GetNbinsY(), hweights_->GetYaxis()->FindFixBin(var2)));
    int zbin = std::max(1, std::min(hweights_->GetNbinsZ(), hweights_->GetZaxis()->FindFixBin(var3)));

    return hweights_->GetBinContent(xbin,ybin,zbin);
  }
  
  
private:
  TH3D* hweights_;
  
};
