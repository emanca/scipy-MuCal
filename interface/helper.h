#include <unordered_map>
#include <memory>
#include <algorithm>
#include <stdexcept>

class Helper {
 
public:
  using lumimap_t = std::unordered_map<std::string, float>;
  
  Helper(const std::vector<std::string> &combs, const std::vector<float> &weights) :
  
  lumimap_(std::make_shared<lumimap_t>()) {
    for (unsigned int i = 0; i < weights.size(); ++i) {
      lumimap_->insert(std::make_pair(combs[i], weights[i]));
    }
  }
    
  double operator () (int HLT_DoubleMu4_JpsiTrk_Displaced,int HLT_Dimuon0er16_Jpsi_NoVertexing,int HLT_Dimuon0_Jpsi_Muon,int HLT_Dimuon20_Jpsi,int HLT_DoubleMu4_3_Jpsi_Displaced,int HLT_Dimuon16_Jpsi,int HLT_Dimuon10_Jpsi_Barrel,int HLT_Mu7p5_Track2_Jpsi,int HLT_Mu7p5_Track3p5_Jpsi,int HLT_Mu7p5_Track7_Jpsi,int HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing) const {

    std::vector<int> my_vector = {HLT_DoubleMu4_JpsiTrk_Displaced,HLT_Dimuon0er16_Jpsi_NoVertexing,HLT_Dimuon0_Jpsi_Muon,HLT_Dimuon20_Jpsi,HLT_DoubleMu4_3_Jpsi_Displaced,HLT_Dimuon16_Jpsi,HLT_Dimuon10_Jpsi_Barrel,HLT_Mu7p5_Track2_Jpsi,HLT_Mu7p5_Track3p5_Jpsi,HLT_Mu7p5_Track7_Jpsi,HLT_Dimuon0er16_Jpsi_NoOS_NoVertexing};
    std::stringstream result;
    std::copy(my_vector.begin(), my_vector.end(), std::ostream_iterator<int>(result));
    const auto it = lumimap_->find(result.str());
    if (it != lumimap_->end()) {
      return it->second;
    }
    throw std::runtime_error("not found");
    
    return 0.;
  }
  
  
private:
  std::shared_ptr<lumimap_t> lumimap_;
  
};
