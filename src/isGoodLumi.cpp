#include "isGoodLumi.hpp"
#include<string>
RNode isGoodLumi::run(RNode d) {

  auto isGoodlumi = [this](unsigned int run, unsigned int lumi) {
    std::string rkey = std::to_string(run);
    // find a run
    if (_datajson.find(rkey) != _datajson.end()) {//first check if run in json
      for( auto& lsL : _datajson.at(rkey)) {//these are the lumi blocks
	unsigned int low = lsL.at(0);
	unsigned int up = lsL.at(1);
	if(lumi >= low && lumi <=up)//Lumi is in range
	  return true;//there is no else part since one has loop over all LS blocks in a Run	
      }//lsb loop
    }//if block 
    return false;
  };

  auto df = d.Define("isGoodLumi", isGoodlumi, {"run", _lumi});
  return df;
}
