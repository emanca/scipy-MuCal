#ifndef ISGOODLUMI_H
#define ISGOODLUMI_H

#include "module.hpp"
#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

class isGoodLumi : public Module
{

private:
  json _datajson;
  std::string _lumi;

public:
  isGoodLumi(const std::string gjsonF, std::string lumi)
  {
    _lumi = lumi;
    std::ifstream fin(gjsonF.c_str());
    fin >> _datajson;
    fin.close();
  };
  ~isGoodLumi(){};
  RNode run(RNode) override;
};

#endif
