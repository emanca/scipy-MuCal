#ifndef GENLEPTONSELECTOR_H
#define GENLEPTONSELECTOR_H

#include "module.hpp"

class genLeptonSelector : public Module {

public:
  genLeptonSelector() {};
  ~genLeptonSelector() {};
  RNode run(RNode) override;
  
};

#endif
