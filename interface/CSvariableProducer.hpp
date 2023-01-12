#ifndef CSVARIABLEPRODUCER_H
#define CSVARIABLEPRODUCER_H

#include "module.hpp"

class CSvariableProducer : public Module {

public:
  CSvariableProducer() {};
  ~CSvariableProducer() {};
  RNode run(RNode) override;
  
};

#endif
