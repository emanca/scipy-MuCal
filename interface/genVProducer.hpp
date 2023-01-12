#ifndef GENVPRODUCER_H
#define GENVPRODUCER_H

#include "module.hpp"

class genVProducer : public Module {

public:
  genVProducer() {};
  ~genVProducer() {};
  RNode run(RNode) override;
  
};

#endif
