#ifndef MODULE_H
#define MODULE_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include <memory>
#include <tuple>

using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;

class Module
{

private:
public:
  virtual ~Module(){};
  virtual RNode run(RNode d) = 0;

};

#endif
