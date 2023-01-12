#ifndef MODULE_H
#define MODULE_H

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include <boost/histogram.hpp>
#include <boost/functional/hash.hpp>
#include "weighted_sum_vec.hpp"
#include <memory>
#include <tuple>

using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;

// using boost_histogram = boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::weighted_sum_vec<double>, std::allocator<boost::histogram::accumulators::weighted_sum_vec<double>>>>>;
// using boost_histogram = boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::weighted_sum<>, std::allocator<boost::histogram::accumulators::weighted_sum<>>>>>;

class Module
{

private:
public:
  virtual ~Module(){};
  virtual RNode run(RNode d) = 0;

  // std::vector<ROOT::RDF::RResultPtr<TH1D>> _h1List;
  // std::vector<ROOT::RDF::RResultPtr<TH2D>> _h2List;
  // std::vector<ROOT::RDF::RResultPtr<TH3D>> _h3List;
  // std::vector<ROOT::RDF::RResultPtr<std::vector<TH1D *>>> _h1Group;
  // std::vector<ROOT::RDF::RResultPtr<std::vector<TH2D *>>> _h2Group;
  // std::vector<ROOT::RDF::RResultPtr<std::vector<TH3D *>>> _h3Group;
  // std::vector<ROOT::RDF::RResultPtr<std::map<std::string, boost_histogram>>> _hNGroup;
  //keep track of systematic variations
  std::map<std::string, std::vector<std::string>> _variationRules; //std::map<std::pair<column,isWeight>, std::vector<variation_name>>

  // std::vector<ROOT::RDF::RResultPtr<TH1D>> getTH1();
  // std::vector<ROOT::RDF::RResultPtr<TH2D>> getTH2();
  // std::vector<ROOT::RDF::RResultPtr<TH3D>> getTH3();
  // std::vector<ROOT::RDF::RResultPtr<std::vector<TH1D *>>> getGroupTH1();
  // std::vector<ROOT::RDF::RResultPtr<std::vector<TH2D *>>> getGroupTH2();
  // std::vector<ROOT::RDF::RResultPtr<std::vector<TH3D *>>> getGroupTH3();
  // std::vector<ROOT::RDF::RResultPtr<std::map<std::string, boost_histogram>>> getGroupTHN();

  // void reset();
  void vary(std::string, std::vector<std::string>);
  void setVariationRules(std::map<std::string, std::vector<std::string>>);
  std::map<std::string, std::vector<std::string>> getVariationRules();
};

#endif
