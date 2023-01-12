#include "module.hpp"

// std::vector<ROOT::RDF::RResultPtr<TH1D>> Module::getTH1()
// {
//     return _h1List;
// }
// std::vector<ROOT::RDF::RResultPtr<TH2D>> Module::getTH2()
// {
//     return _h2List;
// }
// std::vector<ROOT::RDF::RResultPtr<TH3D>> Module::getTH3()
// {
//     return _h3List;
// }

// std::vector<ROOT::RDF::RResultPtr<std::vector<TH1D *>>> Module::getGroupTH1()
// {
//     return _h1Group;
// }
// std::vector<ROOT::RDF::RResultPtr<std::vector<TH2D *>>> Module::getGroupTH2()
// {
//     return _h2Group;
// }
// std::vector<ROOT::RDF::RResultPtr<std::vector<TH3D *>>> Module::getGroupTH3()
// {
//     return _h3Group;
// }

// std::vector<ROOT::RDF::RResultPtr<std::map<std::string, boost_histogram>>> Module::getGroupTHN()
// {
//     return _hNGroup;
// }

// void Module::reset()
// {

//     _h1List.clear();
//     _h2List.clear();
//     _h3List.clear();

//     _h1Group.clear();
//     _h2Group.clear();
//     _h3Group.clear();
//     _hNGroup.clear();
// }

void Module::vary(std::string Column, std::vector<std::string> variations)
{
    _variationRules.insert(std::make_pair(Column, variations));
}

void Module::setVariationRules(std::map<std::string, std::vector<std::string>> variationRules)
{
    _variationRules = variationRules;
}

std::map<std::string, std::vector<std::string>> Module::getVariationRules()
{
    return _variationRules;
}
