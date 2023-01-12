# Built-in/Generic Imports
import os
import sys
import time
import ROOT

def getValueType(obj):
    
    class_name = type(obj).__cpp_name__
    open_br, close_br = class_name.find('<'), class_name.rfind('>')
    value_type = class_name[open_br+1:close_br]
    return value_type

def getD(obj):
    class_name = type(obj).__cpp_name__
    open_br, close_br = class_name.find('thread_safe_withvariance_sample<double,'), class_name.rfind('>')
    value_type = class_name[open_br+39:close_br-6]
    return int(value_type)

bookingCode = """
#include "DataFormat.h"

ROOT::RDF::RResultPtr<std::map<std::string, {boost_histogram}>>
BookIt{N}(RNode d, std::string name, {binsType} bins, const std::vector<std::string> &columns, std::vector<std::vector<std::string>> variationRules) {{
    return Histogram<{boost_histogram},{template_args}>()(d, name, bins, columns, variationRules);
}}
"""