# Base class from which the other modules will inherit
import ROOT
from header import *

class module:
   
    def __init__(self):
        
        self.myTH1 = []
        self.myTH2 = []
        self.myTH3 = []

        self.myTH1Group = []
        self.myTH2Group = []
        self.myTH3Group = []
        self.myTHNGroup = []

        self.variationRules = ROOT.map("std::pair<std::string, bool>", "std::vector<std::string>")()

    def run(self,d):

        pass 

    def getTH1(self):

        return self.myTH1

    def getTH2(self):

        return self.myTH2  

    def getTH3(self):

        return self.myTH3 

    def getGroupTH1(self):

        return self.myTH1Group

    def getGroupTH2(self):

        return self.myTH2Group  

    def getGroupTH3(self):

        return self.myTH3Group

    def getGroupTHN(self):

        return self.myTHNGroup

    def reset(self):

        self.myTH1 = []
        self.myTH2 = []
        self.myTH3 = [] 

        self.myTH1Group = []
        self.myTH2Group = []
        self.myTH3Group = [] 
        self.myTHNGroup = []
    
    def setVariationRules(self, map):

        self.variationRules = map
    
    def getVariationRules(self):

        return self.variationRules


