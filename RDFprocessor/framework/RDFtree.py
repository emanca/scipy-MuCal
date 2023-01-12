from header import *
import os
import copy
import pickle
import gzip
import narf
import hist
import lz4.frame
import numpy as np
from array import array
import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
from pathlib import Path

class RDFtree:
    def __init__(self, outputDir, outputFile, inputFile,treeName='Events', pretend=False):

        self.outputDir = outputDir # output directory
        self.outputFile = outputFile
        self.inputFile = inputFile
        
        self.treeName = treeName

        RDF = ROOT.ROOT.RDataFrame

        self.pretend = pretend
        if self.pretend:

            ROOT.ROOT.DisableImplicitMT()
            self.d = RDF(self.treeName, self.inputFile)
            self.d=self.d.Range(1000)
        else:
            self.d = RDF(self.treeName, self.inputFile)
        
        self.entries = self.d.Count() #stores lazily the number of events
        
        self.modules = []
        
        self.objs = {} # objects to be received from modules
        
        self.node = {} # dictionary branchName - RDF
        self.node['input'] = self.d # assign input RDF to a branch called 'input'

        self.graph = {} # save the graph to write it in the end 

        if not os.path.exists(self.outputDir):
            os.system("mkdir -p " + self.outputDir)
    
    def branch(self,nodeToStart, nodeToEnd, modules=[]):

        self.branchDir = nodeToEnd
        if not self.branchDir in self.objs:
            self.objs[self.branchDir] = []
   
        if nodeToStart in self.graph:
            self.graph[nodeToStart].append(nodeToEnd)
        else: 
            self.graph[nodeToStart]=[nodeToEnd]

        branchRDF = self.node[nodeToStart]

        lenght = len(self.modules)

        self.modules.extend(modules)
        
        # modify RDF according to modules

        for m in self.modules[lenght:]: 
            
            branchRDF = m.run(ROOT.RDF.AsRNode(branchRDF))

        self.node[nodeToEnd] = branchRDF

    def Histogram(self, node, name, cols, axes, tensor_axes=''):

        d = self.node[node]
        self.branchDir = node

        if tensor_axes=='':
            self.objs[self.branchDir].append(d.HistoBoost(name, axes, cols))
        else:
            self.objs[self.branchDir].append(d.HistoBoost(name, axes, cols, tensor_axes=tensor_axes))

    def Snapshot(self, node, blist=[]):

        opts = ROOT.ROOT.RDF.RSnapshotOptions()
        opts.fLazy = True

        branchList = ROOT.vector('string')()

        for l in blist:
            branchList.push_back(l)

        if not len(blist)==0:
            out = self.node[node].Snapshot(self.treeName,self.outputFile, branchList, opts)
        else:
            out = self.node[node].Snapshot(self.treeName,self.outputFile, "", opts)

        self.objs[self.branchDir].append(out)


    def getOutput(self,branchDirs=None):

        #start analysis
        self.start = time.time()

        if branchDirs is None:
            branchDirs = list(self.objs.keys())
        os.chdir(self.outputDir)
        output = {}
        with lz4.frame.open(self.outputFile.replace('root','pkl.lz4'), "wb") as f:
            for branchDir, objs in self.objs.items():
                if objs == []: continue
                if not branchDir in branchDirs: continue #only write selected folders
                for obj in objs:
                    if isinstance(obj.GetValue(), ROOT.TNamed):
                        output[obj.GetName()] = obj.GetValue()

                    elif hasattr(obj.GetValue(), "name"):
                        output[obj.GetValue().name] = obj.GetValue()
                    else:
                        output[str(hash(obj.GetValue()))] = obj.GetValue()

            pickle.dump(output, f)
        
        print(self.entries.GetValue(), "events processed in "+"{:0.1f}".format(time.time()-self.start), "s", "rate", self.entries.GetValue()/(time.time()-self.start))
        os.chdir("..")

    def getObjects(self):
        return self.objs

    def saveGraph(self):

        ROOT.RDF.SaveGraph(self.node['input'],"graph.pdf")
        # print(self.graph)

        # from graphviz import Digraph

        # dot = Digraph(name='my analysis', filename = 'graph.pdf')

        # for node, nodelist in self.graph.items():

        #     dot.node(node, node)
        #     for n in nodelist:
        #         dot.node(n, n)
        #         dot.edge(node,n)

        # dot.render()  

    def EventFilter(self,nodeToStart, nodeToEnd, evfilter, filtername):
        if not nodeToEnd in self.objs:
            self.objs[nodeToEnd] = []
   
        if nodeToStart in self.graph:
            self.graph[nodeToStart].append(nodeToEnd)
        else: 
            self.graph[nodeToStart]=[nodeToEnd]

        branchRDF = self.node[nodeToStart]
        branchRDF = ROOT.RDF.AsRNode(ROOT.RDF.AsRNode(branchRDF).Filter(evfilter, filtername))
        self.node[nodeToEnd] = branchRDF

    def getCutFlowReport(self, node):
        return self.node[node].Report()

    def displayColumn(self, node, columnList=[], nrows=1000000):
        print("careful: this is triggering the event loop!")
        if node not in self.node:
            print("Node {} does not exist! Skipping display!".format(node))
            return -1
        columnVec = ROOT.vector('string')(columnList)
        self.node[node].Display(columnVec, nrows).Print()
