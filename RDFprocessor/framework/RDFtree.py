from header import *
import os
import copy
import h5py
import numpy as np
from array import array
import ROOT
from pathlib import Path
ROOT.gInterpreter.ProcessLine('#include "../RDFprocessor/framework/interface/DataFormat.h"')
ROOT.gInterpreter.ProcessLine('#include "../RDFprocessor/framework/interface/Utility.h"')

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
        
        self.variationsRules = ROOT.map("std::string", "std::vector<std::string>")() #systematic variations

        self.objs = {} # objects to be received from modules
        
        self.node = {} # dictionary branchName - RDF
        self.node['input'] = self.d # assign input RDF to a branch called 'input'

        self.graph = {} # save the graph to write it in the end 

        if not os.path.exists(self.outputDir):
            os.system("mkdir -p " + self.outputDir)
   
        os.chdir(self.outputDir)

        self.fout = ROOT.TFile(self.outputFile, "recreate")
        self.fout.Close()

        os.chdir("..")
    
    def __del__(self):
        # delete helper scripts 
        for p in Path(".").glob("helperbooker*"):
           p.unlink()
        pass
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
            
            m.setVariationRules(self.variationsRules)
            branchRDF = m.run(ROOT.RDF.AsRNode(branchRDF))
            self.variationsRules = m.getVariationRules()

            # tmp_th1 = m.getTH1()
            # tmp_th2 = m.getTH2()
            # tmp_th3 = m.getTH3()

            # tmp_th1G = m.getGroupTH1()
            # tmp_th2G = m.getGroupTH2()
            # tmp_th3G = m.getGroupTH3()
            # tmp_thNG = m.getGroupTHN()    

            # for obj in tmp_th1:
                    
            #     value_type = getValueType(obj)

            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))
                
            # for obj in tmp_th2:
                    
            #     value_type = getValueType(obj)

            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))

            # for obj in tmp_th3:
                    
            #     value_type = getValueType(obj)

            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))

            # for obj in tmp_th1G:
                    
            #     value_type = getValueType(obj)

            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))

            # for obj in tmp_th2G:
                    
            #     value_type = getValueType(obj)

            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))

            # for obj in tmp_th3G:
                    
            #     value_type = getValueType(obj)
                
            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))

            # for obj in tmp_thNG:

            #     value_type = getValueType(obj)

            #     self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(obj))

            # m.reset()

        self.node[nodeToEnd] = branchRDF

    def Histogram(self, columns, types, node, histoname, bins, sample=(), variations={}):

        d = self.node[node]
        rules = self.variationsRules
        self.branchDir = node

        if not len(columns)==len(types): raise Exception('number of columns and types must match')
        nweights = len(columns) - len(bins)
        Dsample = 1
        if not sample==(): 
            types.append('RVec<float>')
            columns.append(sample[0])
            # read column length
            Dsample = int(sample[1])
        
        totalsize = 1
        for b in bins:
            totalsize*=(len(b)-1)
        totalsize*=Dsample

        if totalsize>1.e6:
            if(Dsample==1):
                boost_type = "boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::weighted_sum<>, std::allocator<boost::histogram::accumulators::weighted_sum<>>>>>"
            else:
                boost_type = "boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::thread_safe_withvariance_sample<double, {Dsample}>, std::allocator<boost::histogram::accumulators::thread_safe_withvariance_sample<double, {Dsample}>>>>>".format(Dsample=Dsample)
        else:
            if(Dsample==1):
                boost_type = "boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::weighted_sum<>, std::allocator<boost::histogram::accumulators::weighted_sum<>>>>>"
            else:
                boost_type = "boost::histogram::histogram<std::vector<boost::histogram::axis::variable<>>, boost::histogram::storage_adaptor<std::vector<boost::histogram::accumulators::weighted_sum_vec<double, {Dsample}>, std::allocator<boost::histogram::accumulators::weighted_sum_vec<double, {Dsample}>>>>>".format(Dsample=Dsample)

        variations_vec = ROOT.vector(ROOT.vector('string'))()
        # reorder variations to follow column order
        
        for col in columns:
            if not variations=={} and col in rules:
                variations_vec.push_back(copy.deepcopy(rules.at(col))) #deepcopy otherwise it gets deleted
                columns.append(variations[col]) # append column containing variations
                types.append('RVec<float>')
                variations_vec.push_back(ROOT.vector('string')({""}))
            else:
                variations_vec.push_back(ROOT.vector('string')({""}))

        # passing templated binning arguments to maximise performance
        binningCode = 'auto bins_{} = std::make_tuple('.format(histoname)
        for bin in bins:
            binningCode+='std::make_tuple({}),'.format(', '.join(str(x) for x in bin))
        binningCode = ','.join(binningCode.split(',')[:-1])
        binningCode+=')'
        ROOT.gInterpreter.ProcessLine(binningCode)
        try:
            templ = type(getattr(ROOT,"bins_{}".format(histoname))).__cpp_name__
        except AttributeError:
            print('static method doesnt work. Falling back to dynamical instance...')
            templ = 'std::vector<std::vector<float>>'
            binningCode = 'auto bins_{} = std::vector{{'.format(histoname)
            for bin in bins: 
                binningCode+='std::vector{{{}}},'.format(', '.join(str(x) for x in bin))
            binningCode = ','.join(binningCode.split(',')[:-1])
            binningCode+='}'
            ROOT.gInterpreter.ProcessLine(binningCode)
        #############################################################
        # print("this is how I will make variations for this histogram")
        # for icol, col in enumerate(columns):
        #     print(col, variations_vec[icol])
        print("writing histogram ", histoname)
        #############################################################
        if "helperbooker_{}_cpp.so".format(histoname) not in ROOT.gSystem.GetLibraries():
            print('compiling')
            ROOT.gSystem.SetIncludePath("-I$ROOTSYS/include -I/scratchnvme/emanca/wproperties-analysis/templateMaker/interface -I/opt/boost/include")
            with open("helperbooker_{}.cpp".format(histoname), "w") as f:
                code = bookingCode.format(boost_histogram=boost_type, binsType = templ, template_args="{},{},{},{},{},{}".format(len(bins),nweights,Dsample,totalsize,templ,', '.join(types)),N=histoname)
                f.write(code)                                                                                                   
            ROOT.gSystem.CompileMacro("helperbooker_{}.cpp".format(histoname), "kO")                                                            
        
        histo = getattr(ROOT, "BookIt{}".format(histoname))(d, histoname, getattr(ROOT,"bins_{}".format(histoname)), columns,variations_vec) 
        
        value_type = getValueType(histo)
        self.objs[self.branchDir].append(ROOT.RDF.RResultPtr(value_type)(histo))


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

    def getROOTOutput(self):

        #start analysis
        self.start = time.time()

        # now write all the outputs together

        print("Writing output files in "+ self.outputDir)

        os.chdir(self.outputDir)
        self.fout = ROOT.TFile(self.outputFile, "update")
        self.fout.cd()

        obj_number = 0

        for branchDir, objs in self.objs.items():

            if objs==[]: continue #get rid of empty nodes
            if not self.fout.GetDirectory(branchDir): self.fout.mkdir(branchDir)

            self.fout.cd(branchDir)
            
            for obj in objs:
                
                if not 'TH' in type(obj).__cpp_name__:
                    continue
                elif 'vector' in type(obj).__cpp_name__:
                    
                    for h in obj:
                        obj_number =  obj_number+1
                        h.Write()
                else:
                    obj_number =  obj_number+1
                    obj.Write()

        
        #self.objs = {} # re-initialise object list
        self.fout.Close()
        os.chdir("..")

        print(self.entries.GetValue(), "events processed in "+"{:0.1f}".format(time.time()-self.start), "s", "rate", self.entries.GetValue()/(time.time()-self.start), "histograms written: ", obj_number)

    def gethdf5Output(self,branchDirs=None):

        #start analysis
        self.start = time.time()

        if branchDirs is None:
            branchDirs = list(self.objs.keys())
        os.chdir(self.outputDir)
        with h5py.File(self.outputFile.replace('root','hdf5'), mode="w") as f:
            dtype = 'float64'
            for branchDir, objs in self.objs.items():
                if objs == []: continue
                if not branchDir in branchDirs: continue #only write selected folders
                for obj in objs:
                    if not 'TH' in type(obj).__cpp_name__: #writing boost histograms
                        map = obj.GetValue()
                        for name,h in map:
                            print(name)
                            print(type(h).__cpp_name__)
                            if "boost::histogram::accumulators::thread_safe" in type(h).__cpp_name__:
                                D = getD(h)
                                arr = ROOT.convertAtomics[type(h).__cpp_name__,D](h)
                            else:
                                arr = ROOT.convert[type(h).__cpp_name__](h)
                                D = ROOT.getD[type(h).__cpp_name__](h)
                            rank = ROOT.getRank[type(h).__cpp_name__](h)
                            sizes=[]
                            for i in range(rank):
                                sizes.append(ROOT.getAxisSize[type(h).__cpp_name__](h,i))
                            
                            if not D==1: sizes.append(D)
                            
                            # get bin contents
                            counts = np.asarray(arr[0])
                            counts = np.array(counts.reshape(tuple(sizes),order='F'),order='C')
                            # get sum of squared weights
                            sumw = np.asarray(arr[1])
                            sumw = np.array(sumw.reshape(tuple(sizes),order='F'),order='C')
                            edges=[]
                            for i in range(rank):
                                axis=ROOT.getAxisEdges[type(h).__cpp_name__](h,i)
                                edges.append(np.asarray(axis))
                            dset = f.create_dataset('{}'.format(name), counts.shape, dtype=dtype)
                            dset[...] = counts
                            dset2 = f.create_dataset('{}_sumw2'.format(name), counts.shape, dtype=dtype)
                            dset2[...] = sumw
                            for i,axis in enumerate(edges):
                                dset3 = f.create_dataset('edges_{}_{}'.format(name,i), axis.shape, dtype='float32')
                                dset3[...] = axis
                    elif 'vector' in type(obj).__cpp_name__:
                        for h in obj:
                            nbins = h.GetNbinsX()*h.GetNbinsY() * h.GetNbinsZ()
                            dset = f.create_dataset('{}'.format(h.GetName()), [nbins], dtype=dtype)
                            harr = np.array(h)[1:-1].ravel().astype(dtype) #no under/overflow bins
                            dset[...] = harr
                            #save sumw2
                            if not h.GetSumw2().GetSize()>0: continue 
                            sumw2_hist = h.Clone()
                            dset2 = f.create_dataset('{}_sumw2'.format(h.GetName()), [nbins], dtype=dtype)
                            sumw2f=[sumw2_hist.GetSumw2()[i] for i in range(sumw2_hist.GetSumw2().GetSize())]
                            sumw2f = np.array(sumw2f,dtype='float64').ravel().astype(dtype)
                            dset2[...] = sumw2f
                    else:
                        nbins = obj.GetNbinsX()*obj.GetNbinsY() * obj.GetNbinsZ()
                        dset = f.create_dataset('{}'.format(obj.GetName()), [nbins], dtype=dtype)
                        harr = np.array(h)[1:-1].ravel().astype(dtype) #no under/overflow bins
                        dset[...] = harr
                        #save sumw2
                        if not obj.GetSumw2().GetSize()>0: continue 
                        sumw2_hist = obj.Clone()
                        dset2 = f.create_dataset('{}_sumw2'.format(obj.GetName()), [nbins], dtype=dtype)
                        sumw2f=[sumw2_hist.GetSumw2()[i] for i in range(sumw2_hist.GetSumw2().GetSize())]
                        sumw2f = np.array(sumw2f,dtype='float64').ravel().astype(dtype)
                        dset2[...] = sumw2f
        
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
