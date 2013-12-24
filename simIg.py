# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:05:50 2013

@author: hok1
"""

import numpy as np
import math
from operator import and_, add
import time
from multiprocessing import Pool
import csv
import sys

prand = 0.25
qrand = 1 - prand

class IgSeqNode:
    def __init__(self, N, nc, score=0, maxScore=0, maxScoreIdx=0, 
                 deg=1, matchScore=5, mismatchScore=-4, vlen=100, nr=None):
        self.N = N
        self.nc = nc
        self.score = score
        self.maxScore = maxScore
        self.maxScoreIdx = maxScoreIdx
        self.deg = deg
        self.matchScore = matchScore
        self.mismatchScore = mismatchScore
        self.vlen = vlen
        if nr != None:
            self.nr = nr
        else:
            self.nr = 0
        
    def __eq__(self, other):
        return reduce(and_, [self.N==other.N, self.nc==other.nc, 
                             self.score==other.score, 
                             self.maxScore==other.maxScore,
                             self.maxScoreIdx==other.maxScoreIdx,
                             self.vlen==other.vlen, self.nr==other.nr])
                            
    def __ne__(self, other):
        return (not self.__eq__(other))
        
    def __str__(self):
        return '(N='+str(self.N)+', nc='+str(self.nc)+', vlen='+str(self.vlen)+', nr='+str(self.nr)+', score='+str(self.score)+', maxScore='+str(self.maxScore)+', maxScoreIdx='+str(self.maxScoreIdx)+', deg='+str(self.deg)+')'
        
    def __repr__(self):
        return self.__str__()        
        
    def __hash__(self):
        return self.maxScoreIdx+self.maxScore*1000+self.score*1000000+self.nc*100000000+self.nr*1000000000
       
    def generateNextNodes(self):
        # correct node
        scoreC = self.score+self.matchScore
        maxScoreC = max(scoreC, self.maxScore)
        maxScoreIdxC = self.maxScoreIdx if maxScoreC==self.maxScore else (self.N+1)
        ncc = self.nc + 1 if self.N+1<=self.vlen else self.nc
        nrc = self.nr + 1 if self.N+1>self.vlen else self.nr
        nodec = IgSeqNode(self.N+1, ncc, score=scoreC,
                          maxScore=maxScoreC, maxScoreIdx=maxScoreIdxC,
                          deg=self.deg,
                          matchScore=self.matchScore,
                          mismatchScore=self.mismatchScore,
                          vlen=self.vlen, nr=nrc)
                          
        # incorrect node
        noden = IgSeqNode(self.N+1, self.nc,
                          score=self.score+self.mismatchScore,
                          maxScore=self.maxScore, maxScoreIdx=self.maxScoreIdx,
                          deg=self.deg,
                          matchScore=self.matchScore,
                          mismatchScore=self.mismatchScore,
                          vlen=self.vlen, nr=self.nr)
                          
        return [nodec, noden]
    
    # p : mutation probability
    def calculatelogprob(self, p):
        if p > 0 and p < 1:        
            logprob = (min(self.vlen, self.N)-self.nc)*np.log(p)
            logprob += self.nc*np.log(1-p)
        elif p == 0.0:
            if min(self.vlen, self.N)-self.nc == 0:
                logprob = 0
            else:
                logprob = float('-inf')
                return logprob
        elif p == 1.0:
            if self.nc == 0:
                logprob = 0
            else:
                logprob = float('-inf')
                return logprob
        else:
            logprob = float('-inf')
            return logprob
        if self.N > self.vlen:
            logprob += (self.N-self.vlen-self.nr)*np.log(qrand)
            logprob += self.nr*np.log(prand)
        return logprob
        
# This function is not only slow but does not condense everything
def condenseDegenerateIgSeqNodes(igSeqNodes):
    idx = 0
    while idx < len(igSeqNodes):
        idx2 = idx+1
        while idx2 < len(igSeqNodes):
            if igSeqNodes[idx]==igSeqNodes[idx2]:
                igSeqNodes[idx].deg += igSeqNodes[idx2].deg
                del igSeqNodes[idx2]
            idx2 += 1
        idx += 1
    return igSeqNodes
     
def condenseDegenerateIgSeqNodesHash(nodes):
    nodehash = {}
    for node in nodes:
        hashcode = node.__hash__()
        if nodehash.has_key(hashcode):
            nodehash[hashcode].append(node)
        else:
            nodehash[hashcode] = [node]
    for nodelist in nodehash.values():
        if len(nodelist) > 1:
            sumdeg = sum(map(lambda node: node.deg, nodelist))
            nodelist[0].deg = sumdeg
            del nodelist[1:]
    return reduce(add, nodehash.values()) if len(nodehash.values()) > 0 else []
     
def condenseIgSeqNodes(igSeqNodes, p=None, probtol=None, seqlen=None):
    igSeqNodes = condenseDegenerateIgSeqNodesHash(igSeqNodes)
    if seqlen != None:
        igSeqNodes = filter(lambda node: node.score >= (seqlen-node.N+1)*node.mismatchScore,
                            igSeqNodes)
    if probtol != None and p != None:
        logprobtol = np.log(probtol)
        igSeqNodes = filter(lambda node: math.log(node.deg)+node.calculatelogprob(p)>logprobtol,
                            igSeqNodes)
    return igSeqNodes
    
def generateNextNodes(nodes):
    nextnodes = []
    for node in nodes:
        nextnodes += node.generateNextNodes()
    return nextnodes
    
def simulateIgSeq(seqlen, vlen, p=None, probtol=None):
    nodes = [IgSeqNode(0, 0, vlen=vlen)]
    for N in range(seqlen):
        nodes = generateNextNodes(nodes)
    nodes = condenseIgSeqNodes(nodes, p=p, probtol=probtol, seqlen=seqlen)
    return nodes
    
def generateNextCondensedNodes((nodes, p, probtol, seqlen)):
    nodes = generateNextNodes(nodes)
    nodes = condenseIgSeqNodes(nodes, p=p, probtol=probtol, seqlen=seqlen)
    return nodes
    
def simulateIgSeqPool(seqlen, vlen, p=None, probtol=None, numpools=1):
    nodes = [IgSeqNode(0, 0, vlen=vlen)]
    for N in range(seqlen):
        if len(nodes) > 0:
            nump = min(numpools, len(nodes))
            numPerPool = int(np.ceil(len(nodes)/float(nump)))
            nodepartition = [nodes[numPerPool*i:min(numPerPool*(i+1), len(nodes))] for i in range(nump)]
            nodepartition = filter(lambda array: len(array)>0, nodepartition)
            nump = len(nodepartition)
            print N, len(nodes), nump, numPerPool, map(len, nodepartition)
            pool = Pool(nump)
            nodes_array = pool.map(generateNextCondensedNodes, 
                                   map(lambda node: (node, p, probtol, seqlen),
                                       nodepartition))
            nodes = condenseIgSeqNodes(reduce(add, nodes_array), p=p, 
                                       probtol=probtol, seqlen=seqlen)
    return nodes
    
def analyzeIgSeqNodes(igSeqNodes, p):
    stat = {}
    for node in igSeqNodes:
        if stat.has_key(node.maxScoreIdx):
            stat[node.maxScoreIdx] += float(node.deg)*np.exp(node.calculatelogprob(p))
        else:
            stat[node.maxScoreIdx] = float(node.deg)*np.exp(node.calculatelogprob(p))
    return stat

def printTableFile(igSeqNodes, p, outputfilename):
    fout = open(outputfilename, 'wb')
    writer = csv.writer(fout)
    header = ['VEnd', 'VEndDiff', 'prob']
    writer.writerow(header)

    stat = analyzeIgSeqNodes(igSeqNodes, p)    
    vlen = igSeqNodes[0].vlen if len(igSeqNodes) > 0 else 0
    for maxIdx in sorted(stat.keys()):
        writer.writerow([maxIdx, maxIdx-vlen, stat[maxIdx]])
    fout.close()
    
def favprobtol(p):
    return 1e-10
    
def runOne(seqlen, vlen, p, probtol, numpools):
    time1 = time.time()
    nodes = simulateIgSeqPool(seqlen, vlen, p=p, probtol=probtol, 
                              numpools=numpools)
    stat = analyzeIgSeqNodes(nodes, p)
    time2 = time.time()
    print 'Time = ', (time2-time1), ' sec'
    for maxIdx in sorted(stat.keys()):
        print maxIdx, ':', stat[maxIdx]
    print 'Norm = ', sum(stat.values())
    print '# nodes = ', len(nodes)
    
def BatchSimulation(mutProbs, summaryfilename='IgSeqVEndSim_Summary.csv'):
    fsimsum = open(summaryfilename, 'wb')
    writer = csv.writer(fsimsum)
    header = ['seqlen', 'vlen', 'mutprob', 'num.nodes', 'sumdeg', 'probtol', 
              'norm', 'time', 'numpools']
    writer.writerow(header)
    npools = 10
    seqlen = 318
    vlen = 303
    for mutprob in mutProbs:
        print 'mutprob = ', mutprob
        time1 = time.time()
        optprobtol = favprobtol(mutprob)
        nodes = simulateIgSeqPool(seqlen, vlen, p=mutprob, probtol=optprobtol, 
                                  numpools=npools)
        time2 = time.time()
        stat = analyzeIgSeqNodes(nodes, mutprob)
        sumdeg = sum(map(lambda node: node.deg, nodes))
        norm = sum(stat.values())
        printTableFile(nodes, mutprob,
                       'IgSeqVEndSim_'+('%.2f' % mutprob)+'.csv')
        writer.writerow([seqlen, vlen, mutprob, len(nodes), sumdeg, 
                         optprobtol, norm, (time2-time1), npools])
    fsimsum.close()

if __name__ == '__main__':
    argvs = sys.argv
    if argvs[1]=='even':
        BatchSimulation(np.linspace(0.00, 0.40, num=5),
                        summaryfilename='IgSeqVEndSim_Summary_even.csv')
    elif argvs[1]=='odd':
        BatchSimulation(np.linspace(0.05, 0.45, num=5),
                        summaryfilename='IgSeqVEndSim_Summary_odd.csv')
    else:
        BatchSimulation(np.linspace(0.00, 0.45, num=10))
