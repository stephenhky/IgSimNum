# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:05:50 2013

@author: hok1
"""

import numpy as np
from operator import and_, add
import time
from multiprocessing import Pool

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
    return reduce(add, nodehash.values())
     
def condenseIgSeqNodes(igSeqNodes, p=None, probtol=None, seqlen=None):
    igSeqNodes = condenseDegenerateIgSeqNodes(igSeqNodes)
    if seqlen != None:
        igSeqNodes = filter(lambda node: node.score >= (seqlen-node.N+1)*node.mismatchScore,
                            igSeqNodes)
    if probtol != None and p != None:
        logprobtol = np.log(probtol)
        igSeqNodes = filter(lambda node: node.calculatelogprob(p)>logprobtol,
                            igSeqNodes)
    return igSeqNodes
    
def analyzeIgSeqNodes(igSeqNodes, p):
    stat = {}
    for node in igSeqNodes:
        if stat.has_key(node.maxScoreIdx):
            stat[node.maxScoreIdx] += float(node.deg)*np.exp(node.calculatelogprob(p))
        else:
            stat[node.maxScoreIdx] = float(node.deg)*np.exp(node.calculatelogprob(p))
    return stat
    
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
    
def simulateIgSeqPool(seqlen, vlen, p=None, probtol=None, numpools=1):
    nodes = [IgSeqNode(0, 0, vlen=vlen)]
    for N in range(seqlen):
        nump = min(numpools, len(nodes))
        numPerPool = int(np.ceil(len(nodes)/float(nump)))
        nodepartition = [nodes[numPerPool*i:min(numPerPool*(i+1), len(nodes))] for i in range(nump)]
        print N, len(nodes), nump, numPerPool, map(len, nodepartition)
        pool = Pool(nump)
        nodes_array = pool.map(generateNextNodes, nodepartition)
        nodes = condenseIgSeqNodes(reduce(add, nodes_array), p=p, 
                                   probtol=probtol, seqlen=seqlen)
    return nodes
    
if __name__ == '__main__':
    time1 = time.time()
    #nodes = simulateIgSeqPool(320, 300, p=0.05, probtol=1e-40, numpools=5)
    nodes = simulateIgSeq(15, 10)
    stat = analyzeIgSeqNodes(nodes, 0.05)
    time2 = time.time()
    print 'Time = ', (time2-time1), ' sec'
    for maxIdx in sorted(stat.keys()):
        print maxIdx, ':', stat[maxIdx]
    print 'Norm = ', sum(stat.values())
    print '# nodes = ', len(nodes)
