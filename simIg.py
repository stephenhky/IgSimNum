# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:05:50 2013

@author: hok1
"""

import numpy as np
from operator import and_

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
        return 'N='+str(self.N)+', nc='+str(self.nc)+', vlen='+str(self.vlen)+', nr='+str(self.nr)+', score='+str(self.score)+', maxScore='+str(self.maxScore)+', maxScoreIdx='+str(self.maxScoreIdx)+', deg='+str(self.deg)
        
    def __hash__(self):
        return self.maxScoreIdx+self.maxScore*1000+self.score*1000000+self.nc*100000000
       
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
        
def condenseIgSeqNodes(igSeqNodes, p=None, probtol=None):
    idx = 0
    while idx < len(igSeqNodes):
        idx2 = idx+1
        while idx2 < len(igSeqNodes):
            if igSeqNodes[idx]==igSeqNodes[idx2]:
                igSeqNodes[idx].deg += igSeqNodes[idx2].deg
                del igSeqNodes[idx2]
            idx2 += 1
        idx += 1
    if probtol != None and p != None:
        logprobtol = np.log(probtol)
        igSeqNodes = filter(lambda node: node.calculatelogprob(p)>logprobtol,
                            igSeqNodes)
    return igSeqNodes
    
def analyzeIgSeqNodes(igSeqNodes, p):
    stat = {}
    for node in igSeqNodes:
        if stat.has_key(node.maxScoreIdx):
            stat[node.maxScoreIdx] += node.deg*np.exp(node.calculatelogprob(p))
        else:
            stat[node.maxScoreIdx] = node.deg*np.exp(node.calculatelogprob(p))
    return stat
    
def simulateIgSeq(seqlen, vlen, p=None, probtol=None):
    nodes = [IgSeqNode(0, 0, vlen=vlen)]
    for N in range(seqlen):
        nextnodes = []
        for node in nodes:
            nextnodes += node.generateNextNodes()
        nodes = condenseIgSeqNodes(nextnodes, p, probtol=probtol)
    return nodes
