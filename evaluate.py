'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np


# Global variables that are shared across processes
def eval_one(ranklist,gtItem):
    # hrs,ndcgs,precisions,maps,coverages=[],[],[],[],[]
    # for idx in range(len(ranklist)):
        hr =getHitRatio(ranklist,gtItem)
        ndcg=getNDCG(ranklist,gtItem)
        precision = getPrecision(ranklist,gtItem)
        map=getMap(ranklist,gtItem)
        coverage=getCoverage(ranklist,gtItem)
    #     hrs.append(hr)
    #     ndcgs.append(ndcg)
    #     precisions.append(precision)
    #     maps.append(map)
    #     coverages.append(coverage)
        return hr,ndcg,precision,map,coverage
# def eval_one(ranklist,gtItem):
#     hits,ndcgs,precisions,maps,coverages=[],[],[],[],[]
#     for idx in range(len(gtItem)):
#         hr =getHitRatio(ranklist,gtItem)
#         ndcg=getNDCG(ranklist,gtItem)
#         precision = getPrecision(ranklist,gtItem)
#         map=getMap(ranklist,gtItem)
#         coverage=getCoverage(ranklist,gtItem)
#         hits.append(hr)
#         ndcgs.append(ndcg)
#         precisions.append(precision)
#         maps.append(map)
#         coverages.append(coverage)
#
#     return hits,ndcgs,precisions,maps,coverages

def getHitRatio(ranklist, gtItem):
    hit=0
    for item in ranklist:
        if item in gtItem:
            hit+=1
    # print("hit=",hit)
    if hit>0:
        # print("hit=", hit / len(gtItem))
        return hit/len(gtItem)
    else:
        return 0
def getNDCG(ranklist, gtItem):
    ndcg=0
    for i in range(len(ranklist)):
        item = ranklist[i]
        # print("item", item)
        if item in gtItem:
            ndcg= math.log(2) / math.log(i + 2)
    # print("ndcg=",ndcg)
    if ndcg >0:

        return ndcg
    else:
        return 0
def getPrecision(ranklist,gtItem):
     hit=0
     for item in ranklist:
          if item in gtItem:
              hit+=1
     # print("hitprecision=",hit)
     if hit>0:
         return hit/len(ranklist)
         # print("Pre=",hit/len(ranklist))
     else:
         return 0
def getMap(ranklist,gtItem):
     hit =0
     sum=0
     for i in range(len(ranklist)):
         item = ranklist[i]
         if item in gtItem:
             hit+=1
             sum+=hit/(i+1.0)
     # print("sum=",sum)
     if hit>0:
         # print("map=",sum/len(gtItem))
         return sum/len(gtItem)
     else:
         return 0
def getCoverage(ranklist,gtItem):
    ItemList=[]
    ItemList.append(gtItem)
    rec_items = set()
    for itemid in ranklist:
        rec_items.add(itemid)
    if len(ItemList) > 0:
        # print("cove=",len(rec_items)/len(ItemList))
        return len(rec_items)/len(ItemList)
    else:
        return 0










