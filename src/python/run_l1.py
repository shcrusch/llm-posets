import llm_posets_l1
import sys
import pandas as pd
import itertools

import os
args=sys.argv

xij_min = 1000
xij_max = -1

X_arr= []

#dataset_file = '/home/hayashi/workspace/tbm-python/dataset/'+ args[1] + '.dat'

#itemset_file = '/home/hayashi/workspace/tbm-python/dataset/' + args[1] + '.dat_itemsets' + args[3]
dataset_file = 'test.dat'
#itemset_file = 'test.itemsets'

with open(dataset_file) as f:
    for line in f:
        xi = line.split()
        xi_int = [ int(xij)  for xij in xi ]        
        
        xij_min = min(xi_int) if xij_min > min(xi_int) else xij_min
        xij_max = max(xi_int) if xij_max < max(xi_int) else xij_max

        X_arr.append(xi_int)

X =[]
for xi in X_arr:   
    X.append(tuple([xij - xij_min for xij in xi]))

n = xij_max-xij_min+1
N = len(X)

#Frequently Itemset Mining
k = 3
sigma = 0.01

B = []
for i in range(k):
    if i == 0:
        Bi = []
        for j in range(n):
            phi = (j,)
            count = 0
            for x in X:
                count += set(phi).issubset(x)
            if count / N >= sigma:
                Bi.append(phi)
    else:
        pre_Bi = Bi
        Bi = []        
        for phis in itertools.combinations(pre_Bi,2):

            phi = set(phis[0]) | set(phis[1])

            if len(phi) == i+1:
                t = True
                for p in Bi:
                    if phi == set(p):
                        t = False
                        break
                if t:
                    count = 0
                    for x in X:
                        count += phi.issubset(x)
                    if count / N >= sigma:
                        Bi.append(tuple(phi))

    if len(Bi) == 0:
        break
    B += Bi


S = list(set(B+X+[()]))

llm = llm_posets_l1.LLM(n,B,S)
if args[1] == "coor":
    llm.fit(X, int(args[3]), lambda_ = float(args[2]) , solver = "coor")
elif args[1] == "prox":
    llm.fit(X, int(args[4]), stepsize = float(args[2]) , lambda_ = float(args[3]), solver = "prox")
elif args[1] == "rda":
    llm.fit(X, int(args[4]), stepsize = float(args[2]) , lambda_ = float(args[3]), solver = "rda")
else:
    print("Error")

Bhat = []
theta_hat = {}
for phi in B:
    if llm.theta_[phi] != 0:
        Bhat.append(phi)
        theta_hat[phi] = llm.theta_[phi]

print(Bhat)
print(theta_hat)
