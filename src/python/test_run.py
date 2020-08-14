import llm_posets
import llm_posets_l1
import sys
import pandas as pd
import itertools
import numpy as np
#import glob
import matplotlib.pyplot as plt
import os
args=sys.argv

xij_min = 1000
xij_max = -1

X_arr= []

dataset_file = '/home/hayashi/workspace/tbm-python/dataset/test.dat'


k_list = [1,2,3,4,5,6,7,8,9,10]
freq_list = [1,5,10,50,100,500,700,900]
lcm_dic = {}
for k in k_list:
    lcm_dic[k] = []
    for freq in freq_list:
        lcm_dic[k].append('/home/hayashi/workspace/tbm-python/dataset/test_lcm/test.dat_itemsets_' + str(k) + '_' + str(freq))
theta_star_file = '/home/hayashi/workspace/tbm-python/dataset/test.theta'

#Setting Dataset 
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


n = 10
S = [()]
S_ = []
for i in range(n-1):
    for x in itertools.combinations(range(n) , i+1):
        S.append(tuple(x))
        S_.append(tuple(x))

#Setting θ*
B_star = []
theta_star  = {}
with open(theta_star_file) as f:
    for line in f:
        li = line.split()
        phi = li[:-1]
        phi_int = [int(x) for x in phi]
        phi_int.sort()
        phi_int = tuple(phi_int)
        theta_phi = float(li[-1])
        theta_star[phi_int] = theta_phi
        if theta_phi != 0:
            B_star.append(phi_int)

f = 0.0
for phi in B_star:
    f+= theta_star[phi] ** 2
print('|θ*|^2: ',f)

def includes(phi,x):
  return set(phi).issubset(x)

def compute_theta_perp(theta,B):
    r = 0.0 
    for x in S:
        s = 0.0 # sum theta_phi phi(x) for theta in B
        for phi in B:
            if includes(phi,x):
                s += theta[phi]
        r += np.exp(s)
    return -np.log(r)

def compute_P(theta,theta_perp,B):
    P_ = {}
    for x in S:
        P_[x] = np.exp( compute_logp(x,theta,theta_perp,B) )
    return P_

def compute_logp(x,theta,theta_perp,B):
    ret = 0.0
    for phi in B:
        if includes(phi,x):
            ret += theta[phi]
    ret += theta_perp
    return ret


def compute_KL(P_,P):
    ret = 0.0
    for x in S:
        if P_[x] != 0 and P[x] != 0:
            ret += P_[x] * np.log( P_[x] / P[x] )
    return ret

def get_B_from(filename):
  # file format:                                                             
  # each line is a frequent itemset with 1-start                               
  # ignore the first line that represents "bottom"                             
    B = []
    with open(filename) as f:
        for line in f:
            li = line.split()
            phi = li[:-1]
            phi_int = [int(x) for x in phi]
            phi_int.sort()
            phi_int = tuple(phi_int)
            B.append(phi_int)
    return B

"""

llm_l1 = llm_posets_l1.LLM(n,S_,S)
llm_l1.fit(X, 101, lambda_ = 0, solver = "coor")
theta_hat_l1 = llm_l1.theta_
f = 0.0
for x in S_:
    f += ( theta_star[x] - theta_hat_l1[x] ) ** 2
print(f)
"""
theta_star_perp = compute_theta_perp(theta_star,B_star)
print('θ*_perp: ', theta_star_perp)
P_star = compute_P(theta_star,theta_star_perp,B_star)
print(sum(P_star.values()))

Bhat_size = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
f_dic = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
sym_diff = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
KL_hat = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
KL_star = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
"""
for k in lcm_dic:
    for i in range(len(lcm_dic[k])):
        itemset_file = lcm_dic[k][i]
        B = get_B_from(itemset_file)
        llm = llm_posets.LLM(n,B,S)
        llm.fit(X, 201, solver = "coor")
        theta_hat = llm.theta_
        theta_hat_perp = compute_theta_perp(theta_hat,B)
        P = compute_P(theta_hat,theta_hat_perp,B)
        
        f = 0.0
        for x in S_:
            f += ( theta_star[x]-theta_hat[x] )**2
        Bhat_size[k].append(len(B))
        f_dic[k].append(f)
        symdif = len(set(B) ^ set(B_star))
        sym_diff[k].append(symdif)
        KL_hat[k].append( compute_KL(llm.Phat_, P) )
        KL_star[k].append( compute_KL(P_star, P) )
    print("Finished "+str(k))
"""

"""
for i in range(len(lcm_dic[10])):
    itemset_file = lcm_dic[10][i]
    B = get_B_from(itemset_file)
    llm = llm_posets.LLM(n,B,S)
    llm.fit(X, 201, solver = "coor")
    theta_hat = llm.theta_
    
    f = 0.0
    for x in S_:
        f += ( theta_star[x]-theta_hat[x] )**2
    Bhat_size[10].append(len(B))
    f_dic[10].append(f)
    symdif = len(set(B) ^ set(B_star))
    sym_diff[10].append(symdif)
    KL_hat[10].append( compute_KL(llm.Phat_, llm.P_) )
    KL_star[10].append( compute_KL(P_star, llm.P_) )
print("Finished "+str(10))
print('KL(P^||P): ',KL_hat[10])
#print('KL(P*||P): ',KL_star[10])
"""
Bhat_size_l1 =[]
Bhat_l1 = []
f_l1 = []
sym_diff_l1 = []
KL_hat_l1 = []
KL_star_l1 = []
lambda_list = [0.01,0.02,0.05,0.1,0.2]

for lambda_ in lambda_list:
    print('Start '+str(lambda_))
    llm = llm_posets_l1.LLM(n,S_,S)
    llm.fit(X, 201, lambda_ = lambda_, solver = "coor")
    theta_hat = llm.theta_
    
    B_hat = []
    Bsize = 0
    f = 0.0
    for phi in llm.theta_:
        if llm.theta_[phi] != 0:
            Bsize += 1
            B_hat.append(phi)
    for phi in S_:
        f += (llm.theta_[phi] - theta_star[phi]) ** 2
    symdif = len(set(B_star) ^ set(B_hat))
    sym_diff_l1.append(symdif)
    Bhat_size_l1.append(Bsize)
    f_l1.append(f)
    
    KL_hat_l1.append(compute_KL(llm.Phat_, llm.P_))
    KL_star_l1.append(compute_KL(P_star, llm.P_))
    print('Finished '+str(lambda_))
print(KL_hat_l1)
print(KL_star_l1)

"""
#Plotting |theta^-theta*|**2
save_dir = '/home/hayashi/workspace/tbm-python/src/python/'
savefilename = save_dir + 'test_theta.png'
fig, ax = plt.subplots()

#for k in lcm_dic:
#    ax.plot(Bhat_size[k],f_dic[k],label='k='+str(k),linewidth=1)
ax.plot(Bhat_size[10],f_dic[10],label='k='+str(10),linewidth=1)
ax.plot(Bhat_size_l1,f_l1,label='L1',linewidth=1,color='r')

ax.legend(loc = 'upper right')
ax.set_xlabel('|B^|')
ax.set_ylabel('|theta^-theta*|**2')
fig.savefig(os.path.join(save_dir,savefilename))

#Plotting |B^ △ B*|
savefilename = save_dir + 'test_B.png'
fig, ax = plt.subplots()

#for k in lcm_dic:
#    ax.plot(Bhat_size[k],sym_diff[k],label='k='+str(k),linewidth=1)
ax.plot(Bhat_size[10],sym_diff[10],label='k='+str(10),linewidth=1)
ax.plot(Bhat_size_l1,sym_diff_l1,label='L1',linewidth=1,color='r')

ax.legend(loc = 'upper right')
ax.set_xlabel('|B^|')
ax.set_ylabel('|B^ △ B*|')
fig.savefig(os.path.join(save_dir,savefilename))

#Plotting KL
savefilename = save_dir + 'test_KL.png'
fig, ax = plt.subplots()
#for k in lcm_dic:
#    ax.plot(Bhat_size[k],KL_hat[k],label='k='+str(k)+',KL(P^||P)',linewidth=1,linestyle='dashed')
#    ax.plot(Bhat_size[k],KL_star[k],label='k='+str(k)+'KL(P*||P)',linewidth=1,linestyle='dashed')
ax.plot(Bhat_size[10],KL_hat[10],label='k='+str(10)+',KL(P^||P)',linewidth=1, color = 'b', linestyle='dashed')
ax.plot(Bhat_size[10],KL_star[10],label='k='+str(10)+'KL(P*||P)',linewidth=1, color = 'b', linestyle='solid')
ax.plot(Bhat_size_l1,KL_hat_l1,label='L1,KL(P^||P)',linewidth=1,color='r',linestyle='dashed')
ax.plot(Bhat_size_l1,KL_star_l1,label='L1,KL(P*||P)',linewidth=1,color='r', linestyle = 'solid')

ax.legend(loc = 'upper right')
ax.set_xlabel('|B^|')
ax.set_ylabel('KL divergence')
fig.savefig(os.path.join(save_dir,savefilename))
"""

