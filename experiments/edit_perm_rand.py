import re
import sys
import glob
import os
import numpy as np
datasetname = "mushroom"

save_dir = '/home/hayashi/workspace/llm-posets/experiments/'+datasetname+'/'

permfiles = glob.glob('/home/hayashi/workspace/llm-posets/experiments/'+datasetname+'/coor_perm*')
randfiles = glob.glob('/home/hayashi/workspace/llm-posets/experiments/'+datasetname+'/coor_rand*')
starfile = '/home/hayashi/workspace/llm-posets/experiments/'+datasetname+'/.txt'

L_star=1000
with open(starfile) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        L_star= float(li[1]) if float(li[1]) < L_star else L_star

y_perms = []
y_rands = []
for permfile in permfiles:
    if permfile == permfiles[0]:
        with open(permfile) as f:
            for line in f:
                li = re.split(' : KL divergence: |  time : ',line)
                y_perms.append([float(li[1])])
    else:
        with open(permfile) as f:
            j = 0
            for line in f:
                li = re.split(' : KL divergence: |  time : ',line)
                y_perms[j].append(float(li[1]))
                j += 1
    

for randfile in randfiles:
    if randfile == randfiles[0]:
        with open(randfile) as f:
            for line in f:
                li = re.split(' : KL divergence: |  time : ',line)
                y_rands.append([float(li[1])])
    else:
        with open(randfile) as f:
            j = 0
            for line in f:
                li = re.split(' : KL divergence: |  time : ',line)
                y_rands[j].append(float(li[1]))
                j += 1

y_perm_means = []
y_rand_means = []
for i in range(1001):
    y_perm_means.append(str(i)+' ' +str(np.mean(y_perms[i]) -  L_star) )
    y_rand_means.append(str(i)+' '+str(np.mean(y_rands[i])  - L_star) )

every = 100
y_perm_errors = []
y_rand_errors = []
for i in range(11):
    y_perm_errors.append(str(every*i) +' '+ str(np.mean(y_perms[every*i]) - L_star) +' '+ str(np.std(y_perms[every*i])))
    y_rand_errors.append(str(every*i) +' '+ str(np.mean(y_rands[every*i]) - L_star) +' '+ str(np.std(y_rands[every*i])))

y_perm_means = '\n'.join(y_perm_means)
y_rand_means = '\n'.join(y_rand_means)
with open('mushroom/perm_means.txt','w') as f:
    f.write(y_perm_means)

with open('mushroom/rand_means.txt','w') as f:
    f.write(y_rand_means)

y_perm_errors = '\n'.join(y_perm_errors)
y_rand_errors = '\n'.join(y_rand_errors)
with open('mushroom/perm_errors.txt','w') as f:
    f.write(y_perm_errors)

with open('mushroom/rand_errors.txt','w') as f:
    f.write(y_rand_errors)

