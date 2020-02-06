import re
import sys
args =sys.argv

datasetname = args[1]
coorfile = '/home/hayashi/workspace/tbm-python/experiments2/'+datasetname+'/coor.txt'
starfile = '/home/hayashi/workspace/tbm-python/experiments2/'+datasetname+'/acc_grad_0.97.txt'
L_star=1000
with open(starfile) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        L_star= float(li[1]) if float(li[1]) < L_star else L_star

with open(coorfile) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        li[1]= str(float(li[1])-L_star)
        li[2] = li[2].rstrip('\n')
        print(' '.join(li))


