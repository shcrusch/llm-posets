import re
import sys
args =sys.argv

datasetname = args[1]
filename = '/home/hayashi/workspace/tbm-python/experiments/'+datasetname+'/coor.txt'

L_star=1000
with open(filename) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        L_star= float(li[1]) if float(li[1]) < L_star else L_star

with open(filename) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        li[1]= str(float(li[1])-L_star)
        li[2] = li[2].rstrip('\n')
        print(' '.join(li))


