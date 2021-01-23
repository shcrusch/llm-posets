import sys
import re
import glob

args = sys.argv
datasetname = args[1]
method = args[2]

targetfile = datasetname+'/'+method+'.txt'

gradfiles = glob.glob(datasetname+'/grad_*') #update

accgradfiles = glob.glob(datasetname+'/acc_grad_*')


coorfile = [datasetname+'/coor.txt'] #update

allfiles = gradfiles + accgradfiles + coorfile

L_star=1000

for filename in allfiles:
    with open(filename) as f:                                                                                                                    
        for line in f:                                                                                                                           
            li = re.split(' : KL divergence: |  time : ',line) 
            
            L_star= float(li[1]) if float(li[1]) < L_star else L_star                                                                            


with open(targetfile) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        li[1]= str(float(li[1])-L_star)
        li[2] = li[2].rstrip('\n')
        print(' '.join(li))



