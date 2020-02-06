import sys
import re


args = sys.argv
datasetname = args[1]
method = args[2]
stepormu = args[3]

mainfile = '/home/hayashi/workspace/tbm-python/experiments2/'+datasetname+'/'+method+'_'+stepormu+'.txt'
starfile = '/home/hayashi/workspace/tbm-python/experiments2/'+datasetname+'/acc_grad_0.97.txt'

L_star=1000
with open(starfile) as f:                                                                                                                    
    for line in f:                                                                                                                           
        li = re.split(' : KL divergence: |  time : ',line)                                                               
        L_star= float(li[1]) if float(li[1]) < L_star else L_star                                                                            

with open(mainfile) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : ',line)
        li[1]= str(float(li[1])-L_star)                                                                                                 
        li[2] = li[2].rstrip('\n')
        print(' '.join(li))



