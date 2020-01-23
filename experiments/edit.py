import re
import sys
args =sys.argv

filename = args[1]
L_star=1000
with open(args[1]) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : | Squared Gradient: ',line)
        L_star= float(li[1]) if float(li[1]) < L_star else L_star
#print(L_star)
with open(args[1]) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : | Squared Gradient: ',line)
        li[1]= str(float(li[1])-L_star)
#        li[2] = li[2].rstrip('0').rstrip('0').rstrip('.')
#        li[3] = li[3].rstrip('\n').ljust(13,'0')
        del li[-1]
        print(' '.join(li))

