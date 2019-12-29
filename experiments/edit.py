import re
import sys
args =sys.argv

filename = args[1]

with open(args[1]) as f:
    for line in f:
        li = re.split(' : KL divergence: |  time : | Squared Gradient: ',line)
        li[1] = li[1].rstrip('0')
        li[2] = li[2].rstrip('0').rstrip('0').rstrip('.')
        li[3] = li[3].rstrip('\n').ljust(13,'0')
        print(' '.join(li))

