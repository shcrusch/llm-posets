import llm_posets
import sys
import pandas as pd
import itertools
args=sys.argv

xij_min = 1000
xij_max = -1

X_arr= []

dataset_file = '/home/hayashi/workspace/tbm-python/dataset/'+ args[1] + '.dat'
itemset_file = '/home/hayashi/workspace/tbm-python/dataset/'+ args[1] + '.dat_itemsets'

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


def get_B_from(filename):
  # file format:                                                             
  # each line is a frequent itemset with 1-start                               
  # ignore the first line that represents "bottom"                             
  Bfile = pd.read_csv(filename, header=None, sep=' ',names=[0,1])

  B1=[]
  B2=[]
  for i in range(len(Bfile)):
    if pd.isnull(Bfile.at[i,1]):
      B1.append((Bfile.loc[i,0]-xij_min,))
  for j in range(len(Bfile)):
    if pd.isnull(Bfile.at[j,1])==0:
      B2.append((Bfile.loc[j,0]-xij_min,int(Bfile.loc[j,1]-xij_min)))
  B1.sort()
  B2.sort()

  B=B1+B2

  return B

B = get_B_from(itemset_file)
#B = []
#for i in range(n):
#    B.append((i,))

#for phi in itertools.combinations(range(n), 2):
#    B.append(phi)

S = list(dict.fromkeys(B+X+[()]))
#B = list(dict.fromkeys(B+X))

llm = llm_posets.LLM(n,B,S,X)

if args[2] == "grad":
    llm.fit(int(args[4]), stepsize = float(args[3]) , solver = "grad")
elif args[2] == "da":
    llm.fit(int(args[4]), stepsize = float(args[3]) , solver = "da")
elif args[2] == "coor":
    llm.fit(int(args[3]), solver = "coor")
elif args[2] == "acc_grad":
    llm.fit(int(args[4]), stepsize = float(args[3]) , mu = float(args[5]), solver = "acc_grad")
elif args[2] == "newton":
    llm.fit(int(args[4]), stepsize = float(args[3]) , solver = "newton")
elif args[2] == "coor_l1":
    llm.fit(int(args[4]), lambda_ = float(args[3]) , solver = "coor_l1")
elif args[2] == "prox":
    llm.fit(int(args[5]), stepsize = float(args[3]) , lambda_ = float(args[4]), solver = "prox")
elif args[2] == "rda":
    llm.fit(int(args[5]), stepsize = float(args[3]) , lambda_ = float(args[4]), solver = "rda")
else:
    print("Error")

