import tbm
import sys
import os

args=sys.argv


xij_min = 1000
xij_max = -1

X_arr= []

with open(args[1]) as f:
    for line in f:
        xi = line.split()
        xi_int = [ int(xij)  for xij in xi ]        
        
        xij_min = min(xi_int) if xij_min > min(xi_int) else xij_min
        xij_max = max(xi_int) if xij_max < max(xi_int) else xij_max
        
        X_arr.append(xi_int)

X =[]
for xi in X_arr:   
    X.append(tuple([xij - xij_min for xij in xi]))

#with open(args[1]) as f:
#    for line in f:
#        xi = line.split()
#        xi_int = [ int(xij) - xij_min for xij in xi ]
#        X.append(tuple(xi_int))

n = xij_max-xij_min+1

#print("file ", args[2] , "opened", flush=True)
B = tbm.get_B_from(args[2])
#print("file ", args[2] , "closed", flush=True)
S = list(dict.fromkeys(B+X))
#print("S done")
tbm = tbm.TBM(n,B,S,X)
#print("TBM init done")

if args[3] == "grad":
    tbm.fit(X, 1001, stepsize = float(args[4]) , solver = "grad")
elif args[3] == "coor3":
    tbm.fit(X, 1001, solver = "coor3")
else:
    print("Error")
