import tbm
import sys
import os

args=sys.argv

X= []
m= 1000
n= -1

with open(args[1]) as f:
    for line in f:
        xi =line.split()
        xi_int = [ int(xij)  for xij in xi ]        
        if m > min(xi_int):
            m = min(xi_int)

        if n < max(xi_int):
            n = max(xi_int)

with open(args[1]) as f:
    for line in f:
        xi = line.split()
        xi_int = [ int(xij) - m for xij in xi ]
        X.append(tuple(xi_int))
n = n-m+1        

B = tbm.get_B_from(args[2])

S = list(dict.fromkeys(B+X))

tbm = tbm.TBM(n,B,S,X)

if args[3] == "grad":
    tbm.fit(X,1000, stepsize = float(args[4]) , solver = "grad")
elif args[3] == "coor":
    tbm.fit(X,1000, absd = int(args[4]) , solver = "coor")
elif args[3] == "coor3":
    tbm.fit(X, 1000, solver = "coor3")
else:
    print("Error")
