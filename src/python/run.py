import llm_posets
import sys
import itertools
args=sys.argv

xij_min = 1000
xij_max = -1

X_arr= []
dataset_file = '../../dataset/'+ args[1] + '.dat'
itemset_file = '../../dataset/'+ args[1] + '.dat_itemsets'
X = []
with open(dataset_file) as f:
    for line in f:
        xi = line.split()
        xi_int = [ int(xij)  for xij in xi ]        
        xi_int.sort()

        X.append(tuple(xi_int))


def get_B_from(filename):
    # generate itemset B with itemset file
    # ignore "bottom" {()}                             
    B = []
    with open(filename) as f:
        for line in f:
            phi = line.split()
            phi_int = [int(x) for x in phi]
            phi_int.sort()
            B.append(tuple(phi_int))
    if () in B:
        B.remove(())
    return B

B = get_B_from(itemset_file)

# Generate support of P
S = list(dict.fromkeys(X + B + [()]))


llm = llm_posets.LLM(B,S)


if args[2] == "grad":
    llm.fit(X, n_iter = int(args[4]), stepsize = float(args[3]) , solver = "grad")
elif args[2] == "coor":
    llm.fit(X, n_iter = int(args[3]), solver = "coor")
elif args[2] == "acc_grad":
    llm.fit(X, n_iter = int(args[5]), stepsize = float(args[3]) , mu = float(args[4]), solver = "acc_grad")

else:
    print("Error")