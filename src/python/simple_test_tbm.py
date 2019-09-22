import tbm

X =[ (0,1,2), (2,4), (4,) ]

# S = []
# for num in range(2 ** 5):
#   v = []
#   for i in range(5):
#     if num % 2 == 1:
#       v.append(i)
#     num = num // 2
#   S.append(tuple(v))
B = [(),(2,),(4,),(2,4)]
S = list(dict.fromkeys(B+X))
#print(S)
#print(B)

t = tbm.TBM(5, B, S)
t.set_etahat(X)
#print(tbm.etahat_)

t.set_Phi(S,B)
t.set_Ssub(S,B)
t.init_theta()
t.compute_KLdivergence(X)

t.compute_P()
t.compute_eta()
#print(tbm.eta_)

t.fit(X,10,0.01)

