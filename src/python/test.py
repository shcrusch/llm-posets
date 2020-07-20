import numpy as np
import random
import time
import sys
import math
import copy
import statistics
import itertools 
def includes(phi,x):
  return set(phi).issubset(x)

class LLM:
  def __init__ (self, n, B, method = 'gen'):
    self.B_ = B # list: [()]
    self.n_ = n
    self.set_S() #list: [()]
    self.set_theta(method)
    self.compute_P() # dict (key=tuple : value=float)
    self.Phi_ = [] # list [phi|  |wfrq(phi,alpha)| > lambda]
    self.wfrq_ = {} # dict {key=phi : value=wfrq(phi, alpha)}
    self.alpha_ = {} # dict {key=xi : value=alpha }
    self.F_ = [] #[()]
    self.Phat_ = {} # dict {key=x : value=Phat[x]}
    self.etahat_ = {} # dict {key=phi : value = etahat_phi}
    self.eta_ = {} # dict {key=phi : value = eta_phi}

    
  def set_S(self):
    self.S_ = []
    for i in range(self.n_-1):
      for x in itertools.combinations(range(self.n_) , i+1):
        self.S_.append(tuple(x))


  def set_theta(self, method):
    random.seed(0)
    self.theta_ = {}
    if method == 'gen':
      for phi in self.B_:
        self.theta_[phi] = random.uniform(-100,100)
      self.compute_Z()
    elif method == 'learn':
      for phi in self.B_:
        self.theta_[phi] = 0.0
      self.Z = len(self.S_)
    else:
      print("Method key Error")
  
  def compute_logP(self, x):
    """
    Computing lopP
        Parameters
        -----
        xi : i th row vector of X
    
        Returns
        -----
        ret : logP
    
        """
    ret = 0.0
    for phi in self.B_:
      if includes(phi, x):
        ret +=  self.theta_[phi]
    ret -= np.log(self.Z)
    return ret   

  def compute_Z(self):
    """
    Computing self.Z

    """
    self.Z = 0.0
    
    for x in self.S_:
      s = 0.0 # sum theta_phi phi(x) for theta in B
      for phi in self.B_:
        if includes(phi, x):
          s += self.theta_[phi]
      self.Z += np.exp(s)

  
  def compute_P(self):
    """
    Computing P
    Returns
    -----
    P_ : len(P_) = len(S_)
    """
    self.P_ = {}
    for x in self.S_:
      self.P_[x] = np.exp( self.compute_logP(x) )
  
  def generate(self):
    return np.random.choice(self.S_, p = list(self.P_.values()), size = 100)
  
  def compute_eta(self):
    for phi in self.B_ :
      self.eta_[phi] = 0.0
      for x in self.S_:
        if includes(phi, x):
          self.eta_[phi] += self.P_[x]
    
  def compute_alpha(self, X):
    self.compute_eta()
    for xi in X:
      c = 0.0
      for phi in self.B_:
        if includes(phi, xi):
          c += self.theta_[phi]
      self.alpha_[xi] = ( np.exp(c)/np.exp(self.Z) - 1) / len(X)


  def set_Phi(self, X, lambda_):
    self.Phi_ = []
    for phi in self.B_:
      wfrq_p = 0.0
      wfrq_m = 0.0
      for xi in X:
        if includes(phi, xi):
          if self.alpha_[xi]>0:
            wfrq_p += self.alpha_[xi]
          else:
            wfrq_m -= self.alpha_[xi]
      
      self.wfrq_[phi] = max([wfrq_p, wfrq_m])

      if max([wfrq_p,wfrq_m]) > lambda_:
        self.Phi_.append(phi)
  
  def compute_Phat(self, X):
    for x in self.S_:
      self.Phat_[x] = 0.0
    for xi in X:
      self.Phat_[xi] += 1 / len(X)

  def compute_etahat(self, X):
    """        
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
            
    """
    for phi in self.B_:
      etahat_phi = 0.0
      for x in self.S_:
        if includes(phi, x):
          etahat_phi += self.Phat_[x]
      self.etahat_[phi] = etahat_phi

  def optim(self,X, B, lambda_, epoch):
    u = {}
    for x in self.S_:
      u[x] = self.P_[x] * self.Z
    
    #start = time.time()
    for epoc in range(epoch):
      LD = 0.0
      for x in X:
        LD -= np.log(u[x])
      
      L1_norm = 0.0
      for phi in B:
        L1_norm += np.abs(self.theta_[phi])
      
      LD += len(X) * np.log(self.Z) + lambda_ * L1_norm

      #print( "L_D:",f'{LD:.16f}'  , flush=True)
      
      #index = range(len(self.B_))
      #index = np.random.RandomState().permutation(B) #permutative
      #index = np.random.randint(0,len(self.B_)-1,len(self.B_))  #random
      for phi in B:

        #phi = B[ index[iter] ]
        etahat_phi = self.etahat_[phi]
        
        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          continue

        #compute eta_phi                                                                                                                  
        eta_phi = 0.0
        for x in self.S_:
          if includes(phi, x):
            eta_phi += u[x]
        eta_phi /= self.Z
        
        if etahat_phi < lambda_:
          exp_delta = min([np.exp(-self.theta_[phi]), (1 - eta_phi) / eta_phi * (etahat_phi + lambda_) / (1 - etahat_phi - lambda_)])

        elif etahat_phi > 1-lambda_:
          exp_delta = max([np.exp(-self.theta_[phi]), (1 - eta_phi) / eta_phi * (etahat_phi - lambda_) / (1 - etahat_phi + lambda_)])
        
        else:
          exp_delta_1 = (1 - eta_phi) / eta_phi * (etahat_phi - lambda_) / (1 - etahat_phi + lambda_)
          exp_delta_2 = (1 - eta_phi) / eta_phi * (etahat_phi + lambda_) / (1 - etahat_phi - lambda_)
          exp_delta = statistics.median([np.exp(-self.theta_[phi]), exp_delta_1, exp_delta_2])

        #update theta_phi
        self.theta_[phi] += np.log(exp_delta)

        #update u,Z
        for x in self.S_:
          if includes(phi, x):
            self.Z += u[x] * (exp_delta - 1)
            u[x] *= exp_delta
    print(LD)


  def grab(self, k, X, lambda_, K, epoch):
    self.compute_Phat(X)
    self.compute_etahat(X)
    self.compute_alpha(X)
    self.set_Phi(X, lambda_)
    self.F_ = []
    while self.Phi_ != []:
      #print(len(self.Phi_))
      for j in range(K):
        if self.Phi_ != []:
          jth = max(self.wfrq_, key = self.wfrq_.get)
          del self.wfrq_[jth]
          self.Phi_.remove(jth)
          self.F_.append(jth)
          self.F_ = list(set(self.F_))

      B = copy.copy(self.F_)
      #print(len(B))
      self.optim(X, B, lambda_, epoch)
      self.compute_P()
      self.compute_alpha(X)
      self.set_Phi(X, lambda_)


n = 10

B = []
for i in range(n):
  if i == 0:
    Bi = []
    for j in range(n-1):
      if np.random.choice([0,1], p = [0.5, 0.5]):
        Bi.append((j,))
  else:
    pre_Bi = Bi
    Bi = []
    for phis in itertools.combinations(pre_Bi,2):
      phi = set(phis[0]) | set(phis[1])
      
      if len(phi) == i+1:
        t = True
        for p in Bi:
          if phi == set(p):
            t = False
            break
        if t and np.random.choice([0,1], p = [0.7, 0.3]):
          Bi.append(tuple(phi))
    if len(Bi) == 0:
      break
  B += Bi

print(B)



llm = LLM(n, B, method = 'gen')
X = llm.generate()

k = 2
B_test = []
for i in range(k):
  if i == 0:
    Bi = []
    for j in range(n-1):
      Bi.append((j,))
  else:
    pre_Bi = Bi
    Bi = []
    for phis in itertools.combinations(pre_Bi,2):
      phi = set(phis[0]) | set(phis[1])
      
      if len(phi) == i+1:
        t = True
        for p in Bi:
          if phi == set(p):
            t = False
            break
        if t and np.random.choice([0,1], p = [0.5, 0.5]):
          Bi.append(tuple(phi))
    if len(Bi) == 0:
      break
  B_test += Bi
lambda_ = 0.2
K = 5
epoch = 1000
llm_test = LLM(n, B_test, method = 'learn')
llm_test.grab(k, X, lambda_,K, epoch)

