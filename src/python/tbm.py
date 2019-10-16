import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
 
def get_B_from(filename):
  # file format:
  # each line is a frequent itemset with 1-start
  # ignore the first line that represents "bottom"
  
  Bfile = pd.read_csv(filename, header=None, sep=' ',names=[0,1])

  B1=[(),]
  B2=[]
  for i in range(len(Bfile)):
    if pd.isnull(Bfile.at[i,1]): 
      B1.append((Bfile.loc[i,0]-1,)) 

  for j in range(len(Bfile)):
    if pd.isnull(Bfile.at[j,1])==0: 
      B2.append((Bfile.loc[j,0]-1,int(Bfile.loc[j,1]-1)))
  B1.sort()
  B2.sort()
  B=B1+B2

  return B

def includes(phi,x):
  return set(phi).issubset(x)

class TBM:
  def __init__ (self, n, B, S, X):
    
    self.set_B(B)
    self.set_S(S)
    self.n_= n
    self.set_Phi(self.S_, self.B_)
    self.set_Ssub(self.S_, self.B_)
    self.set_Xsub(X)
  def fit(self, X, n_iter, stepsize=-1,  absd = -1, solver="grad"):
    """Actual implementation of TBM fitting.
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
        n_iter: number of iteration
        method: "grad"  = gradient descent, "coor" coordinate descent
        Returns
        -------
        self : object
    """
    self.compute_Phat(X)
    self.set_etahat(X)
    if solver == "grad":
      self.gradient_descent(X, n_iter, stepsize)
    elif solver == "coor3":
      self.coordinate_descent3(X, n_iter)
    else:
      print("Solver Option Error")

    
    return self 
  
  def set_Phi(self, S, B):
    """
    Phi_[x]= { phi in B | x includes phi }
    """
    self.Phi_ = {}
    for x in S:
      Phix = []
      for phi in B:
        if includes(phi,x):
          Phix.append(phi)
      self.Phi_[x] = Phix

  def set_Ssub(self, S, B):
    """
    Ssub [phi]= { x in S | x includes phi } 
    """
    self.Ssub_ = {}
    for phi in B:
      ssub = []
      for x in S:
        if includes(phi,x):
          ssub.append(x)
      self.Ssub_[phi] = ssub
  
  def set_Xsub(self, X):
    self.Xsub = list(dict.fromkeys(X))
    self.invXsub = {}
    for i in range(len(self.Xsub)):
      self.invXsub[self.Xsub[i]] = i
    
  def compute_Phat(self, X):
    self.Phat = np.zeros(len(self.Xsub))
    for xi in X:
      self.Phat[self.invXsub[xi]] += 1 / len(X)
    

  def set_etahat(self, X):
    """        
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
            
    """
    self.etahat_ = {}
    for phi in self.B_:
      denom = 0.0
      for i in range(len(X)):
        denom += 1 if includes(phi, X[i]) else 0
      self.etahat_[phi] = denom / len(X)


  def compute_ll(self, X):
    """
      Compute log likelihood
      Parameters
      ----------
      X : array-like, shape = [n_samples, n_features]

      Returns
      ----------
      ret :  log-likelihood

    """
    ret = 0.0
    for xi in X:
      ret += self.compute_logP(xi)
    return ret

  def set_B(self, B):
    """
    Generate B_ and invB_
    Parameters
    -----
    B : Set of parameters
    
    invB_ : dict type
               indices are vectors of set_B
    """
    self.B_ = B
    self.invB_ ={}
    for i in range(len(B)):
      self.invB_[B[i]] = i
      

  def set_S(self, S):
    """
    Generate S_ and invS_
    Parameters
    -----
    S : Sample space
    
    inv_S : dict type
               indices are vectors of set_S
    """
    self.S_ = S
    self.invS_ ={}
    for i in range(len(S)):
      self.invS_[S[i]] = i

  def compute_logP(self, xi):
    """
    Computing lopP
    Parameters
    -----
    xi : i th row vector of X
    
    Returns
    -----
    ret : logP
    
    """
    ret = 0.0;
    for phi in self.B_ :
      ret +=  self.get_theta(phi) if includes(phi, xi) else 0    
    return ret 

  def get_theta(self, phi):
    """
    Getting theta
    Parameters 
    -----
    phi : element of set_B
    
    Returns
    -----
    ret : theta
    """
    return self.theta_[self.invB_[phi]]
  def set_theta(self, phi, value):
    
    self.theta_[self.invB_[phi]] = value

    
  def init_theta(self):
    
    self.theta_ = np.zeros(len(self.B_))
    self.compute_theta_perp()    
  
  def compute_theta_perp(self):
    """
    Computing theta_perp
    
    Returns
    -----
    theta_
    """
    r = 0.0 
    for x in self.S_:
      s = 0.0 # sum theta_phi phi(x) for theta in B minus perp
      for phi in self.Phi_[x]:
        if phi: # is not empty
          s += self.get_theta(phi)  
      r += np.exp(s)
    self.theta_[self.invB_[()]] = -np.log(r)
  
  def compute_P(self):
    """
    Computing P
    Returns
    -----
    P_ : len(P_) = len(S_)
    """
    self.P_ = np.zeros(len(self.S_))
    for x in self.S_:
      self.P_[self.invS_[x]] = np.exp( self.compute_logP(x) )

  def compute_eta(self):
    """
    Computing eta
    Returns
    -----
    eta_ : dict type
             len(eta_) = len(B_)
    """
    self.eta_ = {}
    for phi in self.B_ :
      self.eta_[phi] = 0.0
      for x in self.Ssub_[phi] :
        self.eta_[phi] += self.P_[self.invS_[x]]
  
  def compute_KL(self):
    """
    Computing KL_divergence
    
    Parameter
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    -----
    """
    ret  = 0.0
    for x in self.Xsub:
      ret += self.Phat[self.invXsub[x]] * (np.log(self.Phat[self.invXsub[x]]) - self.compute_logP(x))    
    return ret

  def compute_squared_gradient(self):
    ret = 0.0
    for phi in self.B_:
      if phi: # is not empty
        ret += (self.etahat_[phi] - self.eta_[phi] ) ** 2
    return ret

  def gradient_descent(self, X, max_epoch, step):  
    """
    Actual implementation gradient_descent
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    max_epoch
    step 
    """
    
    self.init_theta()
    start = time.time()
    for epoch in range(max_epoch):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
      sg=self.compute_squared_gradient()
      
      print(epoch ,":",  "KL divergence:",f'{kl:.8f}' ," time : %4.2f"% (time.time()-start), "Squared Gradient:",f'{sg:.10f}')
      for phi in self.B_:
        if phi: # is not empty
          new_theta_phi = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi] )
          self.set_theta(phi, new_theta_phi)
      self.compute_theta_perp()
      
  
  def coordinate_descent3(self, X, max_epoch):
    """
    Actual implementation coodinate_descent
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s.
    max_epoch
    -----
    
    """
    u = {}
    for x in self.S_:
      u[x] = 1.0
    self.theta_ = np.zeros(len(self.B_))
    self.theta_[self.invB_[()]] = -np.log(len(self.S_))
    start = time.time()
    for epoch in range(max_epoch):
      self.compute_P()
      self.compute_eta()

      kl = self.compute_KL()
      sg = self.compute_squared_gradient()
      print(epoch ,":",  "KL divergence:",f'{kl:.8f}' ," time : %4.2f"% (time.time()-start), "Squared Gradient:",f'{sg:.10f}')
      index = np.random.RandomState(seed=2019).permutation(range(len(self.B_)-1)) 

      for iter in range(len(self.B_) -1):
        phi = self.B_[index[iter] +1 ] 
        etahat_phi = self.etahat_[phi] 
        if etahat_phi == 1.0 or etahat_phi == 0.0:
          continue
        eta_phi = 0.0
        for x in self.Ssub_[phi]:
            eta_phi += u[x]
        eta_phi *= np.exp(self.theta_[self.invB_[()]])
        
        sigma = eta_phi / np.exp(self.theta_[self.invB_[()]])
        tau = (1-eta_phi) / np.exp(self.theta_[self.invB_[()]])
        
        delta = np.log(tau/sigma * etahat_phi/(1-etahat_phi))
        #print(delta)
        
        self.theta_[self.invB_[phi]] += delta

        for x in self.Ssub_[phi]:
          u[x] *= np.exp(delta)

        Z = np.exp(-self.theta_[self.invB_[()]])
        for x in self.Ssub_[phi]:
          Z += u[x] * (1 - np.exp(-delta))
        self.set_theta( (), -np.log(Z) )
