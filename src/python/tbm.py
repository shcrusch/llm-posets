import numpy as np
import time
import sys
 
def includes(phi,x):
  return set(phi).issubset(x)

class TBM:
  def __init__ (self, n, B, S, X):
    
    self.set_B(B) # list of tuples
    self.set_S(S) # list of tuples
    self.n_= n
    self.set_Phi(self.S_, self.B_) # dict of lists
    self.set_Ssub(self.S_, self.B_) # dict of lists


    self.init_theta() # dict (key: list, val: double)

    self.compute_Phat(X) # dict (key: list, val: double)
    self.compute_etahat(X) # dict (key: list, val: double)
    
  def fit(self, X, n_iter, stepsize=-1, solver="grad"):
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
    
    if solver == "grad":
      self.gradient_descent(X, n_iter, stepsize)
    elif solver == "coor":
      self.coordinate_descent(X, n_iter)
    elif solver == "stoc":
      self.stochastic_gradient_descent(X, n_iter, stepsize)
    else:
      print("Solver Option Error", file = sys.stderr)


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
  
    
  def compute_Phat(self, X):
    self.Phat_ = np.zeros(len(self.S_))
    for xi in X:
      self.Phat_[self.invS_[xi]] += 1 / len(X)


  def compute_etahat(self, X):
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
      for xi in X:
        denom += 1 if includes(phi, xi) else 0
      self.etahat_[phi] = denom / len(X)


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
    self.invS_ = {}
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
    Computing theta_perp and set it to self.theta_[invS_[()]]    

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

  def compute_eta_pP(self):
    self.eta_pP = {}
    for phi in self.B_:
      for Phi in self.B_:
        self.eta_pP[phi,Phi] = 0.0
        for x in list(dict.fromkeys(self.Ssub_[phi]+self.Ssub_[Phi])):
          self.eta_pP[phi,Phi] += self.P_[self.invS_[x]]

  def compute_Hess(self):
    Hess = []
    L =[]
    for phi in self.B_:
      if phi != ():
        hess = []
        for Phi in self.B_:
          if Phi != ():
            hess.append(self.eta_pP[phi,Phi] - self.eta_[phi]*self.eta_[Phi])
            L.append(self.eta_pP[phi,Phi] - self.eta_[phi]*self.eta_[Phi]) if phi == Phi
        Hess.append(hess)

    return Hess,L

  def compute_prox_KL(self):
    """
    Assuming support of P to be D,
    Compute prox KL_divergence normalized by sum_{x in D} exp(-E(x)) 
    
    """
    noramlized_constant = 0.0
    for x in list(dict.fromkeys(X)):
      Px = self.P_[self.invS_[x]]
      normalized_constant += Px

    ret = 0.0
    for x in self.D:
      Phatx = self.Phat_[self.invS_[x]]
      normalized_Px = self.P_[self.invS_[x]] / normalized_constant 
      ret += Phatx * ( np.log(Phatx) - np.log(normalized_Px) ) 

    return ret
  def compute_KL(self):
    """
    Computing KL_divergence sum_{x in S} Phat(x) ( ln Phat(x)  - ln P(x) )
    
    """
    ret = 0.0
    for x in self.S_:
      Phatx = self.Phat_[self.invS_[x] ]
      if Phatx != 0.0:
        ret += Phatx * (np.log(Phatx) - np.log(self.P_[self.invS_[x]]))

        
    return ret

  def compute_squared_gradient(self):
    ret = 0.0
    for phi in self.B_:
      if phi: # is not empty
        ret += (self.etahat_[phi] - self.eta_[phi] ) ** 2
    return ret

  def gradient_descent(self, X, n_iter, step):  
    """
    Actual implementation gradient_descent
    Parameters 
    -----
    X : array-like, shape (n_samples,)
            Training vectors, where n_samples is the number of samples and 
            each element is a tuple of indices of 1s. 
    n_iter
    step 
    """
#    print(len(self.B_))
    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      self.compute_eta_pP()
      Hess,L = self.compute_Hess()
      print("L_max="+str(max(L)))
      a,v =np.linalg.eig(Hess)
      print(min(a))
      kl=self.compute_KL()
#      prox_kl=self.compute_prox_KL()
      sg=self.compute_squared_gradient()      

      print(iter ,":", "KL divergence: ", f'{kl:.8f}' ," time : %4.2f"% (time.time()-start), "Squared Gradient:",f'{sg:.10f}')
      for phi in self.B_:
        if phi: # is not empty
          new_theta_phi = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi] )
          self.set_theta(phi, new_theta_phi)
      self.compute_theta_perp()
      
  
  def coordinate_descent(self, X, max_epoch):
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

    Z = len(self.S_)

    start = time.time()
    for epoch in range(max_epoch):
      self.compute_P()
      self.compute_eta()      
      self.compute_eta_pP()
      Hess,L = self.compute_Hess()
      print("L_max="+str(max(L)))
      #print( np.linalg.det(Hess))
      a,v = np.linalg.eig(Hess)
      print(min(a))
      print("epoch ", epoch,  " compute_KL", flush=True, file = sys.stderr)
      kl = self.compute_KL()
      print("epoch ", epoch,  " compute_SG", flush=True, file = sys.stderr)
      sg = self.compute_squared_gradient()

      prev_kl = kl

      print(epoch ,":",  "KL divergence:",f'{kl:.8f}' ," time : %4.2f"% (time.time()-start), "Squared Gradient:",f'{sg:.10f}', flush=True)
      index = np.random.RandomState().permutation(range(len(self.B_)-1)) 
      print("epoch ", epoch,  " iteration start", flush=True, file = sys.stderr)
      for iter in range(len(self.B_) - 1):
        phi = self.B_[ index[iter] + 1 ] # all phi excluding perp

        etahat_phi = self.etahat_[phi] 
        if etahat_phi == 1.0 or etahat_phi == 0.0:
          continue
        eta_phi = 0.0
        for x in self.Ssub_[phi]:
            eta_phi += u[x]
        eta_phi /= Z
                
        exp_delta = (1-eta_phi)/eta_phi * etahat_phi/(1-etahat_phi)

        self.theta_[self.invB_[phi]] += np.log(exp_delta)

        for x in self.Ssub_[phi]:
          u[x] *= exp_delta

        Z = np.exp(-self.theta_[self.invB_[()]])
        for x in self.Ssub_[phi]:
          Z += u[x] * (1 - 1/exp_delta)

        self.set_theta( (), -np.log(Z) )
      print("epoch ", epoch,  " iteration end", flush=True, file = sys.stderr)  

"""
  def stochastic_gradient_descent(self, X, n_iter, step):
                                                                                                                                                                                                     
    Actual implementation gradient_descent                                                                                                                                                              
    Parameters                                                                                                                                                                                          
    -----                                                                                                                                                                                               
    X : array-like, shape (n_samples,)                                                                                                                                                                  
            Training vectors, where n_samples is the number of samples and                                                                                                                              
            each element is a tuple of indices of 1s.                                                                                                                                                   
    max_epoch                                                                                                                                                                                           
    step                                                                                                                                                                                                
    
    Phi_x = []    
    for phi in self.B_:
      if phi:
        phi_x = []
        for xi in X:
          if includes(phi,xi):
            phi_x.append(1)
          else:
            phi_x.append(0)
        Phi_x.append(phi_x)

    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
#      prox_kl=self.compute_prox_KL()                                               
      sg=self.compute_squared_gradient()

      print(iter ,":", "KL divergence: ", f'{kl:.8f}' ," time : %4.2f"% (time.time()-start), "Squared Gradient:",f'{sg:.10f}')
      index = np.random.RandomState().permutation(range(len(X)))
      for i in range(len(X)):
        for phi in self.B_:
          if phi: # is not empty                                                  
            new_theta_phi = self.get_theta(phi) + step * (Phi_x[### - self.eta_[phi] )
            self.set_theta(phi, new_theta_phi)
        self.compute_theta_perp()



  def grad_check(self, g):    

    # generate_z
    z =np.rand((1,num_theta))
    
    i = 0
    for phi in range():
      set_theta(phi, get_theta(phi) + e* z[i])
    
    kl_perturbed = compute_KL()


    for phi in range():
      set_theta(phi, get_theta(phi) - e* z[i])


    inner =0.0
    for i in range():
      inner += g[i] * z[i]

    kl_approx = compute_KL + e * inner

    print(...)
"""
