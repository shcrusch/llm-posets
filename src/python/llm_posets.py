import numpy as np
import time
import sys
import math
import copy
import itertools 
import random
def includes(phi,x):
  return set(phi).issubset(set(x))

class LLM:
  def __init__ (self, n, B, S):

    self.B_ = B # list of tuples
    self.S_ = S # list of tuples
    self.n_= n
    self.set_Bsub(self.S_, self.B_) #dict (key: tuple, val: list of tuples)
    self.set_Ssub(self.S_, self.B_)  # dict (key: tuple, val: list of tuples)
    self.init_theta() # dict (key: list, val: float)
    self.P_ = {}
    self.Phat_ = {}  # dict (key: list, val: float)
    self.etahat_ = {}  # dict (key: list, val: float)
    
    
    
  def fit(self, X, n_iter, stepsize=-1, mu = 0, lambda_ = -1, solver="grad"):
    """Actual implementation of LLM on posets fitting.
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
    self.compute_etahat(X)
    if solver == "grad":
      self.gradient_descent(X, n_iter, stepsize)
    elif solver == "coor":
      self.coordinate_descent(X, n_iter)
    elif solver == "acc_grad":
      self.accelerated_gradient_descent(X, n_iter, stepsize, mu)
    else:
      print("Solver Option Error", file = sys.stderr)


    return self 
  

  def set_Bsub(self, S, B):
    """
    Phi_[x]= { phi in B | x includes phi }
    """
    self.Bsub_ = {():[]}

    for x in S:
      B_x = []
      for phi in B:
        if includes(phi,x):
          B_x.append(phi)
      self.Bsub_[x] = B_x



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
      for xi in self.Ssub_[phi]:
        etahat_phi += self.Phat_[xi] 
      self.etahat_[phi] = etahat_phi


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
    for phi in self.Bsub_[x]:
      if phi != ():
        ret +=  self.theta_[phi]
    ret += self.theta_perp
    return ret 


  def init_theta(self):    
    self.theta_ = {}
    for phi in self.S_:
      if phi != ():
        self.theta_[phi] = 0.0
    self.compute_theta_perp()   

  def compute_theta_perp(self):
    """
    Computing self.theta_perp

    """

    r = 0.0 
    for x in self.S_:
      s = 0.0 # sum theta_phi phi(x) for theta in B
      for phi in self.Bsub_[x]:
        if phi != ():
          s += self.theta_[phi]
      r += np.exp(s)
    self.theta_perp = -np.log(r)
  
  def compute_P(self):
    """
    Computing P
    Returns
    -----
    P_ : len(P_) = len(S_)
    """
    for x in self.S_:
      self.P_[x] = np.exp( self.compute_logP(x) )

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
        self.eta_[phi] += self.P_[x]

  def compute_KL(self):
    """
    Computing KL_divergence sum_{x in S} Phat(x) ( ln Phat(x)  - ln P(x) )
    
    """
    ret = 0.0
    for x in self.S_:
      Phatx = self.Phat_[x]
      if Phatx != 0.0:
        ret += Phatx * np.log(Phatx / self.P_[x] )

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

    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
        
      for phi in self.B_:
        new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi] )
        self.theta_[phi] = new_theta_phi

      self.compute_theta_perp()
  



  def update_accelerated_theta(self, iter, step, mu):      
    
    if iter == 0:

      self.grad_list = [] 
      # [theta(t) - step*grad( L_D( theta(t)) ), theta_(t-1) - step* ( L_D(theta(t-1)) ) ]
      self.grad_theta_ = {}
      # theta^(t-1) - step * grad( L_D( theta^(t-1) ) )
      self.lambda_list = [0,1]
      self.lambda_ = 1
      for phi in self.B_:
        grad_new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi] )
        self.grad_theta_[phi] = grad_new_theta_phi
        self.theta_[phi] = grad_new_theta_phi
      self.grad_list.append(self.grad_theta_)
      
    elif iter == 1:
      pre_grad_theta_ = copy.copy(self.grad_theta_)
      pre_lambda_ = copy.copy(self.lambda_)
      self.lambda_ = (1 + math.sqrt( 1+ 4 * (pre_lambda_**2)))/2
      for phi in self.B_:
        grad_new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi] )
        self.grad_theta_[phi] = grad_new_theta_phi
        if mu == 0: #no hyperparameter
          new_theta_phi = grad_new_theta_phi   
        else:
          new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[phi]

        self.theta_[phi] = new_theta_phi
      self.grad_list = [pre_grad_theta_,self.grad_theta_]
      self.lambda_list = [pre_lambda_,self.lambda_]
    else:
      pre_grad_theta_ = copy.copy(self.grad_theta_)
      pre_lambda_ = copy.copy(self.lambda_)
      self.lambda_ = (1 + math.sqrt(1 + 4 * (pre_lambda_**2)))/2
      self.gamma = (1 - pre_lambda_) / self.lambda_

      for phi in self.B_:
        grad_new_theta_phi = self.theta_[phi] + step * (self.etahat_[phi] - self.eta_[phi])
        self.grad_theta_[phi] = grad_new_theta_phi
        if mu ==0: #no hyperparameter
          new_theta_phi = (1 -self.gamma) * grad_new_theta_phi + self.gamma * pre_grad_theta_[phi] 
        else:
          new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[phi]

        self.theta_[phi] = new_theta_phi
        
      self.grad_list = [pre_grad_theta_,self.grad_theta_]
      self.lambda_list = [pre_lambda_,self.lambda_]

  def accelerated_gradient_descent(self, X, n_iter, step, mu):
    """                                                                                                     
    Actual implementation accelerated_gradient_descent                                                          
    Parameters                                                                                             
    -----                                                                                                   
    X : array-like, shape (n_samples,)                                                                       
            Training vectors, where n_samples is the number of samples and                                   
            each element is a tuple of indices of 1s.                                                       
    n_iter                                                                                                   
    step                                                                                                      
    mu : momentum
    """

    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
      self.update_accelerated_theta(iter, step ,mu)

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
      
      #compute KL
      kl = 0.0
      for x in self.S_:
        if self.Phat_[x] != 0.0:
          kl += self.Phat_[x] * np.log(self.Phat_[x] / u[x])
      kl += np.log(Z)
      print(epoch ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
      
      
      random.shuffle(self.B_)
      for phi in self.B_:
        etahat_phi = self.etahat_[phi] 

        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          
          continue

        #compute eta_phi
        eta_phi = 0.0
        for x in self.Ssub_[phi]: 
            eta_phi += u[x]
        eta_phi /= Z
        
        #compute exp(delta)
        exp_delta = 1 + (etahat_phi - eta_phi) / eta_phi / (1 - etahat_phi)
        delta = np.log1p( (etahat_phi - eta_phi) / eta_phi / (1 - etahat_phi) )
        


        #update theta_phi

        self.theta_[phi] += delta

        #update u,Z

        for x in self.Ssub_[phi]:

          Z += u[x]* (etahat_phi - eta_phi) / eta_phi / (1 - etahat_phi)  #u[x]*(exp_delta - 1)
          u[x] *= exp_delta

