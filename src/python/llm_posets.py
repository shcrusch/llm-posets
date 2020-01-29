import numpy as np
import time
import sys
import math
import copy
 
def includes(phi,x):
  return set(phi).issubset(x)

class LLM:
  def __init__ (self, n, B, S, X):
    
    self.set_B(B) # list of tuples
    self.set_S(S) # list of tuples
    self.n_= n
    self.set_Phi(self.S_, self.B_) #dict (key: tuple, val: list of tuples)
    self.set_Ssub(self.S_, self.B_)  # dict (key: tuple, val: list of tuples)
    self.init_theta() # dict (key: list, val: double)

    self.compute_Phat(X) # dict (key: list, val: double)
    self.compute_etahat(X) # dict (key: list, val: double)
    

  def fit(self, X, n_iter, stepsize=-1, mu = 0, solver="grad"):
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
    
    if solver == "grad":
      self.gradient_descent(X, n_iter, stepsize)
    elif solver == "coor":
      self.coordinate_descent(X, n_iter)
    elif solver == "acc_grad":
      self.accelerated_gradient_descent(X, n_iter, stepsize, mu)
    elif solver == "acc_coor":
      self.accelerated_coordinate_descent(X, n_iter)
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
    ret += self.theta_perp
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
    Computing self.theta_perp

    """

    r = 0.0 
    for x in self.S_:
      s = 0.0 # sum theta_phi phi(x) for theta in B
      for phi in self.Phi_[x]:
        s += self.get_theta(phi)  
      r += np.exp(s)
    self.theta_perp = -np.log(r)
  
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
    Computing KL_divergence sum_{x in S} Phat(x) ( ln Phat(x)  - ln P(x) )
    
    """
    ret = 0.0
    for x in self.S_:
      Phatx = self.Phat_[self.invS_[x]]
      if Phatx != 0.0:
        ret += Phatx * (np.log(Phatx) - np.log(self.P_[self.invS_[x]]))

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
        new_theta_phi = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi] )
        self.set_theta(phi, new_theta_phi)

      self.compute_theta_perp()
      

  def update_accelerated_theta(self, iter, step, mu):      
    
    if iter == 0:

      self.grad_list = [] # [theta(t) - step*grad( L_D( theta(t)) ), theta_(t-1) - step* ( L_D(theta(t-1)) ) ]
      self.grad_theta_ = np.zeros(len(self.B_)) # theta^(t-1) - step * grad( L_D( theta^(t-1) ) )
      self.lambda_list = [0,1]
      self.lambda_ = 1
      for phi in self.B_:
        grad_new_theta_phi = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi] )
        self.grad_theta_[self.invB_[phi]] = grad_new_theta_phi
        self.set_theta(phi, grad_new_theta_phi)
      self.grad_list.append(self.grad_theta_)
      
    elif iter == 1:
      pre_grad_theta_ = copy.copy(self.grad_theta_)
      pre_lambda_ = copy.copy(self.lambda_)
      self.lambda_ = (1 + math.sqrt( 1+ 4 * (pre_lambda_**2)))/2
      for phi in self.B_:
        grad_new_theta_phi = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi] )
        self.grad_theta_[self.invB_[phi]] = grad_new_theta_phi
#        new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[self.invB_[phi]]
        new_theta_phi = grad_new_theta_phi
        self.set_theta(phi, new_theta_phi)
      self.grad_list = [pre_grad_theta_,self.grad_theta_]
      self.lambda_list = [pre_lambda_,self.lambda_]
    else:
      pre_grad_theta_ = copy.copy(self.grad_theta_)
      pre_lambda_ = copy.copy(self.lambda_)
      self.lambda_ = (1 + math.sqrt(1 + 4 * (pre_lambda_**2)))/2
      self.gamma = (1 - pre_lambda_) / self.lambda_

      for phi in self.B_:
        grad_new_theta_phi = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi])
        self.grad_theta_[self.invB_[phi]] = grad_new_theta_phi
#        new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[self.invB_[phi]]
        new_theta_phi = (1 -self.gamma) * grad_new_theta_phi + self.gamma * pre_grad_theta_[self.invB_[phi]]
        self.set_theta(phi, new_theta_phi)
        
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
    self.P_ = np.zeros(len(self.S_))
    start = time.time()
    for epoch in range(max_epoch):

      for x in self.S_:         #compute P
        self.P_[self.invS_[x]] = u[x]/Z
      
      kl = self.compute_KL()
      print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),  flush=True)

      index = np.random.RandomState().permutation(range(len(self.B_))) #permutative
#      index = np.random.randint(0,len(self.B_)-1,len(self.B_))  #random
#      index = range(len(self.B_))

      for iter in range(len(self.B_)):
        phi = self.B_[ index[iter] ] 
        etahat_phi = self.etahat_[phi] 

        if etahat_phi == 1.0 or etahat_phi == 0.0:
          continue

        eta_phi = 0.0
        for x in self.Ssub_[phi]: #compute eta_phi
            eta_phi += u[x]
        eta_phi /= Z
                
        exp_delta = (1-eta_phi)/eta_phi * etahat_phi/(1-etahat_phi) #compute exp(delta)

        self.theta_[self.invB_[phi]] += np.log(exp_delta) #update theta_phi

        Z = np.exp(-self.theta_perp) #update Z
        for x in self.Ssub_[phi]:
          Z += u[x] * (exp_delta - 1)

        self.theta_perp = - np.log(Z)

        for x in self.Ssub_[phi]: #update u
          u[x] *= exp_delta
      



  def update_parameters(self,epoch,iter): #update parameters for ACDM
    n = len(self.B_)
    if epoch == 0 and iter == 0:
      
      self.y = np.zeros(n)
      self.theta_list = [np.zeros(n),np.zeros(n)]

    else:

      for phi in self.B_:
        self.y[self.invB_[phi]] = (n-1) / n * self.alphas[n*epoch+iter] / self.alphas[n*epoch+iter-1] * self.y[self.invB_[phi]] - self.alphas[n*epoch+iter] * (1/self.alphas[n*epoch+iter-1] - 1/n) *self.theta_list[0][self.invB_[phi]] + (1 - self.alphas[n*epoch+iter]/n - self.alphas[n*epoch+iter]/n/self.alphas[n*epoch+iter-1])* self.theta_[self.invB_[phi]]

#        self.y[self.invB_[phi]] = self.theta_[self.invB_[phi]]
    

  def accelerated_coordinate_descent(self, X, max_epoch):
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
    
    alphas = []
    a = 1
    n = len(self.B_)
    for _ in range(max_epoch*n):
      alphas.append(a)
      a = (- a**2 / n + math.sqrt( a**4 / n**2 + 4 * a**2) )/2
    self.alphas = alphas

    start = time.time()
    
    for epoch in range(max_epoch):
#      print(self.theta_)
      self.P_ = np.zeros(len(self.S_))
      for x in self.S_:
        self.P_[self.invS_[x]] = u[x]/Z
      
      kl = self.compute_KL()
      print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),  flush=True)
      index = np.random.RandomState().permutation(range(n)) #permutative
#      index = np.random.randint(0,n-1, n)  #random
#      index = range(n)
      for iter in range(n):

        pre_theta_ = copy.copy(self.theta_)
#        print(self.theta_)
        self.update_parameters(epoch,iter)
        self.theta_ = copy.copy(self.y)
        

        self.compute_theta_perp()
        Z = np.exp(- self.theta_perp)

        
        for x in self.S_:
          r = 0.0
          for phi in self.B_ :
            r +=  self.get_theta(phi) if includes(phi, x) else 0
          u[x] = np.exp(r)


        phi = self.B_[ index[iter] ]
        etahat_phi = self.etahat_[phi]

        if etahat_phi == 1.0 or etahat_phi == 0.0:
          continue
          
        eta_phi = 0.0
        for x in self.Ssub_[phi]: #compute eta_phi                                                                                      
          eta_phi += u[x]
        eta_phi /= Z
        

        exp_delta = (1-eta_phi)/eta_phi * etahat_phi/(1-etahat_phi) #compute exp(delta)                                                 
 
        self.theta_[self.invB_[phi]] += np.log(exp_delta) #update theta_phi            

        Z = np.exp(-self.theta_perp) #update Z                                                                                           

        for x in self.Ssub_[phi]:
          Z += u[x] * (exp_delta - 1)
        self.theta_perp = - np.log(Z)
        
        for x in self.Ssub_[phi]: #update u                                                                                             
          u[x] *= exp_delta
        
        self.theta_list = [pre_theta_ ,self.theta_]
#        print(self.theta_list)
