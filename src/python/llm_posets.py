import numpy as np
import time
import sys
import math
import copy
import itertools 
def includes(phi,x):
  return set(phi).issubset(x)

class LLM:
  def __init__ (self, n, B, S, X):

    self.set_B(B) # list of tuples
    self.set_S(S) # list of tuples
    self.n_= n
    self.set_Bsub(self.S_, self.B_) #dict (key: tuple, val: list of tuples)
    self.set_Ssub(self.S_, self.B_)  # dict (key: tuple, val: list of tuples)
    self.init_theta() # dict (key: list, val: float)

    self.compute_Phat(X) # dict (key: list, val: float)
    self.compute_etahat(X) # dict (key: list, val: float)


  def fit(self, n_iter, stepsize=-1, mu = 0, lambda_ = -1, solver="grad"):
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
      self.gradient_descent(n_iter, stepsize)
    elif solver == "da":
      self.dual_averaging(n_iter, stepsize)
    elif solver == "coor":
      self.coordinate_descent(n_iter)
    elif solver == "acc_grad":
      self.accelerated_gradient_descent(n_iter, stepsize, mu)
    elif solver == "newton":
      self.newton_method(n_iter, stepsize)
    elif solver == "coor_l1":
      self.coordinate_descent_l1(n_iter, lambda_)
    elif solver == "prox":
      self.proximal_gradient_descent(n_iter, stepsize, lambda_)
    elif solver == "rda":
      self.regularized_dual_averaging(n_iter, stepsize, lambda_)
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
    self.Phat_ = np.zeros(len(self.S_))
    #denom = 0.0
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
      etahat_phi = 0.0
      for xi in self.Ssub_[phi]:
        etahat_phi += self.Phat_[self.invS_[xi]] 
      self.etahat_[phi] = etahat_phi

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
    ret = 0.0;
    for phi in self.Bsub_[x]:
      ret +=  self.get_theta(phi)
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
      for phi in self.Bsub_[x]:
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

  def compute_hess(self):
    eta_p1p2 = np.zeros((len(self.B_),len(self.B_)))
    
    for p1 in self.B_:
      for p2 in self.B_:
        r = 0.0
        for x in self.Ssub_[p1]:
          r += self.P_[self.invS_[x]] if includes(p2,x) else 0
        eta_p1p2[self.invB_[p1],self.invB_[p2]] = r

    hess = np.zeros((len(self.B_),len(self.B_)))
    for phi1 in self.B_:
      for phi2 in self.B_:
        hess[self.invB_[phi1],self.invB_[phi2]] = eta_p1p2[self.invB_[phi1],self.invB_[phi2]] - self.eta_[phi1] * self.eta_[phi2]
    #a,v = np.linalg.eig(hess)
    inv_hess = np.linalg.inv(hess)

    return inv_hess


  def gradient_descent(self, n_iter, step):  
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
  

  def dual_averaging(self, n_iter, step):  
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
    grad = np.zeros(len(self.B_))
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      kl=self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)
      
      for phi in self.B_:
        grad[self.invB_[phi]] = ( iter * grad[self.invB_[phi]] +self.etahat_[phi] - self.eta_[phi]) / (iter+1)
        new_theta_phi = self.get_theta(phi) + step / math.sqrt(iter+1) * grad[self.invB_[phi]]
        self.set_theta(phi, new_theta_phi)

      self.compute_theta_perp()


  def newton_method(self, n_iter, step):
    start = time.time()
    for iter in range(n_iter):
      self.compute_P()
      self.compute_eta()
      
      kl = self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)


      grad = np.zeros(len(self.B_))
      for phi in self.B_:
        grad[self.invB_[phi]] = self.etahat_[phi] - self.eta_[phi]
      inv_hess = self.compute_hess()
      for phi in self.B_:
        new_theta_phi = self.get_theta(phi) + step * inv_hess[self.invB_[phi]].dot(grad)
        self.set_theta(phi, new_theta_phi)
      
      self.compute_theta_perp()


  def update_accelerated_theta(self, iter, step, mu):      
    
    if iter == 0:

      self.grad_list = [] 
      # [theta(t) - step*grad( L_D( theta(t)) ), theta_(t-1) - step* ( L_D(theta(t-1)) ) ]
      self.grad_theta_ = np.zeros(len(self.B_)) 
      # theta^(t-1) - step * grad( L_D( theta^(t-1) ) )
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
        if mu == 0:
          new_theta_phi = grad_new_theta_phi   #no hyperparameter
        else:
          new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[self.invB_[phi]]

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
        if mu ==0:
          new_theta_phi = (1 -self.gamma) * grad_new_theta_phi + self.gamma * pre_grad_theta_[self.invB_[phi]]      #no hyperparameter
        else:
          new_theta_phi = (1 + mu) * grad_new_theta_phi - mu * pre_grad_theta_[self.invB_[phi]]

        self.set_theta(phi, new_theta_phi)
        
      self.grad_list = [pre_grad_theta_,self.grad_theta_]
      self.lambda_list = [pre_lambda_,self.lambda_]

  def accelerated_gradient_descent(self, n_iter, step, mu):
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

  
  def coordinate_descent(self, max_epoch):
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
        Phatx = self.Phat_[self.invS_[x]]
        if Phatx != 0.0:
          kl += Phatx * (np.log(Phatx) - np.log(u[x]/Z))

      print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),  flush=True)
      #index = range(len(self.B_))
      #index = np.random.RandomState().permutation(range(len(self.B_))) #permutative
      index = np.random.randint(0,len(self.B_)-1,len(self.B_))  #random

      for iter in range(len(self.B_)):

        phi = self.B_[ index[iter] ] 
        etahat_phi = self.etahat_[phi] 

        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          continue

        #compute eta_phi
        eta_phi = 0.0
        for x in self.Ssub_[phi]: 
            eta_phi += u[x]
        eta_phi /= Z
        
        #compute exp(delta)
        exp_delta = (1-eta_phi)/eta_phi * etahat_phi/(1-etahat_phi) 

        #update theta_phi
        self.theta_[self.invB_[phi]] += np.log(exp_delta) 

        #update Z

        for x in self.Ssub_[phi]:
          Z += u[x] * (exp_delta - 1)

        self.theta_perp = - np.log(Z)
        
        #update u
        for x in self.Ssub_[phi]:
          u[x] *= exp_delta
        

  def coordinate_descent_l1(self, max_epoch, lambda_):
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
        Phatx = self.Phat_[self.invS_[x]]
        if Phatx != 0.0:
          kl += Phatx * (np.log(Phatx) - np.log(u[x]/Z))

      print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),  flush=True)
      
      #index = range(len(self.B_))
      #index = np.random.RandomState().permutation(range(len(self.B_))) #permutative
      index = np.random.randint(0,len(self.B_)-1,len(self.B_))  #random
      for iter in range(len(self.B_)):

        phi = self.B_[ index[iter] ]
        etahat_phi = self.etahat_[phi]

        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          continue

        #compute eta_phi                                                                                                                  
        eta_phi = 0.0
        for x in self.Ssub_[phi]:
            eta_phi += u[x]
        eta_phi /= Z

        exp_delta = None
        exp_delta_1 = (1 - eta_phi) / eta_phi * (etahat_phi - lambda_) / (1 - etahat_phi + lambda_)
        exp_delta_2 = (1 - eta_phi) / eta_phi * (etahat_phi + lambda_) / (1 - etahat_phi - lambda_)
        if np.exp(-self.theta_[self.invB_[phi]]) < exp_delta_1:
          exp_delta = exp_delta_1
        elif np.exp(-self.theta_[self.invB_[phi]]) > exp_delta_2:
          exp_delta = exp_delta_2

        #update theta_phi
        if exp_delta == None:
          self.theta_[self.invB_[phi]] = 0
          exp_delta = np.exp( - copy.copy(self.theta_[self.invB_[phi]]))
        else:
          self.theta_[self.invB_[phi]] += np.log(exp_delta)


        #update Z
        for x in self.Ssub_[phi]:
          Z += u[x] * (exp_delta - 1)

        self.theta_perp = - np.log(Z)

        #update u                                                                                                                         
        for x in self.Ssub_[phi]:
          u[x] *= exp_delta
    print(np.count_nonzero(self.theta_)/len(self.theta_))


  def proximal_gradient_descent(self, n_iter, step, lambda_):  
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
        theta_phi_ = self.get_theta(phi) + step * (self.etahat_[phi] - self.eta_[phi] )
        if np.abs(theta_phi_) <= lambda_*step:
          new_theta_phi=0
        else:
          new_theta_phi = theta_phi_ - lambda_ * step * np.sign(theta_phi_)
        self.set_theta(phi, new_theta_phi)

      self.compute_theta_perp()
    print(self.theta_)


  def regularized_dual_averaging(self, n_iter, step, lambda_):  
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
    grad = np.zeros(len(self.B_))
    for iter in range(n_iter):
      
      self.compute_P()
      self.compute_eta()

      kl=self.compute_KL()
      print(iter ,":", "KL divergence: ", f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),flush=True)

      for phi in self.B_:
        grad[self.invB_[phi]] = ( iter * grad[self.invB_[phi]] +self.etahat_[phi] - self.eta_[phi]) / (iter+1)
        theta_phi_ = step * math.sqrt(iter+1) * grad[self.invB_[phi]]
        if np.abs(theta_phi_) <=lambda_ * step * math.sqrt(iter+1):
          new_theta_phi=0
        else:
          new_theta_phi = theta_phi_ - lambda_ * step * math.sqrt(iter+1) * np.sign(theta_phi_)
        self.set_theta(phi, new_theta_phi)

      self.compute_theta_perp()
    print(self.theta_)
    

  def accelerated_coordinate_descent(self, max_epoch):
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
    self.alphas = [0.25]
    self.ys = np.zeros(len(self.B_))
    self.thetas = [np.zeros(len(self.B_))]

    start = time.time()
    for epoch in range(max_epoch):

      #compute KL
      kl = 0.0
      for x in self.S_:
        Phatx = self.Phat_[self.invS_[x]]
        if Phatx != 0.0:
          kl += Phatx * (np.log(Phatx) - np.log(u[x]/Z))

      print(epoch ,":",  "KL divergence:",f'{kl:.16f}' ," time : %4.2f"% (time.time()-start),  flush=True)

      #index = np.random.RandomState().permutation(range(len(self.B_))) #permutative        
 
      #index = np.random.randint(0,len(self.B_)-1,len(self.B_))  #random                   


      for iter in range(len(self.B_)):
        phi = self.B_[ index[iter] ]
        etahat_phi = self.etahat_[phi]

        if etahat_phi >= 1.0 or etahat_phi == 0.0:
          continue
        
        #self.update_parameters(epoch,iter)

        #compute eta_phi                                                                   
        
        eta_phi = 0.0
        for x in self.Ssub_[phi]:
            eta_phi += u[x]
        eta_phi /= Z

        #compute exp(delta)                                                                
 
        exp_delta = (1-eta_phi)/eta_phi * etahat_phi/(1-etahat_phi)

        #update theta_phi                                                                  
 
        self.theta_[self.invB_[phi]] += np.log(exp_delta)                                 
        #update Z
        for x in self.Ssub_[phi]:
          Z += u[x] * (exp_delta - 1)

        self.theta_perp = - np.log(Z)                                                     
 

        #update u                                                                          
        for x in self.Ssub_[phi]:
          u[x] *= exp_delta
