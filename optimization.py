import torch
import torch.nn.functional as F
import numpy as np
import time

class Optimizer:
  def __init__(self, optimizer, trainloader, device, update_centroids=False):
    self.optimizer = optimizer
    self.trainloader = trainloader
    self.n = len(trainloader.dataset)
    self.update_centroids = update_centroids
    self.device=device
    self.best_acc=0

  def train_epoch(self, net, criterion, verbose=False):
    train_loss, correct, conf = 0, 0, 0
    t1, t2, t3, t4 = 0,0,0,0
    start_time=time.time()
    net.train() 
    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      self.optimizer.zero_grad()
      loss, P = criterion.loss(inputs,targets, net)
      if verbose:
        print("loss:",loss.item())
      loss.backward()
      self.optimizer.step()
      inputs.requires_grad_(False)

      with torch.no_grad():
        #if self.update_centroids:
        #  net.eval()
        #  criterion.classifier.update_centroids(embedding, criterion.Y)
        #  net.train()
        n_b = targets.shape[0]
        M, v, residuals = self.compute_centers(net)
        M = torch.from_numpy(M.T).float()
        X = net.embed(inputs)
        t1_, t2_, t3_, t4_ = self.custom_mu_loss_terms(X, targets, P, M)
        t1+= t1_*n_b
        t2+= t2_*n_b
        t3+= t3_*n_b
        t4+= t4_*n_b
        train_loss += loss.item()*n_b
        confBatch, predicted = P.max(1)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    execution_time = (time.time() - start_time)
    print('Loss: %.6f | Acc: %.3f%% (%d/%d) | Conf %.2f | time (s): %.2f'% (train_loss/self.n, 100.*correct/self.n, correct, self.n, 100*conf/self.n, execution_time))
    print('||x-mu_y||^2: %.6f | ||x-p^T M||^2: %.6f | p_kp_l||mu_k-mu_l||^2: %.6f | H(p): %.6f'% (t1/self.n, t2/self.n, t3/self.n, t4/self.n))
    return (train_loss/self.n,t1/self.n, t2/self.n, t3/self.n, t4/self.n)
  
  def test_acc(self, net, criterion, data_loader, min_conf=0):
    net.eval()
    test_loss, correct, conf, total = 0,0,0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #outputs = net.embed(inputs)
            loss,Y_pred = criterion.loss(inputs, targets, net)

            test_loss += loss.item()
            confBatch, predicted = Y_pred.max(1)
            idx = (confBatch>min_conf)
            correct += predicted[idx].eq(targets[idx]).sum().item()
            conf+=confBatch[idx].sum().item()
            total+= idx.sum()
    print('Loss: %.6f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (test_loss/max(len(data_loader),1), 100.*correct/total, correct, total, 100*conf/total))
    return (100.*correct/total, 100*conf/total)

  def custom_mu_loss_terms(self, X, y, P, M, eps=1e-14):
    n, d = X.shape
    M_y = M[:, y].T #bxd
    p_y = P.gather(1, y[:, None]).squeeze(1)

    term1 = 0.5*((X - M_y) ** 2).sum(dim=1)

    PM = P @ M.T
    term2 = 0.5*((X -PM ) ** 2).sum(dim=1)

    mu_sq = (M ** 2).sum(dim=0, keepdim=True)
    D = (mu_sq.T + mu_sq - 2.0 * (M.T @ M)).clamp_min(0.0)

    term3 = 0.25 * (P * (P @ D)).sum(dim=1)

    P_safe = P.clamp_min(eps)
    term4 = -(P_safe * P_safe.log()).sum(dim=1)  

    return term1.mean().item(), term2.mean().item(), term3.mean().item(), term4.mean().item()

  def compute_centers(self, net):
    W = net.classifier.weight.detach().numpy()
    W = W - np.outer(np.ones(W.shape[0]),np.mean(W,0))
    b = net.classifier.bias.detach().numpy()
    return self.compute_centers_np(W,b)

  def compute_centers_np(self, W,b):
    c=W.shape[0]
    v, residuals, rank, s = np.linalg.lstsq(np.vstack([W.T,np.ones(c)]).T,-b-0.5*np.sum(W**2,1),rcond=1e-3)  
    C = W+np.outer(np.ones(c),v[:-1])
    return C,v, residuals

  def optimize_centroids(self, net):
    net.eval()
    d,c = net.classifier.in_features,net.classifier.out_features
    Z=torch.zeros(d,c).to(self.device)
    y_sum = torch.zeros(c).to(self.device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            D = net.embed(inputs)
            Y = F.one_hot(targets, c).float().to(self.device)
            Z += D.t().mm(Y)
            y_sum += torch.sum(Y,0)
    Z = Z/y_sum
    net.classifier.weight.data = Z.t()
