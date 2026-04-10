import torch
import torch.nn.functional as F
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
    start_time=time.time()
    net.train() 
    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      self.optimizer.zero_grad()
      loss, Y_pred = criterion.loss(inputs,targets, net)
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
        train_loss += loss.item()
        confBatch, predicted = Y_pred.max(1)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    execution_time = (time.time() - start_time)
    print('Loss: %.6f | Acc: %.3f%% (%d/%d) | Conf %.2f | time (s): %.2f'% (train_loss/len(self.trainloader), 100.*correct/self.n, correct, self.n, 100*conf/self.n, execution_time))
    return (train_loss/len(self.trainloader),100.*correct/self.n, 100*conf/self.n)
  
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
