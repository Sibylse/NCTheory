import torch.nn as nn
import torch
import torch.nn.functional as F


def gradient_penalty(inputs, outputs):
    gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
        )[0]
    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty
    
class CE_Loss(nn.Module):
    def __init__(self, c, device):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        #self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        #self.logits = 0
 
    def forward(self, logits, targets):  
        #self.logits = self.classifier(inputs) # prediction before softmax
        return self.ce_loss(logits, targets)

    def loss(self, inputs, targets, net):
        logits = net(inputs)
        return self.ce_loss(logits, targets), self.conf_logits(logits,net)

    def conf_logits(self,logits, net):
        if hasattr(net.classifier,'conf_logits'):
            return net.classifier.conf_logits(logits)
        return self.softmax(logits)
        
    def conf(self, inputs, net):
        logits = net(inputs)
        return self.softmax(logits)
    
    def prox(self, net):
        return

class CTLoss(nn.Module):
    def __init__(self, c, device, weight_nll=1):
        super(CTLoss, self).__init__()
        #self.I = torch.eye(c).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
        self.weight_nll = weight_nll
        #self.classifier = classifier.to(device)
 
    def forward(self, logits_views, targets):        
        #Y = self.I[targets].float().unsqueeze(1) #m x c
    #    logits_views = self.classifier(inputs) # m x d/d_view x c
        #logits_views = Y*logits_views + self.delta*(1-Y)*logits_views
        if len(logits_views.shape)==3:
            logits = logits_views.transpose(1,2)
            targets_rep = targets.repeat(logits.size(2),1).t()
            loss = self.ce_loss(logits,targets_rep) 
            loss+= self.weight_nll * self.nll_loss(logits,targets_rep)
        else:
            loss = self.ce_loss(logits_views,targets) 
            loss+= self.weight_nll * self.nll_loss(logits_views,targets)
        return loss

    def loss(self, inputs, targets, net):
        logits_views = net(inputs) # m x d/d_view x c
        #logits = logits_views.transpose(1,2)
        #targets_rep = targets.repeat(logits.size(2),1).t()
        #loss = self.ce_loss(logits,targets_rep) 
        #loss+= self.nll_loss(logits,targets_rep)
        if len(logits_views.shape)==3:
            y_pred = torch.exp(torch.sum(logits_views,1))
        else:
            y_pred = torch.exp(logits_views)
        return self.forward(logits_views, targets), y_pred
    
    def conf(self, inputs, net):
        logits_views = net(inputs)
        return torch.exp(torch.sum(logits_views,1))
    
    def prox(self,net):
        #torch.clamp_(self.delta, self.delta_min, self.delta_max)
        net.classifier.prox()


class BCE_DUQLoss(nn.Module):
    
    def __init__(self, c, device, weight_gp=0):
        super(BCE_DUQLoss, self).__init__()
        #self.bce_loss = nn.BCELoss()
        self.I = torch.eye(c).to(device)
        self.weight_gp = weight_gp
        self.embedding = 0
        #self.classifier = classifier.to(device)
        #self.Y_pred = 0 #predicted class confidences
        self.Y= 0
    
    def forward(self, logs, targets):
        self.Y = self.I[targets].float()
        Y_pred = torch.exp(logs)
        #loss = self.bce_loss(Y_pred, self.Y)
        loss = F.binary_cross_entropy(Y_pred, self.Y, reduction="mean")
        return loss

    def loss(self, inputs, targets, net):
        if self.weight_gp >0 and net.training:
            inputs.requires_grad_(True)
        self.Y = self.I[targets].float()
        self.embedding = net.embed(inputs)
        logs = net.classifier(self.embedding)
        Y_pred = torch.exp(logs)
        loss = F.binary_cross_entropy(Y_pred, self.Y, reduction="mean")
        if self.weight_gp > 0 and net.training:
            gp = gradient_penalty(inputs, Y_pred)
            loss += self.weight_gp * gp
        return loss, Y_pred
    
    def conf(self, inputs, net):
        logs = net(inputs)
        return torch.exp(logs)
    
    def prox(self, net):
        net.eval()
        net.classifier.update_centroids(self.embedding, self.Y)
        net.train()
