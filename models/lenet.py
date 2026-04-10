'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
from models.spectral_normalization.spectral_norm_fc import spectral_norm_fc

# The embedding architecture returns the 
# output of the penultimate layer
class LeNetEmbed(nn.Module):
    def __init__(self,embedding_dim=84,coeff=None,n_power_iterations=1):
        super(LeNetEmbed, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, embedding_dim)
        if coeff is not None: #do spectral normalization constraining L<coeff (approximately)
            self.conv1 = spectral_norm_conv(self.conv1, coeff, (1,28,28), n_power_iterations)
            self.conv2 = spectral_norm_conv(self.conv2, coeff, (6,12,12), n_power_iterations)
            self.fc1 = spectral_norm_fc(self.fc1, coeff, n_power_iterations)
            self.fc2 = spectral_norm_fc(self.fc2, coeff, n_power_iterations)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out

#class LeNetEmbedActiv(nn.Module):
#    def __init__(self,embed, last_activation):
#        super(LeNetEmbedActiv, self).__init__()
#        self.embed =  embed
#        self.last_activation = last_activation

#    def forward(self, x):
#        out = self.embed(x)
#        out = self.last_activation(out)
#        return out     
    
class LeNet(nn.Module):
    def __init__(self,embedding_dim, classifier,coeff=None,n_power_iterations=1):
        super(LeNet, self).__init__()
        self.embed = LeNetEmbed(embedding_dim=embedding_dim, coeff=coeff, n_power_iterations=n_power_iterations)
        self.classifier = classifier

    def forward(self, x):
        out = self.embed(x)
        out = self.classifier(out)
        return out

    def conf(self,x):
        out = self.embed(x)
        if hasattr(self.classifier,'conf'):
            return self.classifier.conf(out)
        return F.softmax(self.classifier(out),dim=1)
