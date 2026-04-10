'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Centroid_Squared_Distances(nn.Module):
    def __init__(self,in_features,out_features):
        super(Centroid_Squared_Distances, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features)) # (cxd) centroids
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, D):
        out = D.unsqueeze(2) - self.weight.t().unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        out = (out**2) #mxdxc
        return out
        
    def get_margins(self):
        #X is dxc, out is cxc matrix, containing the distances ||X_i-X_j||
        # only the upper triangle of out is needed
        X = self.weight.data.t()
        out = X.t().unsqueeze(2) - X.unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        out= torch.sqrt(torch.sum((out**2),1))
        triu_idx = torch.triu_indices(out.shape[0], out.shape[0],1)
        return out[triu_idx[0],triu_idx[1]]

        
class CTroid(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features,d_view=None, gamma=0.5, gamma_min=0.05,gamma_max=1000):
        super(CTroid, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        #self.gamma=nn.Parameter(gamma*torch.ones(int(in_features/d_view),out_features)) #exp(-gamma_k||D_j.^T - C_.k||^2)
        self.gamma=nn.Parameter(gamma*torch.ones(out_features)) #exp(-gamma_k||D_j.^T - C_.k||^2)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) # (cxd) centroids
        self.d_view = d_view
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, D):
        out = D.unsqueeze(2) - self.weight.t().unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        out = (out**2) #mxdxc
        if self.d_view is not None and self.training:
            out = out.view(-1,int(self.in_features/self.d_view),self.d_view,self.out_features).sum(2)
        else :
            out = out.sum(1)
        out = (out*self.gamma) # (mxd/d_viewxc) or (mxc)
        return -out # # (mxd/d_viewxc) or (mxc)
    
    def conf(self,D):
        return self.conf_logits(self.forward(D))

    def conf_logits(self,logits):
        if self.d_view is not None and self.training:
            return torch.exp(torch.sum(logits,1))
        return torch.exp(logits)

    def conf_view(self, D,i):
        """
        For plotting purposes - returns a two-dimensional view (dimensions i and i+1) of the confidences assigned to the points in D (m x 2)
        """
        out = D.unsqueeze(2) - self.weight.t()[[i,i+1],:].unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc
        return torch.exp(-self.gamma*torch.sum((out**2),1)) # (mxc)
    
    def prox(self):
        torch.clamp_(self.gamma, self.gamma_min, self.gamma_max)
            
    def get_margins(self):
        #X is dxc, out is cxc matrix, containing the distances ||X_i-X_j||
        # only the upper triangle of out is needed
        X = self.weight.data.t()
        out = X.t().unsqueeze(2) - X.unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        out= torch.sqrt(torch.sum((out**2),1))
        triu_idx = torch.triu_indices(out.shape[0], out.shape[0],1)
        return out[triu_idx[0],triu_idx[1]]
    
class Gauss_DUQ(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, gamma, N_init=None, m_init=None, alpha=0.999):
        super(Gauss_DUQ, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "gamma", gamma 
        )
        #self.gamma=gamma
        self.alpha=alpha
        if N_init==None:
            N_init = torch.ones(out_features)*10
        if m_init==None:
            m_init = torch.normal(torch.zeros(in_features, out_features), 0.05)
        self.register_buffer("N", N_init) # 
        self.register_buffer(
            "m", m_init # (dxc)
        )
        self.m = self.m * self.N
        self.W = nn.Parameter(torch.zeros(in_features, out_features, in_features)) # (dxcxr) (r=d)
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

    def forward(self, D):
        DW = torch.einsum("ij,mnj->imn", D, self.W) # (mxdxc)
        Z = self.m / self.N.unsqueeze(0) # centroids (dxc)
        out = DW - Z.unsqueeze(0)
        return -self.gamma*torch.mean((out**2),1) # (mxc)

    def conf(self,D):
        return self.conf_logits(self.forward(D))

    def conf_logits(self,logits):
        return torch.exp(logits)
    
    def update_centroids(self, D, Y):
        DW = torch.einsum("ij,mnj->imn", D, self.W) # (mxdxc)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.alpha * self.N + (1 - self.alpha) * Y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", DW, Y)

        self.m = self.alpha * self.m + (1 - self.alpha) * features_sum

class Gauss_Process(nn.Module): #SNGP final layer
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, rff_features=1024, ridge_penalty=1.0, rff_scalar=None, mean_field_factor=25):
        super(Gauss_Process, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ridge_penalty=ridge_penalty
        self.mean_field_factor = mean_field_factor
        
        self.rff = RandomFourierFeatures(in_features, rff_features, rff_scalar)
        self.logit = nn.Linear(rff_features, out_features) #multiply with beta matrix, why is there a bias? Might be a mistake.
        
        precision = torch.eye(rff_features) * self.ridge_penalty
        self.register_buffer("precision", precision)
        self.register_buffer("covariance", torch.eye(rff_features)) #precision is inverse of covariance
        

    def forward(self, D):
        Phi = self.rff(D)
        pred = self.logit(Phi)

        if self.training:
            self.precision += Phi.t() @ Phi
        else: #the covariance has to be updated before by invoking eval()
            with torch.no_grad():
                pred_cov = Phi @ ((self.covariance @ Phi.t()) * self.ridge_penalty)
            if self.mean_field_factor is None:
                return pred, pred_cov
            # Do mean-field approximation as alternative to MC integration of Gaussian-Softmax
            # Based on: https://arxiv.org/abs/2006.07584
            logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
            if self.mean_field_factor > 0:
                pred = pred / logits_scale.unsqueeze(-1)
        return pred

    def conf(self,D):
        return self.conf_logits(self.forward(D))

    def conf_logits(self,logits):
        return F.softmax(logits,dim=1)
    
    def train(self,mode=True):
        if mode: #training is starting (optimizer calls train() each epoch)
            identity = torch.eye(self.precision.shape[0], device=self.precision.device)
            self.precision = identity * self.ridge_penalty
            print("reset precision matrix")
        elif self.training: #switch from training to eval mode
            self.update_covariance()
            print("updated covariance matrix")
        return super().train(mode)
        
    
    def update_covariance(self):
        with torch.no_grad():
            eps = 1e-7  
            jitter = eps * torch.eye(self.precision.shape[1],device=self.precision.device)
            u, info = torch.linalg.cholesky_ex(self.precision + jitter)
            assert (info == 0).all(), "Precision matrix inversion failed!"
            torch.cholesky_inverse(u, out=self.covariance)
        

class RandomFourierFeatures(nn.Module):
    __constants__ = ['in_features', 'rff_features']
    
    def __init__(self, in_features, rff_features, rff_scalar=None):
        super().__init__()
        if rff_scalar is None:
            rff_scalar = math.sqrt(rff_features / 2)

        self.register_buffer("rff_scalar", torch.tensor(rff_scalar))

        if rff_features <= in_features:
            W = self.random_ortho(in_features, rff_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = rff_features
            ws = []
            while dim_left > in_features:
                ws.append(self.random_ortho(in_features, in_features))
                dim_left -= in_features
            ws.append(self.random_ortho(in_features, dim_left))
            W = torch.cat(ws, 1)

        # From: https://github.com/google/edward2/blob/d672c93b179bfcc99dd52228492c53d38cf074ba/edward2/tensorflow/initializers.py#L807-L817
        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b = torch.empty(rff_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x):
        k = torch.cos(x @ self.W + self.b)
        k = k / self.rff_scalar
        return k
    
    def random_ortho(self,n, m):
        q, _ = torch.linalg.qr(torch.randn(n, m))
        return q
  
# Gaussian Multivariate Layer for epistemic and aleatoric uncertainty as proposed for the DDU model
class Gauss_DDU(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features, gamma =5e-3):
        super(Gauss_DDU, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('classwise_mean_features', torch.zeros(out_features, in_features))
        self.register_buffer('classwise_cov_features', torch.eye(in_features).unsqueeze(0).repeat(out_features, 1, 1))
        self.gda = self.init_gda()  # class-wise multivatiate Gaussians, to be initialized with fit()
        self.mahalanobis = torch.distributions.multivariate_normal._batch_mahalanobis
        self.gamma=nn.Parameter(gamma*torch.ones(out_features))
        #self.gamma = gamma
    
    def forward(self, D):
        L  = self.gda._unbroadcasted_scale_tril
        return -self.gamma * self.mahalanobis(L, D[:, None, :]-self.gda.loc).float()
    
    def get_log_probs(self,D):    
        return self.gda.log_prob(D[:, None, :]).float()

    def conf(self,D):
        return self.conf_logits(self.forward(D))

    def conf_logits(self,logits):
        return torch.exp(logits)
    
    def prox(self):
        return
    
    def fit(self, embeddings, labels): #embeddings should be num_samples x dim_embedding
        with torch.no_grad():
            classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(self.out_features)])
            classwise_cov_features = torch.stack(
                [torch.cov(embeddings[labels == c].T) for c in range(self.out_features)])

            for jitter_eps in [0, torch.finfo(torch.float).tiny] + [10 ** exp for exp in range(-308, 0, 2)]:
                try:
                    jitter = jitter_eps * torch.eye(
                        classwise_cov_features.shape[1], device=classwise_cov_features.device,
                    ).unsqueeze(0)
                    self.classwise_mean_features = classwise_mean_features
                    self.classwise_cov_features = classwise_cov_features + jitter
                    self.init_gda()
                except RuntimeError as e:
                    continue
                except ValueError as e:
                    continue
                break
    
    def init_gda(self):
        self.gda = torch.distributions.MultivariateNormal(loc=self.classwise_mean_features, covariance_matrix=(self.classwise_cov_features))

# simple layer to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

