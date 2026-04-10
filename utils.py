import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_conf(conf, show_class_assignment=False, x_max=20, y_max=20, x_min=-1, y_min=-1):
    x = np.arange(x_min, x_max, 0.05)
    y = np.arange(y_min, y_max, 0.05)

    xx, yy = np.meshgrid(x, y)
    X = np.array([xx,yy]).reshape(2,x.shape[0]*y.shape[0]).T
    Z = conf(torch.from_numpy(X).float()).t()
    Z = Z.reshape(-1,y.shape[0],x.shape[0]).cpu().detach().numpy()
    if show_class_assignment:
        h = plt.contourf(x,y,Z.argmax(axis=0),cmap='magma')
    else:
        h = plt.contourf(x,y,Z.max(axis=0),cmap='magma')
        plt.clim(0, 1)
        cb = plt.colorbar()
        cb.set_label('Confidence')
    plt.axis('scaled')

def plot_epoch(net, data_loader, device, figsize = (18,7), conf_view=None):
    plt.figure(figsize=figsize)
    with torch.no_grad():
        (inputs, targets) = next(iter(data_loader))
        inputs,targets = inputs.to(device), targets.to(device)
        outputs = net.embed(inputs).detach().cpu()
        d = outputs.shape[1]
        for i in range(0,min(int(d/2),5)):
            plt.subplot(1,min(int(d/2),5),i+1)
            if conf_view is not None:
              plot_conf(lambda D: conf_view(D.to(device),2*i), x_max =max(outputs[:,2*i])+1, y_max =max(outputs[:,2*i+1])+1, x_min =min(outputs[:,2*i])-1, y_min =min(outputs[:,2*i+1])-1)
              plt.title("dim %i, %i"%(2*i,2*i+1))
            plt.scatter(outputs[:, 2*i], outputs[:, 2*i+1], c=targets.cpu(), s=20, alpha=0.5, cmap="gist_rainbow")
        plt.show()
        
def load_net(name,architecture, path = "checkpoint/"):
    checkpoint = torch.load(path+name,map_location='cpu')
    architecture.load_state_dict(checkpoint['net'])
    architecture.eval()
    print(name+' ACC:\t',checkpoint['acc'])
    return architecture

#get the embeddings for all data points
def gather_embeddings(net, d, loader: torch.utils.data.DataLoader, device, storage_device):
    num_samples = len(loader.dataset)
    output = torch.empty((num_samples, d), dtype=torch.double, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            out = net.embed(data)

            end = start + len(data)
            output[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return output, labels
