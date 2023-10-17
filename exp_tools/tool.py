import torch, torchvision, numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, size):
        super(MLPNet, self).__init__()
        self.size = size
        self.layers = nn.ModuleList()
        for i in range(len(size)-1):
            self.layers.append(nn.Linear(size[i], size[i+1]))

    def forward(self, x):
        x = x.view(-1, self.size[0])
        for layer in self.layers:
            if layer is self.layers[-1]:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x

    def all_hidden_neurons(self, x):
        hidden_neurons = []
        x = x.view(self.size[0])
        for layer in self.layers[:-1]:
            if layer is self.layers[0]:
                x = layer(x)
            else:
                x = layer(F.relu(x))
            hidden_neurons.append(x)
        return torch.cat(hidden_neurons, dim=-1)

    def activation_pattern(self, x):
        x_activation_pattern = self.all_hidden_neurons(x) > 0
        return [entry.item() for entry in x_activation_pattern]

def compute_accuracy(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def success_unlearn_rate(model, inputs, labels):
    with torch.no_grad():
        fail_data=[]
        fail_label=[]
        pred = model(inputs)
        for i in range(len(inputs)):
            if pred.argmax(1)[i]==labels[i]:
                fail_data.append(inputs[i])
                fail_label.append(labels[i])
    for i in range(len(fail_data)):
        x = np.reshape(fail_data[i].numpy(), (1, 600))
        fail_data[i] = x[0]
    return fail_data,fail_label


def Format_tran(data):
    data = torch.as_tensor(np.array(data), dtype=None, device=None)
    data = data.to(torch.float32)
    return data

class PatchSum(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, pnns):
        super(PatchSum, self).__init__()
        self.model = model
        self.pnns=pnns
    def forward(self, x):
        m=self.model(x)
        if len(x.size()) >= 3:
            for p in range(len(self.pnns)):
                m+=self.pnns[p](x.view(len(x), -1))
        else:
            for p in range(len(self.pnns)):
                m += self.pnns[p](x)
        return m
