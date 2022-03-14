import torch
import torch.nn as nn
import torch.nn.functional as F


# Fully connected neural network with one hidden layer
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, net_type):
        super(ClassifierNetwork, self).__init__()
        self.net_type = net_type
        if net_type=='MLP':
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        elif net_type=='Linear':
            self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        if self.net_type=='MLP':
            out = self.tanh(out)
            out = self.fc2(out)
        return out


class kNN(nn.Module):
    def __init__(self, x_train, labels_train, distance_type='L2', k=1):
        super(kNN, self).__init__()
        self.k = k
        self.distance_type = distance_type
        self.support_embeddings = x_train
        self.support_labels = labels_train
        self.num_classes = max(labels_train)+1

    def cosine_similarity(self, class_hvs, enc_hv):
        class_hvs = torch.div(class_hvs, torch.norm(class_hvs, dim=1, keepdim=True))
        enc_hv = torch.div(enc_hv, torch.norm(enc_hv, dim=1, keepdim=True))
        return torch.matmul(enc_hv, class_hvs.t())

    def forward(self, x):
        if self.distance_type in ['L1', 'L2']:
            #  [num_query, 1, embed_dims]
            query = torch.unsqueeze(x, 1)
            #  [1, num_support, embed_dims]
            support = torch.unsqueeze(self.support_embeddings, 0)

            if self.distance_type == 'L1':
                #  [num_query, num_support]
                distance = torch.linalg.norm(query-support, dim=2, ord=1)
            else:
                distance = torch.linalg.norm(query-support, dim=2, ord=2)
        elif self.distance_type == 'cosine':
            distance = -1.0 * self.cosine_similarity(self.support_embeddings, x)
        else:
            raise ValueError('Distance must be one of L1, L2 or cosine.')        
             
        _, idx = torch.topk(-distance, k=self.k)
        idx = torch.squeeze(idx, dim=1)
        idx = torch.gather(self.support_labels, 0, idx)
        one_hot_classification = F.one_hot(idx, num_classes=self.num_classes)
        return one_hot_classification
