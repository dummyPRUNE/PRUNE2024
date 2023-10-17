import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

import torch, torchvision, time, numpy as np
from exp_tools.tool import *
from exp_tools.reassure import *
from sklearn.cluster import KMeans
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset
import random
from torch.utils.data import DataLoader, Subset

def PRUNE_multipoint(n_clusters, n, num_core, unlearn_number,iteration):
    # Load the original model
    origin_model = MLPNet([600, 256, 20])
    origin_model.load_state_dict(torch.load('PurchaseModel.pt'))

    # Obtain training and test sets corresponding to the original model. Note that the data has been shuffled.
    data = np.load('data/purchase/purchase20.npz')
    x, y = data["features"], data["labels"]
    y = np.argmax(y, axis=1)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    dataset = TensorDataset(x, y)
    train_data = Subset(dataset, [i for i in range(0, 38758)])
    test_data = Subset(dataset, [i for i in range(38759, 48448)])

    # Data for making a withdrawal request
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    unlearn_data= Subset(train_data, indices[:unlearn_number])
    rest_data = Subset(train_data, indices[unlearn_number:])

    unlearn_dataloader = DataLoader(unlearn_data, batch_size=1)
    rest_dataloader = DataLoader(rest_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    print("Before Unlearning")
    accuracy = compute_accuracy(origin_model, unlearn_dataloader)
    print("Unlearn Accuracy: ", accuracy)
    accuracy = compute_accuracy(origin_model, rest_dataloader)
    print("Rest Accuracy: ", accuracy)
    accuracy = compute_accuracy(origin_model, test_dataloader)
    print("Test Accuracy: ", accuracy)

    #########################
    with torch.no_grad():
        data_clu=[]
        tr_labels=[]
        for X, y in unlearn_dataloader:
            data_clu += X
            tr_labels += y
    data_clu = list(data_clu)
    tr_labels = list(tr_labels)
    for i in range(len(data_clu)):
        x = np.reshape(data_clu[i].numpy(), (1, 600))
        data_clu[i] = x[0]

    start = time.time()  # Start timing
    unlearn_set = []

    iter = 0
    while iter < iteration:
        iter += 1
        # K-Means. Each centroid is one of the original data points
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        kmeans.fit(data_clu)
        print("not forget data number:",len(data_clu))
        cluster_centers_indices = np.argsort(
            [np.linalg.norm(kmeans.cluster_centers_[i] - data_clu, axis=1) for i in range(n_clusters)])
        kmeans.predict(data_clu)
        cluster_centers_indices = cluster_centers_indices[:, 0]

        repre_points = []
        repre_labels = []
        for i in range(n_clusters):
            repre_points.append(data_clu[cluster_centers_indices[i]])
            repre_labels.append(tr_labels[cluster_centers_indices[i]])
        clusters = [[] for _ in range(n_clusters)]
        for i in range(len(data_clu)):
            label = kmeans.labels_[i]
            clusters[label].append(data_clu[i])
        for i in range(n_clusters):
            clusters[i] = torch.as_tensor(np.array(clusters[i]), dtype=None, device=None)
            clusters[i] = clusters[i].to(torch.float32)

        tr_labels = Format_tran(tr_labels)
        data_clu = Format_tran(data_clu)
        repre_points = Format_tran(repre_points)
        repre_labels = torch.tensor(repre_labels)
        # repre_labels = Format_tran(repre_labels)

        # Randomly generated confusing labels
        confused_labels = []
        for k in range(n_clusters):
            rand_label = np.random.randint(20)
            if rand_label != repre_labels[k]:
                confused_labels.append(rand_label)
            elif repre_labels[k] != 19:
                confused_labels.append(rand_label + 1)
            else:
                confused_labels.append(0)
        confused_labels = torch.tensor(confused_labels)

        for k in range(n_clusters):
            repre_point, repre_label, confused_label = repre_points[k:k+1], repre_labels[
                                                                                     k:k+1], confused_labels[
                                                                                                   k:k+1]
            # Use REASSURE to calculate support network
            P, ql, qu = specification_matrix_from_labels(confused_labels, dim=20)

            PRUNE = Unlearn_multipoints(origin_model, n, bounds=[torch.zeros(600), torch.ones(600)])
            PRUNE.point_wise_unlearn(repre_point, clusters[k], P, ql, qu, False)
            unlearning_model = PRUNE.compute(num_core)
            unlearn_set.append(unlearning_model)

        unlearned_model = PatchSum(origin_model, unlearn_set)
        torch.save(unlearned_model.state_dict(), 'PPunlearn_model_{}.pt'.format(n_clusters))
        origin_model = MLPNet([600, 256, 20])
        origin_model.load_state_dict(
            {k.replace('model.', ''): v for k, v in torch.load('PPunlearn_model_2.pt').items()})
        # target_model=repaired_model
        data_clu, tr_labels = success_unlearn_rate(unlearned_model, data_clu, tr_labels)

    cost_time = time.time() - start
    print("****************************************************************************")
    print('cost time:', cost_time)
    print('iteration:', iteration)
    print("After Unlearning")
    accuracy = compute_accuracy(unlearned_model, unlearn_dataloader)
    print("Unlearn Accuracy: ", accuracy)
    accuracy = compute_accuracy(unlearned_model, rest_dataloader)
    print("Rest Accuracy: ", accuracy)
    accuracy = compute_accuracy(unlearned_model, test_dataloader)
    print("Test Accuracy: ", accuracy)
    print("****************************************************************************")


if __name__ == '__main__':
    for unlearn_number in [100,200,300,500]:
        print('-'*30, 'unlearn data number =', unlearn_number, '-'*30)
        PRUNE_multipoint(n_clusters=2, n=0.1, num_core=4,unlearn_number=unlearn_number,iteration=2)






