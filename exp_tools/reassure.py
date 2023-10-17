import torch
import numpy as np
from scipy.optimize import linprog
import multiprocessing
import torch.nn as nn


def specification_matrix_from_labels(right_labels, dim=10):
    P_list, qu_list, ql_list = [], [], []
    for k in right_labels:
        e = np.zeros([1, dim])
        e[0][k] = 1
        P = np.eye(dim) - np.matmul(np.ones([dim, 1]), e)
        P_list.append(np.delete(P, k, 0))
        qu_list.append(np.zeros(dim-1))
        ql_list.append(np.ones(dim-1) * -100)
    return P_list, ql_list, qu_list


def linear_region_from_input(x, all_neurons, bounds):
    A_list, b_list = [], []
    for neuron in all_neurons:
        neuron.backward(retain_graph=True)
        grad_x = x.grad
        A = grad_x.clone().detach()
        b = torch.matmul(A, x.squeeze()) - neuron
        if neuron >= 0:
            A, b = -A, -b
        A_list.append(A.detach().numpy())
        b_list.append(b.detach().numpy())
    A = np.concatenate([-np.eye(len(bounds[0])), np.eye(len(bounds[1])), np.stack(A_list)])
    b = np.concatenate([-bounds[0], bounds[1], np.stack(b_list)])
    "-x <= -bounds[0]; x <= bounds[1] <=> "
    "bounds[0] <= x <= bounds[1]"
    return np.float32(A), np.float32(b)


class SupportNN(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, n):
        # Ax <= b
        super(SupportNN, self).__init__()
        assert len(A) == len(b)
        A = torch.tensor(A)
        b = torch.tensor(b)
        assert len(A.size()) == 2
        layer = nn.Linear(*A.size())
        layer.weight = torch.nn.Parameter(-A)
        layer.bias = torch.nn.Parameter(b)
        self.layer, self.l, self.n = layer, len(A), n

    def forward(self, x):
        s = - self.l + 1
        y = self.layer(x)
        s += torch.sum(nn.ReLU()(self.n * y + 1), -1, keepdim=True) - torch.sum(nn.ReLU()(self.n * y), -1, keepdim=True)
        return nn.ReLU()(s)

# Generate multiple support networks to match one confusion network
class Unlearn_multipoints:
    def __init__(self, model, n, bounds, test_model=False):
        self.model = model
        self.n = n
        self.bounds = bounds
        self.test_model = test_model
        self.g_list, self.cd_list = [], []
        self.total_constraint_num = []
        self.buggy_points = None
        self.linear_regions = None
        self.P = None
        self.ql = None
        self.qu = None
        self.remove_redundant_constraint = None

    def point_wise_unlearn(self, buggy_points,buggy_points_,P, ql, qu, remove_redundant_constraint):
        assert self.linear_regions is None
        self.buggy_points = buggy_points
        self.buggy_points_ = buggy_points_
        self.P, self.ql, self.qu = P, ql, qu
        self.remove_redundant_constraint = remove_redundant_constraint

    def PatchForOnePoint(self, nn_input: torch.Tensor, P_i, ql_i, qu_i, is_gurobi):
        nn_input = nn_input.view(-1)
        nn_input.requires_grad = True
        A, b = linear_region_from_input(nn_input, self.model.all_hidden_neurons(nn_input).view(-1), self.bounds)
        x_dim = len(A[0])
        # simplified_A, simplified_b = A[2*x_dim:], b[2*x_dim:]
        simplified_A, simplified_b = A[2*x_dim:], b[2*x_dim:]
        support_nn = SupportNN(simplified_A, simplified_b, self.n)
        f1, f2 = LinearizedNN(nn_input, self.model)  # f1, f2 are np.ndarray
        cd = LinearPatchNN(A, b, P_i, ql_i, qu_i, f1, f2)
        return support_nn, cd, len(simplified_A)  # len(simplified_A) is for computation of parameters overhead


    ##################################
    def PatchForOnePoint1(self, nn_input_: torch.Tensor):
        nn_input_ = nn_input_.view(-1)
        nn_input_.requires_grad = True
        A_, b_ = linear_region_from_input(nn_input_, self.model.all_hidden_neurons(nn_input_).view(-1), self.bounds)
        x_dim_ = len(A_[0])
        # simplified_A, simplified_b = A[2*x_dim:], b[2*x_dim:]
        simplified_A, simplified_b = A_[2*x_dim_:], b_[2*x_dim_:]
        support_nn = SupportNN(simplified_A, simplified_b, self.n)
        return support_nn  # len(simplified_A) is for computation of parameters overhead
    ###################################

    def compute(self, core_num=multiprocessing.cpu_count()-1, is_gurobi=False):
        assert (self.buggy_points is not None) or (self.linear_regions is not None)
        print('Working on {} cores.'.format(core_num))
        pool = multiprocessing.Pool(core_num)
        if self.buggy_points is not None:
            # print([len(self.buggy_points),len(self.P),len(self.ql),len(self.qu)])
            arg_list = [[self.buggy_points[i], self.P[i], self.ql[i], self.qu[i], is_gurobi] for i in
                        range(len(self.buggy_points))]
            arg_list1 = [[self.buggy_points_[i]] for i in
                        range(len(self.buggy_points_))]
            res = pool.starmap(self.PatchForOnePoint, arg_list)
            res1 = pool.starmap(self.PatchForOnePoint1, arg_list1)
        else:
            arg_list = [[self.linear_regions[i][0], self.linear_regions[i][1], self.linear_regions[i][2], self.P[i],
                         self.ql[i], self.qu[i]] for i in range(len(self.linear_regions))]
            res = pool.starmap(self.PatchForOneLinearRegion, arg_list)
        pool.close()
        cd_list=[res_g[1] for res_g in res]
        print(len(res1))
        g_list= res1
        self.total_constraint_num = [res_g[2] for res_g in res]
        h = MultiPRUNE(g_list, cd_list, self.bounds)
        # print('avg_constraint_num:', sum(self.total_constraint_num) / len(self.total_constraint_num))
        # return NNSum(self.model, h)
        return h

def LinearizedNN(input, model):
    f1, f2 = [], []
    for output in model(input).squeeze():
        model(input)
        output.backward(retain_graph=True)
        f1.append(input.grad.clone().detach())
        f2.append(output - torch.matmul(input.grad, input))
    return torch.stack(f1).detach().numpy(), torch.stack(f2).detach().numpy()

def OneDimLinearTransform(lb, ub, b, ql, qu):
    # [lb, ub]: objective interval, b:
    # [ql, qu]: require interval
    if ub - lb <= qu - ql:
        alpha = 1.0
    else:
        alpha = (qu - ql)/(ub - lb)
        lb, ub = alpha*lb, alpha*ub
    if lb + b >= ql and ub + b <= qu:
        beta = 0.0
    elif lb + b < ql:
        beta = ql - lb - alpha*b
    else:
        beta = qu - ub - alpha*b
    return alpha, beta

def LinearPatchNN(A: np.ndarray, b: np.ndarray, P: np.ndarray, ql: np.ndarray, qu: np.ndarray, f1, f2, my_dtype=torch.float):

    l_P = len(P)
    assert np.linalg.matrix_rank(P) == l_P
    P = OrthogonalComplement(P)
    P_inv = np.linalg.inv(P)
    alpha = np.eye(len(P[0]))
    beta = np.zeros(len(P[0]))
    for i in range(l_P):
        tem1 = linprog(np.matmul(P[i], f1), A, b, bounds=(None, None)).fun
        tem2 = -linprog(-np.matmul(P[i], f1), A, b, bounds=(None, None)).fun
        alpha[i, i], beta[i] = OneDimLinearTransform(tem1, tem2, np.matmul(P[i], f2), ql[i], qu[i])
    trans_matrix = np.matmul(P_inv, np.matmul(alpha, P)) - np.eye(len(P[0]))
    c = np.matmul(trans_matrix, f1)
    d = np.matmul(trans_matrix, f2) + np.matmul(P_inv, beta)
    return torch.tensor(c, dtype=my_dtype), torch.tensor(d, dtype=my_dtype)

def OrthogonalComplement(P):
    n, m = len(P), len(P[0])
    if n >= m: return P
    orthogonal_space = []
    P = np.concatenate([P, np.zeros([m-n, m])], axis=0)
    w, v = np.linalg.eig(P)
    epsilon = 0.1**5
    for i in range(m):
        if np.abs(w[i]) < epsilon:
            orthogonal_space.append(v[i]/np.linalg.norm(v[i], 2))
    P[n:] = orthogonal_space
    return P

# generate patch network
class MultiPRUNE(nn.Module):
    def __init__(self, g_list, cd_list, bounds):
        super(MultiPRUNE, self).__init__()
        self.K_list, self.layer_list = [], []
        temp_c, temp_d = 0, 0
        bound_A = np.concatenate([-np.eye(len(bounds[0])), np.eye(len(bounds[1]))])
        bound_b = np.concatenate([-bounds[0], bounds[1]])
        for c, d in cd_list:
            c, d = c - temp_c, d - temp_d
            temp_c, temp_d = c + temp_c, d + temp_d
            K = torch.tensor(0.0)
            for i in range(len(c)):
                out_lowerbound = linprog(c[i], bound_A, bound_b, bounds=[None, None]).fun + d[i]
                out_upperbound = -linprog(-c[i], bound_A, bound_b, bounds=[None, None]).fun + d[i]
                abs_bound = max(abs(out_upperbound), abs(out_lowerbound))
                K = torch.max(K, torch.tensor(abs_bound, dtype=torch.float))
            self.K_list.append(K)
            layer = nn.Linear(*c.size())
            layer.weight = torch.nn.Parameter(c)
            layer.bias = torch.nn.Parameter(d)
            self.layer_list.append(layer)
        self.g_list = g_list
    def forward(self, x):
        res = 0
        for i in range(len(self.K_list)):
            if i < len(self.g_list) - 1:
                y = [self.g_list[k](x) for k in range(i, len(self.g_list))]
                g_max = torch.max(torch.cat(y, dim=-1), dim=-1).values.unsqueeze(1)
            else:
                g_max = self.g_list[len(self.g_list) - 1](x)
            res += nn.ReLU()(self.layer_list[i](x) + self.K_list[i]*g_max-self.K_list[i]) \
               - nn.ReLU()(-self.layer_list[i](x) + self.K_list[i]*g_max-self.K_list[i])
        return res