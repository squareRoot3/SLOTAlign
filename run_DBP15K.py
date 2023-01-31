import numpy as np
import torch
import json
import os
import dgl
from dgl.nn import GraphConv
import time


def my_check_align1(pred, ground_truth, result_file=None):
    g_map = {}
    for i in range(ground_truth.size(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())
    ind = (-pred).argsort(axis=1)[:, :10]
    a1, a5, a10 = 0, 0, 0
    for i, node in enumerate(g_list):
        for j in range(10):
            if ind[node, j].item() == g_map[node]:
                if j < 1:
                    a1 += 1
                if j < 5:
                    a5 += 1
                if j < 10:
                    a10 += 1
    a1 /= len(g_list)
    a5 /= len(g_list)
    a10 /= len(g_list)
    print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%%' % (a1 * 100, a5 * 100, a10*100))
    return a1,a5,a10


def gw_torch(cost_s, cost_t, p_s=None, p_t=None, trans=None, beta=0.01, error_bound=1e-6, outer_iter=1000, inner_iter=10):
    device = cost_s.device
    gt = torch.tensor(np.array(list(range(cost_s.shape[0]))), device=device)
    if p_s is None:
        p_s = torch.ones([cost_s.shape[0],1], device=device)/cost_s.shape[0]
    if p_t is None:
        p_t = torch.ones([cost_t.shape[0],1], device=device)/cost_t.shape[0]
    if trans is None:
        trans = p_s @ p_t.T
    obj_list = []
    for oi in range(outer_iter):
        cost = - 2 * (cost_s @ trans @ cost_t.T)
        kernel = torch.exp(-cost / beta) * trans
        a = torch.ones_like(p_s)/p_s.shape[0]
        for ii in range(inner_iter):
            b = p_t / (kernel.T@a)
            a_new = p_s / (kernel@b)
            relative_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
            a = a_new
            if relative_error < 1e-6:
                break
        trans = (a @ b.T) * kernel
        if oi % 20 == 0 and oi > 2:
            # print('acc: ', (trans.argmax(1) == gt).sum()/trans.shape[0])
            obj = (cost_s ** 2).mean() + (cost_t ** 2).mean() - \
                torch.trace(cost_s @ trans @ cost_t @ trans.T)
            print(oi, obj)
            if len(obj_list) > 0:
                print('obj gap: ', torch.abs(obj-obj_list[-1])/obj_list[-1])
                if torch.abs(obj-obj_list[-1])/obj_list[-1] < error_bound:
                    print('iter:{}, smaller than eps'.format(ii))
                    break
            obj_list.append(obj.item())
    return trans


def SLOTAlign(As, Bs, X, step_size=1, gw_beta=0.01, joint_epoch=100, gw_epoch=2000):
    layers = As.shape[2]
    alpha0 = np.ones(layers).astype(np.float32)/layers
    beta0 = np.ones(layers).astype(np.float32)/layers
    Adim, Bdim = As.shape[0], Bs.shape[0]
    a = torch.ones([Adim,1]).cuda()/Adim
    b = torch.ones([Bdim,1]).cuda()/Bdim
    for ii in range(joint_epoch):
        alpha = torch.autograd.Variable(torch.tensor(alpha0)).cuda()
        alpha.requires_grad = True
        beta = torch.autograd.Variable(torch.tensor(beta0)).cuda()
        beta.requires_grad = True
        A = (As * alpha).sum(2)
        B = (Bs * beta).sum(2)
        objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ X @ B @ X.T)
        alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
        alpha = alpha - step_size * alpha_grad
        alpha0 = alpha.detach().cpu().numpy()
        alpha0 = euclidean_proj_simplex(alpha0)
        beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
        beta = beta - step_size * beta_grad
        beta0 = beta.detach().cpu().numpy()
        beta0 = euclidean_proj_simplex(beta0)
        X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=gw_beta, outer_iter=1,
                     inner_iter=50).clone().detach()
        if ii == 99:
            X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X, beta=0.001, outer_iter=gw_epoch, inner_iter=2)
    return X


def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def NeuralSinkhorn(cost, p_s=None, p_t=None, trans=None, beta=0.1, outer_iter=20):
    if p_s is None:
        p_s = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    if p_t is None:
        p_t = torch.ones([cost.shape[1],1],device=cost.device)/cost.shape[1]
    if trans is None:
        trans = p_s @ p_t.T
    a = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    cost_new = torch.exp(-cost / beta)
    for oi in range(outer_iter):
        kernel = cost_new * trans
        b = p_t / (kernel.T@a)
        a = p_s / (kernel@b)
        trans = (a @ b.T) * kernel
    return trans


def run_DBP15K(lang, translate=True, layers=2, gw_beta=0.001, gw_epoch=1000, step_size=1):
    file = np.load('dbp/KG{}.npz'.format(lang))
    sparse_rel_matrix = file['mat']
    test_pair = file['test_pair']
    BERTfeature = np.load('dbp/LaBSE_{}.npy'.format(lang))
    feature = torch.tensor(BERTfeature)
    node_size = feature.shape[0]
    rel_mat = torch.sparse_coo_tensor(sparse_rel_matrix[:,:2].T, np.ones_like(sparse_rel_matrix[:,2]), (node_size,node_size)).float()
    idx = rel_mat.coalesce().indices()
    G = dgl.graph((idx[0], idx[1]))
    rel_mat_dense = G.adj().to_dense()
    Aadj = rel_mat_dense[test_pair[:,0]][:,test_pair[:,0]]
    Badj = rel_mat_dense[test_pair[:,1]][:,test_pair[:,1]]
    Aadj/=Aadj.max()
    Badj/=Badj.max()

    time_st = time.time()
    new_feature = feature / (feature.norm(dim=1)[:, None]+1e-16)
    features = [new_feature]
    conv = GraphConv(0, 0, norm='both', weight=False, bias=False)
    for i in range(layers):
        new_feature = conv(G,features[-1])
        new_feature = new_feature / (new_feature.norm(dim=1)[:, None]+1e-16)
        features.append(new_feature)
    Asims, Bsims = [Aadj], [Badj]
    for feature in features:
        Asims.append(feature[test_pair[:,0]].mm(feature[test_pair[:,0]].T))
        Bsims.append(feature[test_pair[:,1]].mm(feature[test_pair[:,1]].T))

    sim1 = features[0][test_pair[:,0]].mm(features[0][test_pair[:,1]].T)
    initX=NeuralSinkhorn(1-sim1.float().cuda())
    As = torch.stack(Asims,dim=2).cuda()
    Bs = torch.stack(Bsims,dim=2).cuda()
    X = SLOTAlign(As, Bs, initX, step_size=step_size, gw_beta=gw_beta, gw_epoch=gw_epoch)
    ground_truth = torch.cat([torch.tensor(list(range(4500,X.shape[0]))).unsqueeze(0),
                            torch.tensor(list(range(4500,X.shape[0]))).unsqueeze(0)], 0)
    a1,a5,a10 = my_check_align1(X, ground_truth)
    time_ed = time.time()
    with open('result.txt', 'a+') as f:
        f.write('Lang: {} Trans {} Layers {} beta {} H@1 {:.3f} H@5 {:.3f} H@10 {:.3f} Time {:.2f}\n'.format(
            lang, translate, layers, gw_beta, a1, a5, a10, time_ed - time_st))


run_DBP15K('fr', translate=False, layers=2, gw_beta=0.01, gw_epoch=500, step_size=1)
run_DBP15K('ja', translate=False, layers=2, gw_beta=0.01, gw_epoch=500, step_size=1)
run_DBP15K('zh', translate=False, layers=2, gw_beta=0.01, gw_epoch=500, step_size=1)
