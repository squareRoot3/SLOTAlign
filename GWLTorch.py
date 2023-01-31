import numpy as np
import torch
import networkx as nx
import random
import copy
from utils import *


def gw_torch(cost_s, cost_t, p_s=None, p_t=None, trans0=None, beta = 1e-1, error_bound = 1e-10,
                             outer_iter = 200, inner_iter = 1, gt=None):
    # a = torch.ones_like(p_s)/p_s.shape[0]
    if trans0 is None:
        trans0 = p_s @ p_t.T
    for oi in range(outer_iter):
        a = torch.ones_like(p_s)/p_s.shape[0]
        cost = - 2 * (cost_s @ trans0 @ cost_t.T)
        kernel = torch.exp(-cost / beta) * trans0
        for ii in range(inner_iter):
            b = p_t / (kernel.T@a)
            a_new = p_s / (kernel@b)
            relative_error = torch.sum(torch.abs(a_new - a)) / torch.sum(torch.abs(a))
            a = a_new
            if relative_error < error_bound:
                break
        trans = (a @ b.T) * kernel
        relative_error = torch.sum(torch.abs(trans - trans0)) / torch.sum(torch.abs(trans0))
        if relative_error < error_bound:
            print(relative_error)
            break
        trans0 = trans
        if oi % 20 == 0 and oi > 2:
            if gt is not None:
                res=trans0.T.cpu().numpy()
                a1,a5,a10 = my_check_align1(res, gt)
            print(oi, (cost_s ** 2).mean() + (cost_t ** 2).mean()-torch.trace(cost_s @ trans @ cost_t @ trans.T),a1,a5,a10)
    return trans


def add_noisy_edges(graph: nx.graph, noisy_level: float) -> nx.graph:
    nodes = list(graph.nodes)
    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges)
    graph_noisy = copy.deepcopy(graph)
    if num_noisy_edges > 0:
        i = 0
        while i < num_noisy_edges:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if (src, dst) not in graph_noisy.edges:
                graph_noisy.add_edge(src, dst)
                i += 1
    return graph_noisy


def add_noisy_nodes(graph: nx.graph, noisy_level: float) -> nx.graph:
    num_nodes = len(graph.nodes)
    num_noisy_nodes = int(noisy_level * num_nodes)

    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges / num_nodes + 1)

    graph_noisy = copy.deepcopy(graph)
    if num_noisy_nodes > 0:
        for i in range(num_noisy_nodes):
            graph_noisy.add_node(int(i + num_nodes))
            j = 0
            while j < num_noisy_edges:
                src = random.choice(list(range(i + num_nodes)))
                if (src, int(i + num_nodes)) not in graph_noisy.edges:
                    graph_noisy.add_edge(src, int(i + num_nodes))
                    j += 1
    return graph_noisy


def node_correctness(coup, perm_inv):
    coup_max = coup.argmax(1)
    perm_inv_max = perm_inv.argmax(1)
    acc = np.sum(coup_max == perm_inv_max) / len(coup_max)
    return acc
