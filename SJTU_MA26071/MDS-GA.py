"""
Programme for find the minimal dominating set by greedy algorithm.
For more information about the concepts and implementations, refer to corresponding report (MDS_GA_report.pdf).
"""

import numpy as np
import math


class Graph:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix  # adjacency-matrix representation
        self.Delta = np.max(self.adj_matrix.sum(axis=0))  # the maximal degree of the vertices of a graph G
        self.delta = np.min(self.adj_matrix.sum(axis=0))  # the minimal degree of the vertices of a graph G
        # all vertices are white initially
        self.num_choice = self.adj_matrix.sum(axis=0)
        self.num_choice_matrix = -np.ones(self.adj_matrix.shape)
        # current dominating set and minimal dominating set
        self.dom = np.zeros(self.adj_matrix.shape[0])  # white 0; blue -1; red 1
        self.min_size = self.adj_matrix.shape[0]
        self.min_dom = np.ones(self.adj_matrix.shape[0])


class MDS:
    def __init__(self, config):
        self.config=config
        self.Delta =self.config.Delta
        self.delta = self.config.delta
        self.adj_matrix = self.config.adj_matrix
        self.num_choice = self.config.num_choice
        self.num_choice_matrix = self.config.num_choice_matrix
        self.dom = self.config.dom
        self.min_size = self.config.min_size
        self.min_dom = self.config.min_dom

    # add u to dom, call the routine recursively
    # u: label of current vertex
    # is_nback: whether it is the first step of backtrack process for validation.
    # is_nback: True: is not 1st backtrack; False: is 1st backtrack.
    # gamma.shape[0] ==0 means u is isolated
    def min_dom_set(self, u, is_nback):
        if u == 0:
            self.num_choice = self.adj_matrix.sum(axis=0)
        else:
            self.num_choice = self.num_choice_matrix[u-1]
        if u == self.adj_matrix.shape[0]:
            return self
        else:
            # label of vertex u's blue neighbourhood
            v = np.intersect1d(np.argwhere(self.dom == -1).reshape(np.argwhere(self.dom == -1).shape[0]),
                               np.argwhere(self.adj_matrix[u] == 1).reshape(np.argwhere(self.adj_matrix[u] == 1).shape[0]))
            # label of vertex u's neighbourhood
            gamma = np.argwhere(self.adj_matrix[u] == 1).reshape(int(self.adj_matrix[u].sum(axis=0)))
            if np.argwhere(self.num_choice[v] == 1).shape[0] == 0 and gamma.shape[0] != 0 and is_nback:
                # dye u blue
                self.num_choice[gamma] -= 1
                self.dom[u] = -1
            else:
                # dye u red
                self.dom[u] = 1
            self.num_choice_matrix[u] = self.num_choice
            # Uncomment the line below if you wish to view the results step-by-step.
            # print("u={},gamma={},v={},dom={},num_choice={}".format(u, gamma, v, self.dom, self.num_choice))
            return self.min_dom_set(u + 1, True)

    def greedy_search(self):
        self.min_dom_set(0, True)
        self.min_size = list(self.dom).count(1)
        self.min_dom = self.dom
        for i in range(self.dom.shape[0]):
            u = self.dom.shape[0] - i - 1
            # return to the blue vertex and try dyeing it red
            if self.dom[u] == -1:
                self.dom = np.append(self.dom[:u], np.zeros(i + 1), axis=0)
                self.dom[u] = 1
                self.min_dom_set(u, False)
                if list(self.dom).count(1) < self.min_size:
                    self.min_dom = self.dom
                    self.min_size = list(self.dom).count(1)
        return self

    def upper_bound(self):
        upperbound = self.adj_matrix.shape[0]*(math.log(self.delta+1)+1)/(self.delta+1)
        return int(upperbound)


# n: the order of the graph
def adj_generate(n, is_use_example=True, threshold=0.5):
    if is_use_example:
        adj_example = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                                [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                                [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                                [0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
                                [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]])
        return adj_example
    else:
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                a = 1 if np.random.rand() >= threshold else 0
                adj_matrix[j][i] = int(a)
                adj_matrix[i][j] = int(a)
        return adj_matrix


if __name__ == '__main__':
    graph_order = 100
    G = Graph(adj_generate(graph_order, False))
    mds = MDS(G)
    mds.greedy_search()
    print("The size of the Mimnimal Dominating Set is {}.\n"
          "The derived upper bound of given graph is {}.\n".format(mds.min_size, mds.upper_bound()))
