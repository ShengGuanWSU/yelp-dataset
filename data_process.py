# -*- coding: utf-8 -*-
import bz2
import cPickle
import numpy as np


def main():
    file_name = 'yelp_2014_2015_case_0.pkl'
    yelp_data_0 = cPickle.load(bz2.BZ2File(file_name))
    # main related attributes.
    # costs: edge costs are all set to 1.0
    # edges: list of edges
    # data_matrix: the node x feature matrix
    # words_dict: keywords map
    # nodes_dict: each node stands for a restaurant.
    # k: true subgraph size
    # m: number of edges
    # n: number of nodes
    print('number of nodes: %d' % yelp_data_0['n'])
    print('number of edges: %d' % yelp_data_0['m'])
    m, n = yelp_data_0['data_matrix'].shape
    print('size of data_matrix: %d x %d' % (m, n))
    print('true subgraph size: %d' % yelp_data_0['k'])


if __name__ == '__main__':
    main()
