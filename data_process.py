# -*- coding: utf-8 -*-
# packages need to install
# pip install geopy nltk networkx numpy
from __future__ import print_function
import os
import sys
import csv
import time
import json
import string
import random
import os.path
import operator
import numpy as np
import networkx as nx

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from geopy.distance import vincenty
from sklearn.feature_extraction.text import TfidfTransformer


def get_distance((id_current, id_next, current_node, next_node)):
    distance = vincenty(current_node, next_node).meters
    return id_current, id_next, distance


def get_business_graph(file_path_, nearest_nei_size):
    business_dict = dict()
    for line in open(file_path_).readlines():
        each_record = json.loads(line)
        business_id_ = each_record["business_id"]
        lati_ = float(each_record["latitude"])
        longi_ = float(each_record["longitude"])
        business_dict[business_id_] = [lati_, longi_]
    assert len(business_dict) == 25881
    print('total number of restaurant: ', len(business_dict))
    pool = Pool(processes=8)
    list_nodes = [item_ for item_ in business_dict]
    index = 0
    for i in range(len(list_nodes)):
        id_current = list_nodes[i]
        lati_ = business_dict[id_current][0]
        longi_ = business_dict[id_current][1]
        current_node = (lati_, longi_)
        list_edges = []
        for j in range(len(list_nodes)):
            id_next = list_nodes[j]
            if id_next == id_current:
                continue
            next_lati = business_dict[id_next][0]
            next_longi = business_dict[id_next][1]
            next_node = (next_lati, next_longi)
            list_edges.append((id_current, id_next, current_node, next_node))
        list_distance = []
        for item_ in pool.map(get_distance, list_edges):
            list_distance.append(item_)
        list_distance.sort(key=lambda x: x[2])
        with open(output_root_path + 'processed_selected_business_graph.csv',
                  'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)
            for item_ in list_distance[:nearest_nei_size]:
                csv_writer.writerow(item_)
        if index % 100 == 0:
            print('processed: ', index)
        index += 1
    return business_dict


def pre_processing_data(file_path_):
    all_review_records = open(file_path_).readlines()
    index = 0
    output_file_path = output_root_path + 'processed_review.json'
    fp = open(output_file_path, 'a')
    for review in all_review_records:
        review_data = json.loads(review)
        business_id = review_data["business_id"]
        date = review_data["date"]
        review_sentence = review_data["text"]
        words = word_tokenize(review_sentence)
        filtered_words = [word.lower() for word in words if
                          word.lower() not in stopwords_]
        filtered_words = [word for word in filtered_words if
                          word not in punctuations_]
        month_ = date[0:7]
        fp.write(json.dumps([business_id, month_, filtered_words]) + '\r\n')
        index += 1
        if index % 10000 == 0:
            print(index)
    fp.close()

    print('finish the pre-processing.')


def test(file_path_):
    index = 0
    all_review_records = open(file_path_).readlines()
    business_data = dict()
    months_dict = dict()
    time.sleep(10000)
    for review in all_review_records:
        review_data = json.loads(review)
        business_id = review_data["business_id"]
        date = review_data["date"]
        review_sentence = str(review_data["text"])
        print(str(review_sentence))
        words = word_tokenize(review_sentence)
        print(words)
        filtered_words = [word for word in words if
                          word not in stopwords_ and word not in punctuations_]
        print(filtered_words)
        print('-------------------')
        month_ = date[0:7]
        if month_ not in months_dict:
            months_dict[month_] = 0
        months_dict[month_] += 1
        if business_id not in business_data:
            business_data[business_id] = dict()
        if month_ not in business_data[business_id]:
            business_data[business_id][month_] = []
        business_data[business_id][month_].append(review_data)
        if index % 10000 == 0:
            print('processed: ', index)
        index += 1
        time.sleep(2)
    for item_ in months_dict:
        print(item_, months_dict[item])


def load_processed_data():
    processed_data = dict()
    index = 0
    with open(output_root_path + "processed_review.json") as file_:
        for each_line in file_:
            record_ = json.loads(each_line)
            if record_[0] not in processed_data:
                processed_data[record_[0]] = []
            processed_data[record_[0]].append([record_[1], record_[2]])
            if index % 10000 == 0:
                print(index)
            index += 1
    print(
        'load data finished. we have {} records.'.format(len(processed_data)))
    return processed_data


def generate_graph(num_of_nearest_nei):
    file_path_ = output_root_path + "processed_selected_business_graph.csv"
    edge_list = dict()
    nodes_dict = dict()
    with open(file_path_) as csvFile:
        data_reader = csv.reader(csvFile, delimiter=',', skipinitialspace=True)
        for item_ in data_reader:
            if item_[0] not in edge_list:
                edge_list[item_[0]] = []
            edge_list[item_[0]].append([item_[1], float(item_[2])])
    g = nx.Graph()
    index = 0
    for item_ in edge_list:
        if item_ not in nodes_dict:
            nodes_dict[item_] = index
            index += 1
        edge_list[item_].sort(key=lambda x: x[1])
        for each_nei in edge_list[item_][:num_of_nearest_nei]:
            g.add_edge(item_, each_nei[0], weight=1.0)
    connected_components = nx.connected_component_subgraphs(g)
    print('number of nodes in graph: ', g.number_of_nodes())
    print('number of edges in graph: ', g.number_of_edges())
    index = 0
    random_nodes = []
    for cc in connected_components:
        index += 1
        random_node = list(cc)[random.randint(0, len(cc) - 1)]
        random_nodes.append(random_node)
        print('number of node in connected component: ', len(cc))
    print('total number of connected components: ', index)
    for i in range(len(random_nodes) - 1):
        g.add_edge(random_nodes[i], random_nodes[i + 1], weight=1.0)
    if nx.number_connected_components(g) != 1:
        print('the graph is not connected ...')
        sys.exit(0)
    file_path_ = 'processed_graph_' + str(num_of_nearest_nei) + '.txt'
    if os.path.isfile(output_root_path + file_path_):
        print('finished load')
        return g
    else:
        fp = open(output_root_path + file_path_, 'w')
        sorted_nodes = sorted(nodes_dict.items(), key=operator.itemgetter(1))
        for item_ in sorted_nodes:
            fp.write('{} {}\n'.format(item_[0], item_[1]))
        for edge in g.edges():
            i = nodes_dict[edge[0]]
            j = nodes_dict[edge[1]]
            fp.write('{} {}\n'.format(i, j))
        fp.close()
        print('finished load')
        return g


def load_graph(file_name):
    nodes_dict = dict()
    g = nx.Graph()
    with open(output_root_path + file_name) as f:
        for each_line in f.readlines():
            i = each_line.split(' ')[0]
            j = each_line.rstrip().split(' ')[1]
            if i.isdigit():
                g.add_edge(int(i), int(j), weight=1.0)
            else:
                nodes_dict[i] = int(j)
    business_id_list = [0] * len(nodes_dict)
    for node_ in nodes_dict:
        business_id_list[nodes_dict[node_]] = node_
    print('total number of nodes: ', g.number_of_nodes())
    print('total number of edges: ', g.number_of_edges())
    if nx.number_connected_components(g) != 1:
        print('the graph is not connected ...')
        sys.exit(0)
    else:
        print('connected graph. finish to load graph ...')
    return nodes_dict, business_id_list, g


def generate_data_by_month_():
    month_list = []
    with open(output_root_path + 'processed_date_by_month.txt') as f:
        for each_line in f.readlines():
            month_ = str(each_line.lstrip().rstrip())
            month_list.append(month_)
    for item_ in month_list:
        print('processing {}'.format(item_))
        index = 0
        fp = open(output_root_path + 'processed_{}.txt'.format(item_), 'a')
        with open(output_root_path + "processed_review.json") as file_:
            for each_line in file_:
                record_ = json.loads(each_line)
                if record_[1] == item_:
                    fp.write(json.dumps(record_) + '\r\n')
                    index += 1
        fp.close()
        print('total number of records in {} is {}'.format(item_, index))


def get_data_file_list():
    file_list = []
    for file_path in sorted(os.listdir(output_root_path)):
        if file_path.startswith('processed_records_') and file_path.endswith(
                '.json'):
            file_list.append(output_root_path + file_path)
    return file_list


def process_each_data_file(file_path):
    data_records = []
    business_id = dict()
    print('file: ', file_path)
    with open(file_path) as _file_:
        for each_line_ in _file_:
            record = json.loads(each_line_)
            business_id[record[0]] = 1
            data_records.append(record)
    print('total number of records is {}, {} business id involved.'.format(
        len(data_records), len(business_id)))


def generate_data_set():
    for year in ['2004', '2005', '2006', '2007', '2008', '2009', '2010',
                 '2011', '2012', '2013', '2014', '2015',
                 '2016']:
        index = 0
        fp = open(output_root_path + 'processed_records_{}.json'.format(year),
                  'a')
        with open(output_root_path + "processed_review.json") as file_:
            for each_line in file_:
                record_ = json.loads(each_line)
                if record_[1].startswith(year):
                    fp.write(json.dumps(record_) + '\r\n')
                    index += 1
        fp.close()
        print('total number of records in {} is {}'.format(year, index))
        # generate_each_file()
        # load_graph(1)
        # get_business_graph(root_path +
        # 'yelp_academic_dataset_business.json', 200)
        # pre_processing_data(root_path + 'yelp_academic_dataset_review.json')


def select_restaurant_related_id():
    categories = dict()
    with open(root_path + 'yelp_academic_dataset_business.json') as f:
        for each_line in f.readlines():
            record = json.loads(each_line)
            # for item in record:
            #    print(item, ' : ', record[item])
            # print('-----------------------------------------------------')
            # time.sleep(1)
            if 'Restaurants' in record['categories']:
                for item_ in record['categories']:
                    if item_ not in categories:
                        categories[item_] = 0
                    categories[item_] += 1
    index = 0
    top_categories = dict()
    for item_ in sorted(categories.items(), key=operator.itemgetter(1),
                        reverse=True):
        print(index, item_)
        index += 1
        if index >= 71:
            break
        if item_[0] != 'Restaurants':
            top_categories[item_[0]] = 1
            # generate_data_set()
            # for item in get_data_file_list():
            #    process_each_data_file(item)
    total_selected = 0
    selected_business_records = dict()
    fp = open(output_root_path + 'processed_selected_business.json', 'w')
    with open(root_path + 'yelp_academic_dataset_business.json') as f:
        for each_line in f.readlines():
            record = json.loads(each_line)
            if 'Restaurants' in record['categories']:
                for item_ in [it_ for it_ in record['categories'] if
                              item != 'Restaurants']:
                    if item_ in top_categories:
                        total_selected += 1
                        selected_business_records[record['business_id']] = \
                            record['categories']
                        fp.write(json.dumps(record) + '\r\n')
                        break
    fp.close()
    print('total number of top_k restaurants is {}'.format(total_selected))
    print(
        '-------------------------------------------------------------------')
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0
    num_6 = 0
    num_7 = 0
    for item_ in selected_business_records:
        if len(selected_business_records[item_]) == 2:
            num_2 += 1
        if len(selected_business_records[item_]) == 3:
            num_3 += 1
        if len(selected_business_records[item_]) == 4:
            num_4 += 1
        if len(selected_business_records[item_]) == 5:
            num_5 += 1
        if len(selected_business_records[item_]) == 6:
            num_6 += 1
        if len(selected_business_records[item_]) == 7:
            num_7 += 1
    print(num_2, num_3, num_4, num_5, num_6, num_7)


def get_categories(file_path_, nodes_dict):
    categories_dict = dict()
    with open(file_path_) as f:
        for each_line in f.readlines():
            record = json.loads(each_line)
            id_ = nodes_dict[record['business_id']]
            for category in record['categories']:
                if category not in categories_dict:
                    categories_dict[category] = []
                categories_dict[category].append(id_)
    print('total number of categories: ', len(categories_dict))
    cat_list = [[item_, len(categories_dict[item_])] for item_ in
                categories_dict]
    for item_id in sorted(cat_list, key=operator.itemgetter(1), reverse=True):
        if item_id[1] >= 1000:
            print(item_id)
    print('finished to get categories ...')
    return categories_dict


def load_reviews_data(each_file_path, nodes_dict, selected_type,
                      word_freq_threshold_lower, word_freq_threshold_upper):
    file_path_ = output_root_path + 'processed_selected_business.json'
    categories_dict = get_categories(file_path_, nodes_dict)
    all_review_records = []
    for each_ in each_file_path:
        all_review_records = all_review_records + open(each_).readlines()
    reviews_list = []
    total_words_dict = dict()
    for each_line in all_review_records:
        record = json.loads(each_line)
        if record[0] in nodes_dict:
            reviews_list.append([nodes_dict[record[0]], record[2]])
            for word in record[2]:
                if word not in total_words_dict:
                    total_words_dict[word] = 0
                total_words_dict[word] += 1
    print('selected type: ', selected_type)
    print('number of reviews in {} is {}'.format(each_file_path,
                                                 len(reviews_list)))
    print('finished to load reviews ...')
    selected_nodes = categories_dict[selected_type]
    selected_nodes_dict = dict()
    for item_ in selected_nodes:
        selected_nodes_dict[item_] = 1.0
    words_dict_threshold = dict()
    print('total number of keywords in raw data: ', len(total_words_dict))
    for item_ in total_words_dict:
        if total_words_dict[item_] >= word_freq_threshold_lower:
            words_dict_threshold[item_] = total_words_dict[item_]
    total_words_dict = words_dict_threshold
    print('remove the keywords <={}'.format(word_freq_threshold_lower))
    print('total number of retained keywords: ', len(total_words_dict))
    return categories_dict, reviews_list, selected_nodes_dict, total_words_dict


def get_selected_matrix_df_idf(nodes_id_list, total_words_dict, reviews_list):
    print('start calculating df and idf ....')
    counts_matrix = []
    for i in range(len(nodes_id_list)):
        counts_matrix.append([0.0] * len(total_words_dict))
    total_words_list = []
    for item_ in total_words_dict:
        total_words_list.append(item_)
    for i in range(len(total_words_list)):
        total_words_dict[total_words_list[i]] = i
    for item_ in reviews_list:
        for word in item_[1]:
            if word in total_words_dict:
                counts_matrix[item_[0]][total_words_dict[word]] += 1.0
    print('starts to run fit_transform ....')
    transformer = TfidfTransformer(smooth_idf=False)
    tf_idf = transformer.fit_transform(counts_matrix)
    print('fit_transform finished ....')
    selected_keywords = dict()
    zeros_ = 0
    non_zeros = 0
    tf_idf_matrix = tf_idf.toarray()
    index_processed = 0
    for tf_idf_array in tf_idf_matrix:
        if np.sum(tf_idf_array) == 0.0:
            zeros_ += 1
        else:
            arr_with_id = []
            index = 0
            for entry_ in tf_idf_array:
                arr_with_id.append([entry_, index])
                index += 1
            non_zeros += 1
            arr_with_id.sort(reverse=True)
            for item_ in arr_with_id[:5]:
                if item_[0] > 0.0:
                    selected_keywords[total_words_list[item_[1]]] = 1
            if index_processed % 1000 == 0:
                print('processed: {}'.format(index_processed))
        index_processed += 1
    print('number of zeros: ', zeros_)
    print('number of non-zeros: ', non_zeros)
    print('number of selected keywords: ', len(selected_keywords))
    words_dict = dict()
    words_id_list = []
    index = 0
    for item_ in selected_keywords:
        words_dict[item_] = index
        words_id_list.append(item_)
        index += 1
    time.sleep(100000)
    return words_dict, words_id_list


def generate_shuffle_data(reviews_list, selected_nodes_dict, nodes_dict,
                          words_dict):
    reviews_in_list = []
    reviews_out_list = []
    selected_anomalous_words_dict = dict()
    n = len(nodes_dict)
    for reviews in reviews_list:
        if reviews[0] in selected_nodes_dict:
            # we do not keep the reviews out of selected region
            reviews_in_list.append(reviews)
            for word in reviews[1]:
                if word in words_dict:
                    selected_anomalous_words_dict[int(words_dict[word])] = 1
        else:
            reviews[0] = random.randint(0, n - 1)
            reviews_out_list.append(reviews)
    print('number of reviews in selected: ', len(reviews_in_list))
    print('number of reviews out selected: ', len(reviews_out_list))
    print('number of words in selected type: ',
          len(selected_anomalous_words_dict))
    return reviews_in_list, reviews_out_list, selected_anomalous_words_dict


def get_mean_std_from_shuffled_data(reviews_out_list, nodes_dict, words_dict):
    n = len(nodes_dict)
    p = len(words_dict)
    graph_matrix = []
    for i in range(n):
        graph_matrix.append([0.0] * p)
    for reviews in reviews_out_list:
        i = reviews[0]
        for word in reviews[1]:
            if word in words_dict:
                j = words_dict[word]
                graph_matrix[i][j] += 1.0
    mean_list = []
    std_list = []
    for i in range(p):
        col_i = [graph_matrix[j][i] for j in range(n)]
        mean_ = np.mean(col_i)
        std_ = np.std(col_i)
        mean_list.append(mean_)
        std_list.append(std_)
        if mean_ == 0.0 or std_ == 0.0:
            print('warning, (mean,std): ({},{})'.format(mean_, std_))
            sys.exit(0)
    print('mean std generated ...')
    return mean_list, std_list, graph_matrix


def normalized_graph_matrix(mean_list, std_list, reviews_list, nodes_dict,
                            words_dict):
    normalized_matrix = []
    n = len(nodes_dict)
    p = len(words_dict)
    for i in range(n):
        normalized_matrix.append([0.0] * p)
    for review in reviews_list:
        i = review[0]
        for word in review[1]:
            if word in words_dict:
                j = words_dict[word]
                normalized_matrix[i][j] += 1.0
    for i in range(p):
        if std_list[i] == 0.0:
            print('warning! std is 0.0')
        else:
            for j in range(n):
                normalized_matrix[j][i] = (normalized_matrix[j][i] - mean_list[
                    i]) / std_list[i]
    return normalized_matrix


def save_to_file(normalized_matrix, nodes_dict, words_dict,
                 selected_nodes_dict,
                 selected_anomalous_words_dict, edges, file_name):
    root_ = '/home/baojian/Dropbox/expriments/Yelp/graph/'
    fp = open(root_ + file_name, 'w')
    n = len(normalized_matrix)
    p = len(normalized_matrix[1])
    k = len(selected_nodes_dict)
    s = len(selected_anomalous_words_dict)
    m = len(edges)
    fp.write('{} {} {} {} {}\n'.format(n, p, k, s, m))
    for item_ in normalized_matrix:
        fp.write(' '.join([str(entry_) for entry_ in item_]) + '\n')
    for item_ in nodes_dict:
        fp.write(
            '{} {}\n'.format(item_.encode('utf-8').strip(), nodes_dict[item_]))
    for item_ in words_dict:
        fp.write(
            '{} {}\n'.format(item_.encode('utf-8').strip(), words_dict[item_]))
    for item_ in selected_nodes_dict:
        fp.write('{} '.format(item_))
    fp.write('\n')
    for item_ in selected_anomalous_words_dict:
        fp.write('{} '.format(item_))
    fp.write('\n')
    for item_ in edges:
        fp.write('{} {}\n'.format(item_[0], item_[1]))
    fp.close()
    print('save to file. ')


def load_from_file():
    normalized_matrix = []
    words_id_list = []
    nodes_dict = dict()
    selected_nodes_dict = dict()
    selected_anomalous_words_dict = dict()
    index = 0
    with open('test_case.txt') as f:
        for each_line in f.readlines():
            items_ = each_line.rstrip().split(' ')
            if index == 0:
                n1 = int(items_[0])
                n2 = int(items_[1])
                print('{} {}'.format(n1, n2))
            elif 1 <= index <= n1:
                normalized_matrix.append([float(item_) for item_ in items_])
            elif n1 + 1 <= index <= 2 * n1:
                nodes_dict[items_[0]] = int(items_[1])
            elif index == 2 * n1 + 1:
                print('size of normalized_matrix: {} {}'.format(
                    len(normalized_matrix),
                    len(normalized_matrix[0])))
                print('size of nodes_dict: {}'.format(len(nodes_dict)))
                for item_ in items_:
                    selected_nodes_dict[int(item_)] = 1
            elif index == 2 * n1 + 2:
                for item_ in items_:
                    selected_anomalous_words_dict[item_] = 1
            elif index == 2 * n1 + 3:
                for item_ in items_:
                    words_id_list.append(item_)
            else:
                print(' ')
            index += 1
    print('load finish ...')
    return normalized_matrix, nodes_dict, words_id_list, selected_nodes_dict, \
           selected_anomalous_words_dict


def get_largest_connected_component(anomalous_nodes, g):
    sub_graph = nx.Graph()
    for node in anomalous_nodes:
        for nei in g.neighbors(node):
            if nei in anomalous_nodes:
                sub_graph.add_edge(node, nei)
    largest_cc = []
    for cc in nx.connected_components(sub_graph):
        if len(cc) > len(largest_cc):
            largest_cc = list(cc)
    cc_dict = dict()
    for item_ in largest_cc:
        cc_dict[item_] = 1
    return cc_dict


def subroutine_random_walk(adj_list, all_abnormal_nodes, restart_prob,
                           sub_graph_size):
    sub_graphs = []
    for node in all_abnormal_nodes:
        sub_graph = set()
        sub_graph.add(node)
        initial_node = node
        current_node = initial_node
        count = 0
        while True:
            count += 1
            rand_index = random.randint(0, len(adj_list[current_node]) - 1)
            next_node = adj_list[current_node][rand_index]
            sub_graph.add(next_node)
            if np.random.uniform() < restart_prob:
                current_node = initial_node
            else:
                current_node = next_node
            if len(sub_graph) >= sub_graph_size:
                break
            if count >= 50000:
                break
        item_ = [len(set(all_abnormal_nodes).intersection(sub_graph)),
                 sub_graph]
        if item_[0] >= 40:
            print('--------------', item_[0], len(item_[1]), count)
        sub_graphs.append(item_)
    best_sub_graph = [-1, []]
    for sub_graph in sub_graphs:
        if sub_graph[0] > best_sub_graph[0]:
            best_sub_graph = sub_graph
    print('best subgraph: ', best_sub_graph[0], best_sub_graph[1])
    sub_graph = dict()
    for item_ in list(best_sub_graph[1]):
        sub_graph[item_] = 1
    return sub_graph


def random_walk(normalized_matrix, selected_nodes_dict, nodes_dict,
                words_id_list, selected_anomalous_words_dict,
                anomalous_threshold, g):
    n = len(nodes_dict)
    p = len(words_id_list)
    print('start random walk ............')
    print('n: {} p: {}'.format(n, p))
    print('number of related words: {}'.format(
        len(selected_anomalous_words_dict)))
    counts = 0
    total_anomalous_counts = 0

    largest_cc_s = dict()
    for j in range(p):
        anomalous_ = dict()
        for i in selected_nodes_dict:
            if j in selected_anomalous_words_dict and normalized_matrix[i][
                j] >= anomalous_threshold:
                anomalous_[i] = 1
        if anomalous_ >= 50:
            largest_cc = get_largest_connected_component(anomalous_, g)
            if len(largest_cc) > 5:
                largest_cc_s[words_id_list[j]] = largest_cc
    for entry_ in sorted(
            [[len(largest_cc_s[item_]), item_] for item_ in largest_cc_s],
            reverse=True):
        print('size of cc: {}, word: {}'.format(entry_[0], entry_[1]))
    words = ['mexican', 'taco', 'salsa', 'asada', 'tacos', 'carne', 'burrito']
    combined_nodes = dict()
    for word in words:
        for item_ in largest_cc_s[word]:
            combined_nodes[item_] = 1
    largest_cc = get_largest_connected_component(combined_nodes, g)
    print('largest cc {} after combined.'.format(len(largest_cc)))
    print('number of good candidate:', counts)
    print('size of largest cc: ', len(largest_cc))
    print('total anomalous counts is : ', total_anomalous_counts)
    print('total number of entries >= {}: {}, which are anomalous. '.format(
        anomalous_threshold, counts))
    adj = []
    for i in range(g.number_of_nodes()):
        adj.append([])
    for edge in g.edges():
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])
    random_sub_graph = subroutine_random_walk(adj,
                                              [node for node in largest_cc],
                                              0.20, 100)
    print('intersection with largest cc:',
          len(set([item_ for item_ in random_sub_graph]).intersection(
              set([item_ for item_ in largest_cc]))))
    return random_sub_graph


def do_analysis_on_graph(g, selected_nodes_dict):
    sub_graph = nx.Graph()
    for i in selected_nodes_dict:
        for nei in g.neighbors(i):
            if nei in selected_nodes_dict:
                sub_graph.add_edge(i, nei, weight=1.0)
    for i in selected_nodes_dict:
        if i not in sub_graph.nodes():
            sub_graph.add_node(i)
    index = 0
    print('number of nodes in selected type: ', sub_graph.number_of_nodes())
    print('number of edges in selected type: ', sub_graph.number_of_edges())
    print('number of connected component: ',
          nx.number_connected_components(sub_graph))
    largest_cc = []
    cc_ = nx.connected_components(sub_graph)
    for cc in cc_:
        if not largest_cc:
            largest_cc = cc
        elif len(largest_cc) < len(cc):
            largest_cc = cc
    for cc in sorted([len(cc) for cc in cc_]):
        if cc > 15:
            print('connected component[{}] has size: {}'.format(index, cc))
        index += 1
    largest_connected_component = dict()
    for item_ in largest_cc:
        largest_connected_component[item_] = 1
    return largest_connected_component


def generate_each_file(each_file_path, year, i_index,
                       word_freq_threshold_lower=5000.0,
                       word_freq_threshold_upper=100000.0,
                       anomalous_threshold=1.5):
    # ----------- load graph ------------
    nodes_dict, nodes_id_list, g = load_graph('processed_graph_15.txt')
    # ----------- load reviews and types for each business id ----------------
    selected_type = 'Mexican'  # 1939 # Chinese 1714
    categories_dict, reviews_list, selected_nodes_dict, total_keywords_dict = \
        load_reviews_data(
            each_file_path, nodes_dict, selected_type,
            word_freq_threshold_lower,
            word_freq_threshold_upper)
    do_analysis_on_graph(g, selected_nodes_dict)
    # ----------- select a type ---------------------------------------------
    print('number of selected nodes in {} is {}. '.format(selected_type, len(
        selected_nodes_dict)))
    # ----------- select subset of keywords based on df-idf -----------------
    words_dict, words_id_list = get_selected_matrix_df_idf(
        nodes_id_list, total_keywords_dict, reviews_list)
    print('number of keywords in processed graph: ', len(words_dict))
    reviews_in_list, reviews_out_list, selected_anomalous_words_dict = \
        generate_shuffle_data(
            reviews_list, selected_nodes_dict, nodes_dict, words_dict)
    mean_list, std_list, graph_matrix = get_mean_std_from_shuffled_data(
        reviews_out_list, nodes_dict, words_dict)

    normalized_matrix = normalized_graph_matrix(
        mean_list, std_list, reviews_out_list + reviews_in_list, nodes_dict,
        words_dict)
    largest_cc = random_walk(
        normalized_matrix, selected_nodes_dict, nodes_dict, words_id_list,
        selected_anomalous_words_dict, anomalous_threshold, g)
    copy_data = reviews_out_list
    list_largest_cc = [item_ for item_ in largest_cc]
    for item_ in reviews_in_list:
        if item_[0] in largest_cc:
            rand_ = list_largest_cc[random.randint(0, len(largest_cc) - 1)]
            item_[0] = rand_
            copy_data.append(item_)
    print('number of removed: ', (len(reviews_list) - len(copy_data)))
    normalized_matrix = normalized_graph_matrix(mean_list, std_list, copy_data,
                                                nodes_dict, words_dict)
    n = len(nodes_dict)
    p = len(words_dict)
    mean_val_list = []
    total_val_mat = 0.0
    for j in range(p):
        sum_ = np.sum(
            [normalized_matrix[i][j] for i in range(n) if i in largest_cc])
        total_val_mat += sum_
    for j in range(p):
        sum_ = np.sum(
            [normalized_matrix[i][j] for i in range(n) if i in largest_cc])
        mean_in = sum_ / (len(largest_cc) * 1.0)
        mean_out = (total_val_mat - sum_) / (n * p * 1.0)
        mean_val_list.append([mean_in, mean_out, j])
    for item_ in sorted(mean_val_list, reverse=True):
        if words_id_list[item_[2]] in selected_anomalous_words_dict:
            print('abnormal: ', item_[:3])
        else:
            print('normal: ', item_[:3])
    time.sleep(10000)
    save_to_file(normalized_matrix, nodes_dict, words_dict, largest_cc,
                 selected_anomalous_words_dict, g.edges(),
                 'processed_{}_test_case_{}.txt'.format(year, i_index))


root_path = '/home/baojian/Dropbox/expriments/Yelp/' \
            'yelp_dataset_challenge_academic_dataset/'
output_root_path = '/home/baojian/Dropbox/expriments/Yelp/'
stopwords_ = [item.lower() for item in stopwords.words('english')]
punctuations_ = list(string.punctuation)
punctuations_.append("'s")

if __name__ == "__main__":
    for i_index in range(25, 52):
        file_paths = [output_root_path + 'processed_records_2014.json',
                      output_root_path + 'processed_records_2015.json']
        generate_each_file(file_paths, '2014_2015', i_index)
