#!/usr/bin/env python3
from random import randint
import numpy as np
import collections
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt


def compute_I(d):
    d_i = d[0]
    i_value = 0
    for d_j in d:
        i_value += (d_i - d_j)
    return i_value

def my_compute(d, start, end):
    d_i = d[start]
    cost = 0
    for i in range(start,end+1):
        cost+= d_i - d[i]
    return cost

def compute_i(d, start, end):
    d_i = d[0]
    res = 0
    for i in range(start, end + 1):
        res += d_i - d[i]

    return res

def c_merge(d, d1, k):
    c_merge_cost = d1 - d[k] + compute_I(d[k+1:min(len(d), 2*k)])
    return c_merge_cost


def c_new(d, k):
    t = d[k:min(len(d), 2*k-1)]
    c_new_cost = compute_I(t)
    return c_new_cost

def sum_is_even(array):
    res = 0
    for val in array:
        res+=val
    if(res%2==0):
        return True
    return False


def greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension):
    if pos_init + extension >= len(array_degrees) - 1:
        for i in range(pos_init, len(array_degrees)):
            array_degrees[i] = array_degrees[pos_init]
        return array_degrees

    else:
        d1 = array_degrees[pos_init]
        c_merge_cost = c_merge(array_degrees, d1, pos_init + extension)
        c_new_cost = c_new(d, pos_init + extension)
        if c_merge_cost > c_new_cost:
            for i in range(pos_init, pos_init + extension):
                array_degrees[i] = d1
            greedy_rec_algorithm(array_degrees, k_degree, pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension + 1)


def dp_graph_anonymization(array_degrees, k):
    cur = 0
    i = 0
    aux = []
    res = []
    while i < len(array_degrees):
        if(cur < (2*k)-1):
            aux.append(array_degrees[i])
            cur+=1
            i+=1
        else:
            aux.append(array_degrees[i])
            total_cost = my_compute(aux,0,cur)
            save = total_cost
            best_t = 0
            minimum = max(k, cur-2*k+2)
            for t in range(minimum,(cur+2-k)):
                cost = my_compute(aux,0,t-1)+my_compute(aux,t,cur)
                if(cost < save):
                    save = cost
                    best_t = t
            if(save == total_cost):
                cur+=1
                i+=1
            else:
                for test in range (0,best_t):
                    res.append(aux[0])    
                aux = []
                cur = 0
                i=len(res)
    if(len(aux)!= 0 and len(aux)<k):
        for o in range(0,len(aux)):
            res.append(res[len(res)-1])
    elif(len(aux)!= 0 and len(aux)>=k):
        for o in range(0,len(aux)):
            res.append(aux[0])  
    return res

def construct_graph(tab_index, anonymized_degree):
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return None

    while True:
        if not all(di >= 0 for di in anonymized_degree):
            return None
        if all(di == 0 for di in anonymized_degree):
            return graph
        v = np.random.choice((np.where(np.array(anonymized_degree) > 0))[0])
        dv = anonymized_degree[v]
        anonymized_degree[v] = 0
        #print("v:{}\ndv:{}\nnp.argsort: \n{}\nnp.sort:\n{}".format(v,dv,np.argsort(anonymized_degree)[-dv:][::-1],np.sort(anonymized_degree)[-dv:][::-1]))
        for index in np.argsort(anonymized_degree)[-dv:][::-1]:
            if index == v:
                return None
            if not graph.has_edge(tab_index[v], tab_index[index]):
                graph.add_edge(tab_index[v], tab_index[index])
                anonymized_degree[index] = anonymized_degree[index] - 1

def super_graph(G,degrees_init, array_index, degrees_dp, l):
    graph = G
    map_with_key_index = dict(zip(np.sort(array_index),G.nodes))
    map_with_key_nodes = dict(zip((G.nodes),np.sort(array_index)))
    a = degrees_dp - degrees_init  
    index_order_by_a = [x for y, x in sorted(zip(a, array_index))]

    if(not sum_is_even(a)):
        return None

    Vl = (index_order_by_a[-l:])[::-1]
    a_Vl = (a[-l:])[::-1]
    V_minus_Vl = (index_order_by_a[:-l])[::-1]
    a_V_minus_Vl = (a[:-l])[::-1]
    sum_a_Vl = sum(a_Vl)
    second_sum_lemma2 = 0
    third_sum_lemma2 = 0

    for i in range(0,len(Vl)):
        count=0
        for x in range(0,len(Vl)):
            if(G.has_edge(map_with_key_index[i],map_with_key_index[x])):
                count += 1
        second_sum_lemma2 += (len(Vl)- 1 + count)

    for y in range(0,len(V_minus_Vl)):
        count=0
        for z in range(0,len(V_minus_Vl)):
            if(G.has_edge(map_with_key_index[y],map_with_key_index[z])):
                count+=1
        third_sum_lemma2+=min((len(V_minus_Vl)-count),a_V_minus_Vl[y])


    if(sum_a_Vl > second_sum_lemma2 + third_sum_lemma2):
        return None

    c = 0
    sum_a = sum(a)
    
    while True:

        if all(di == 0 for di in a):
            if c == sum_a / 2:
                nx.relabel_nodes(graph,map_with_key_nodes,copy=False)
                return graph
            else:
                return None
        
        v = np.random.choice((np.where(np.array(a) > 0))[0])
        dv = a[v]

        a[v] = 0
        for ind in np.array((np.where(np.array(a) > 0))[0]):
            if not graph.has_edge(map_with_key_index[v], map_with_key_index[ind]) and a[ind] > 0 and dv > 0 and ind != v:
                graph.add_edge(map_with_key_index[v], map_with_key_index[ind])
                a[ind] -= 1
                dv -= 1
                c += 1


if __name__ == "__main__":

    k_degree = int(sys.argv[1])
    file_graph = sys.argv[2]
    G = nx.Graph()
    
    if os.path.exists(file_graph): 
        # if file exist
        with open(file_graph) as f:
            content = f.readlines()
        # read each line
        content = [x.strip() for x in content]
        for line in content:
            # split name inside each line
            names = line.split(",")
            start_node = names[0]
            if start_node not in G:
                G.add_node(start_node)
            for index in range(1, len(names)):
                node_to_add = names[index]
                if node_to_add not in G:
                    G.add_node(node_to_add)
                G.add_edge(start_node, node_to_add)

    #Degree arrays preparation
    d = [x[1] for x in G.degree()]
    array_index = np.argsort(d)[::-1]
    array_degrees = np.sort(d)[::-1]
    print("Array of nodes (array_index): {}".format(array_index))
    print("Array of degrees sorted (array_degrees): {}".format(array_degrees))
    # TODO insert here the code
    array_degrees_dp = dp_graph_anonymization(array_degrees.copy(), k_degree)
    c_graph = construct_graph(array_index, array_degrees_dp.copy())
    #print("New Graph(construct_graph):\nnodes: {}\nedges: {}".format(c_graph.nodes, c_graph.edges))
    c_degrees = [x[1] for x in c_graph.degree()]
    c_arrayIndex = np.argsort(c_degrees)[::-1]
    c_arrayDegrees = np.sort(c_degrees)[::-1]

    Supergraph = super_graph(G.copy(),array_degrees,array_index,array_degrees_dp, 1000)

    #print("Array Degrees dp: {}\n".format(array_degrees_dp))
    #print("c_arrayIndex: \n{}\n c_arrayDegrees:\n{}".format(c_arrayIndex, c_arrayDegrees))

    AVG_Path_Length_Original = nx.average_shortest_path_length(G)
    print("\nAverage Path Shortest Length (Original Graph) = {}\n".format(AVG_Path_Length_Original)) 

    AVG_Path_Length_Supergraph = nx.average_shortest_path_length(Supergraph)
    print("Average Path Shortest Length (SuperGraph) = {}\n".format(AVG_Path_Length_Supergraph)) 

    clus = nx.clustering(Supergraph)
    array_clustering = []
    for i in clus:
        array_clustering.append(clus[i])
    print("Clustering Coefficent (SuperGraph) = {}\n".format(sum(array_clustering)/len(Supergraph.nodes)))

    clus_original = nx.clustering(G)
    array_clustering_original = []
    for i in clus_original:
        array_clustering_original.append(clus_original[i])
    cc = sum(array_clustering_original)/len(G.nodes)
    print("Clustering Coefficent (Original Graph) = {}\n".format(cc))


'''
#To save array_norm for each k in file csv

    array_norm = []
    for i in range(1,50):
        r = dp_graph_anonymization(array_degrees,i)
        norm_dp = vector_norm(r - array_degrees)
        array_norm.append(norm_dp)

    file_norm = np.savetxt('array_norm.csv',file_norm,fmt="%d",delimiter=",")
    np.savetxt("array_norm.csv",file_norm,fmt="%d", delimiter=",")
'''