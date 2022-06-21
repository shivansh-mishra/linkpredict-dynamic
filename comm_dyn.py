import os
import numpy as np
import datetime
import math
import networkx as nx


import time
import sys
import json


var_dict_main = {}


def local_path(adj, parameter=0.05):
    G = nx.Graph(adj)
    adj2 = nx.to_numpy_matrix(G)
    common = np.zeros(shape=(len(adj), len(adj)))
    common1 = np.zeros(shape=(len(adj), len(adj)))
    common2 = np.zeros(shape=(len(adj), len(adj)))
    adj2 = np.zeros(shape=(len(adj), len(adj)))
    for (u, v) in G.edges:
        adj2[u][v] = 1
        adj2[v][u] = 1
    print(len(G.edges))
    print("before lp matmul1")
    common1 = np.matmul(adj2, adj2)
    print("before lp matmul2")
    common2 = np.matmul(common1, adj2)
    common2 = common2 * parameter
    for i in range(len(adj)):
        for j in range(len(adj)):
            common[i][j] = common1[i][j] + common2[i][j]
    return common


def gen_rand_edges_pool_single(num):
    starttime = time.time()
    G = var_dict_main['graph']
    print("inside pool num = "+str(num))
    seen = set()
    x, y = random.choice(list(G.nodes)), random.choice(list(G.nodes))
    t = 0
    while t < num:
        if not G.has_edge(x, y) and (y, x) not in seen:
            seen.add((x, y))
            t = t + 1
            #print(t)
        x, y = random.choice(list(G.nodes)), random.choice(list(G.nodes))

    print("after random edge generation inside pool "+str(len(seen)))
    endtime = time.time()
    currentDT = datetime.datetime.now()
    print(str(currentDT))
    #print(len(seen_inside))
    #print(seen_inside)

    return seen


def gen_rand_edges(num, G):
    #print(str(G.nodes))
    #print(str(G.edges))
    num = num*5
    print("generating random edges")
    print(type(G))
    print("nodes = "+str(len(G.nodes)))
    print("edges = "+str(len(G.edges)))
    cpu = mp.cpu_count()
    cpu = cpu - 1
    print("cpu cores = " + str(cpu))
    arglist = []
    for i in range(cpu):
        arglist.append(int(num/cpu))
    var_dict_main['graph'] = G
    seen = set()
    with mp.Pool(cpu) as p:
        element = p.map(gen_rand_edges_pool_single,arglist)
        for e in element:
            seen.update(e)
    x, y = random.choice(list(G.nodes)), random.choice(list(G.nodes))
    t = len(seen)
    print("outside pool t = "+str(t)+" num = "+str(num))
    #print(seen)
    #time.sleep(100)
    while t < num:
        if not G.has_edge(x,y) and (y, x) not in seen:
            seen.add((x, y))
            t = t + 1
        x, y = random.choice(list(G.nodes)), random.choice(list(G.nodes))
    seen = list(seen)
    print("after random edge generation")
    return seen


def data(m, t):
    print("reading dataset from file")
    data = open('./datasets_dynamic/'+str(t) + ".txt")
    edgelist = map(lambda q: list(map(int, q.split())), data.read().split("\n")[:-1])
    data.close()
    maxi = 0
    mini = 100000000000000000
    edgelist = list(edgelist)
    for x in edgelist:
        if x[-1] > maxi:
            maxi = x[-1]
        if x[-1] < mini:
            mini = x[-1]
    min1 = mini
    w = int((maxi - mini) / m)
    edgelist.sort(key=lambda x: x[-1])
    arr = []
    i = 0
    for i in range(0, m + 1):
        arr = arr + [min1 + w * i]
    arri = []
    # print(arr)
    nodes = set()
    for i in range(0, m):
        temp = []
        for j in edgelist:
            if j[-1] >= arr[i] and j[-1] <= arr[i + 1] and j[0]!=j[1]:
                temp += [[j[0], j[1]]]
        if temp != []:
            arri += [temp]
    # print(arri)
    # for x in arri:
    #     print(len(x))
    print("after read")
    #random.shuffle(arri)
    #print(str(arri))
    for item in arri: print(len(item))
    return arri


def gen_graph(l):
    print("inside gen graph")
    t_graph = []
    node_set = set()
    max = -99999
    min = 99999
    for i in l:
        for edge in i:
            node_set.add(edge[0])
            node_set.add(edge[1])
            u = edge[0]
            v = edge[1]
            if u<v:
                if min > u:
                   min = u
                if max < v:
                    max = v
            else:
                if min > v:
                   min = v
                if max < u:
                    max = u
    print(str(min) + "-" + str(max))
    #sys.exit()
    edgelist_new = []
    count = -1
    for i in l:
        graph = nx.Graph()
        #graph.add_nodes_from(node_set)
        graph.add_edges_from(i)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        t_graph.append(graph)
        edgelist_new.append(list(graph.edges))
    return [t_graph,edgelist_new]


def partition_to_lp (x,y, graph, comm_list):
    count = 0
    comm_dict = dict()
    comm_no_indi = dict()
    #create dictionary which gives node communities
    for item in comm_list:
        count += 1
        comm_no_indi[count] = len(item)
        for node in item:
            comm_dict[node] = count
    #case to check faulty algorithm
    for node in graph:
        if node not in comm_dict.keys():
            print("non labelled node found")
            sys.exit()
    #getting 3 hop regions of x and y nodes
    reg_x = set()
    reg_y = set()
    reg_x.add(x)
    reg_y.add(y)
    #for 3 hop region
    for i in range(3):
        temp_x = set()
        temp_y = set()
        for curr in reg_x:
            temp_x.add(curr)
            for single in graph.neighbors(curr):
                temp_x.add(single)
        for curr in reg_y:
            temp_y.add(curr)
            for single in graph.neighbors(curr):
                temp_y.add(single)
        reg_x = temp_x
        reg_y = temp_y
    reg_x.remove(x)
    reg_y.remove(y)
    #intersection of ego regions
    common_reg = reg_x.intersection(reg_y)
    lp = 0
    count_comm = dict()
    #count number of instances of each community in region of common infleunce
    for key in comm_no_indi.keys():
        count_comm[key] = 0
    for item in common_reg:
        count_comm[comm_dict[item]] += 1
    for key in comm_no_indi.keys():
        if comm_dict[x] == comm_dict[y] == key:
            # special case for 3 equal labels to be defined
            #count_comm[key] = count_comm[key] / comm_no_indi[key]
            count_comm[key] = count_comm[key]
        else:
            count_comm[key] = count_comm[key]/comm_no_indi[key]
    for key in count_comm.keys():
        lp += count_comm[key]
    return lp


def features_comm_opti(m, t, f_no=12):
    identity = " algo - comm dyn dataset - " + str(t)

    data_curr = data(m,t)
    l1 = data_curr
    l = []
    for i in l1:
        edgelist = list(set(tuple(sorted(sub)) for sub in i))
        l.append(edgelist)

    t_graph = gen_graph(l)[0]
    l = gen_graph(l)[1]

    teCt = l[m - 1]
    reCt = gen_rand_edges(len(teCt), t_graph[m - 1])

    for i in reCt:
        teCt.append(i)

    teCt = list(set(tuple(sorted(sub)) for sub in teCt))

    list1 = []
    print("before graph" + str(identity))
    edges_dict = dict()
    for t1 in range(0, m - 1):
        starttime = time.time()
        print("before dictionary" + str(identity))
        sum = 0
        count = 0
        dict_count = 0
        dict_node = dict()
        G = t_graph[t1]
        print(str(G))
        for node in G.nodes:
            dict_node[node] = dict_count
            dict_count += 1
        currentDT = datetime.datetime.now()
        print(str(currentDT))
        from cdlib.algorithms import cpm, label_propagation, chinesewhispers,\
            der, eigenvector, sbm_dl_nested, walktrap, surprise_communities, \
            spinglass, significance_communities, rb_pots, rber_pots, pycombo, \
            paris, mcode, markov_clustering, lswl_plus, infomap, greedy_modularity, \
            leiden
        
        current_comm = greedy_modularity(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_greed = current_comm_dict['communities']
        print("no. of communities greed - " + str(len(current_comm_list_greed)))
        for item in current_comm_list_greed:
            print(item)
        print("after community greed")
        
        current_comm = eigenvector(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_eigen = current_comm_dict['communities']
        print("no. of communities eigen - " + str(len(current_comm_list_eigen)))
        for item in current_comm_list_eigen:
            print(item)
        print("after community eigen")
        
        current_comm = cpm(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_cpm = current_comm_dict['communities']
        print("no. of communities cpm - " + str(len(current_comm_list_cpm)))
        for item in current_comm_list_cpm:
            print(item)
        print("after community cpm")
        
        current_comm = significance_communities(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_signi = current_comm_dict['communities']
        print("no. of communities signi - " + str(len(current_comm_list_signi)))
        for item in current_comm_list_signi:
            print(item)
        print("after community signi")
        
        current_comm = der(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_der = current_comm_dict['communities']
        print("no. of communities der - " + str(len(current_comm_list_der)))
        for item in current_comm_list_der:
            print(item)
        print("after community der")
        
        current_comm = surprise_communities(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_surp = current_comm_dict['communities']
        print("no. of communities surp - " + str(len(current_comm_list_surp)))
        for item in current_comm_list_surp:
            print(item)
        print("after community surp")
        
        current_comm = sbm_dl_nested(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_sbm = current_comm_dict['communities']
        print("no. of communities sbm - " + str(len(current_comm_list_sbm)))
        for item in current_comm_list_sbm:
            print(item)
        print("after community sbm")
        
        current_comm = leiden(G)
        print(type(current_comm))
        current_comm_dict = json.loads(current_comm.to_json())
        current_comm_list_leiden = current_comm_dict['communities']
        print("no. of communities leiden - " + str(len(current_comm_list_leiden)))
        for item in current_comm_list_leiden:
            print(item)
        print("after community leiden")
        #sys.exit()
        print(len(G.edges))
        adj = nx.to_numpy_matrix(G)
        edge_length = len(G.edges)
        triangles_dict = dict()
        
        print("before laplacian" + str(identity))
        from scipy.sparse.csgraph import laplacian
        L = laplacian(adj)
        #print(L)
        
        try:
            currentDT = datetime.datetime.now()
            print(str(currentDT))
            print("before p inverse" + str(identity))
            L_pinverse = np.linalg.pinv(L)
            np.save(path_pinv, L_pinverse)
            #print(L_pinverse)
        except:
        	currentDT = datetime.datetime.now()
            print(str(currentDT))
            print("before empty error p inverse" + str(identity))
            L_pinverse = np.zeros(shape=(len(adj), len(adj)))
        
        print("before inverse" + str(identity))
        I = np.identity(len(adj))
        #print(I)
        middle = I + L
        inverse = np.linalg.inv(middle)
        
        print("before quasi")
        print("before local path" + str(identity))
        common_lp = local_path(adj)

        print("before edges after graph" + str(identity))
        for e in teCt:
            i = e[0]
            j = e[1]
            edge_key = str(e[0]) + "+" + str(e[1])
            count += 1
            if count % 1000 == 0:
                print(str(count) + " - " + str(len(teCt)) + " t1 = " + str(t1) + str(identity))
            # print(e)
            list2 = []
            curr_tuple = np.zeros(7)
            if G.has_node(i) and G.has_node(j):
                i_orig = i
                j_orig = j
                i = dict_node[i]
                j = dict_node[j]
                ##aa
                aa = 0
                common_neighbours_all = nx.common_neighbors(G, i_orig, j_orig)
                for common_neighbour in common_neighbours_all:
                    if G.degree[common_neighbour] != 0 and G.degree[common_neighbour] != 1:
                        aa = aa + 1 / math.log(G.degree[common_neighbour])
                sum += aa
                list2.append(aa)
                ##cn
                cn = len(sorted(nx.common_neighbors(G, i_orig, j_orig)))
                sum += cn
                list2.append(cn)
                ##pa
                pa = 0
                n1 = G.neighbors(i_orig)
                n2 = G.neighbors(j_orig)
                n1 = [item for item in n1]
                n2 = [item for item in n2]
                length = len(set().union(n1, n2))
                if length > 0:
                    pa = len(n1) * len(n2)
                sum += pa
                list2.append(pa)
                # cosp
                if L_pinverse[j][j] * L_pinverse[i][i] >= 0:
                    cosp = L_pinverse[i][j] / (math.sqrt(L_pinverse[j][j] * L_pinverse[i][i]))
                else:
                    cosp = -1
                sum += cosp
                if cosp >= 0:
                    list2.append(cosp)
                else:
                    list2.append(0)
                # mfi
                mfi = inverse[i][j]
                sum += mfi
                if mfi >= 0:
                    list2.append(mfi)
                else:
                    list2.append(0)
                # shortest path
                sp = 0
                try:
                    sp = nx.shortest_path_length(G, i_orig, j_orig)
                except:
                    sp = 0
                    file_write_name = './result_commlp/' + str(t) + '_sp/error.txt'
                    os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
                sum += sp
                list2.append(sp)
                # l3
                l3 = 0
                for intermediate_node1 in G.neighbors(i_orig):
                    for intermediate_node2 in G.neighbors(intermediate_node1):
                        if G.has_edge(j_orig, intermediate_node2):
                            l3 += 1 / math.sqrt(
                                G.degree[intermediate_node1] * G.degree[intermediate_node2])
                sum += l3
                list2.append(l3)
                # local path
                lp = common_lp[i][j]
                sum += lp
                list2.append(lp)
                #print("before der for edge")
                der = partition_to_lp(i_orig, j_orig, G, current_comm_list_der)
                if der >= 0:
                    sum += der
                    list2.append(der)
                else:
                    print("problem non negative der")
                    sys.exit()
                #print("before surp for edge")
                surp = partition_to_lp(i_orig, j_orig, G, current_comm_list_surp)
                if surp >= 0:
                    sum += surp
                    list2.append(surp)
                else:
                    print("problem non negative surp")
                    sys.exit()
                #print("before sbm for edge")
                sbm = partition_to_lp(i_orig, j_orig, G, current_comm_list_sbm)
                if sbm >= 0:
                    sum += sbm
                    list2.append(sbm)
                else:
                    print("problem non negative sbm")
                    sys.exit()
                eigen = partition_to_lp(i_orig, j_orig, G, current_comm_list_eigen)
                if eigen >= 0:
                    sum += eigen
                    list2.append(eigen)
                else:
                    print("problem non negative eigen")
                    sys.exit()
                # print("success"+str(identity))
            else:
                # print("out of bound")
                for i in range(f_no):
                    list2.append(0)

            if t1 == m - 2:
                if t_graph[m - 1].has_edge(e[0], e[1]):
                    list2.append(1)
                else:
                    list2.append(0)

            if edge_key in edges_dict:
                # print("old value = "+str(edges_dict[edge_key]))
                temp_list = list(edges_dict[edge_key])
                temp_list = temp_list + list2
                edges_dict[edge_key] = temp_list
                # print("new value = " + str(edges_dict[edge_key]))
            else:
                edges_dict[edge_key] = list2

        endtime = time.time()
        currentDT = datetime.datetime.now()
        print(str(currentDT))
        file_all = open('./result_comm/current_all.txt', 'a')
        text_final = str(identity) + " slice = " + str(t1) + " nodes = " + \
                     str(len(G.nodes)) + " edges = " + str(len(G.edges)) + \
                     " time - " + str(currentDT) + "\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()

    for e in teCt:
        edge_key = str(e[0]) + "+" + str(e[1])
        list1.append(list(edges_dict[edge_key]))
    f = np.array(list1)
    np.take(f, np.random.permutation(f.shape[0]), axis=0, out=f)
    # print("list1 = " + str(list1))
    print("after get feature" + str(identity))
    print("shape of feature = " + str(f.shape))
    return f

