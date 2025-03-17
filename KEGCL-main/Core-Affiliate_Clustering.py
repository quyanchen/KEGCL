from optparse import OptionParser
from numpy import *

from numpy.linalg import *
import string
import os


def f_key(a):
    return(a[-1])


def density_score(temp_set, matrix):
    temp_density_score = 0.
    for m in temp_set:
        for n in temp_set:
            if n != m and matrix[m, n] != 0:
                temp_density_score += matrix[m, n]

    temp_density_score = temp_density_score / (len(temp_set) * (len(temp_set) - 1))
    return temp_density_score


def merge_cliques(new_cliques_set, matrix):
    seed_clique = []

    while (True):
        temp_cliques_set = []
        if len(new_cliques_set) >= 2:
            seed_clique.append(new_cliques_set[0])

            for i in range(1, len(new_cliques_set)):
                if len(new_cliques_set[i].intersection(new_cliques_set[0])) == 0:
                    temp_cliques_set.append(new_cliques_set[i])
                elif len(new_cliques_set[i].difference(new_cliques_set[0])) >= 3:
                    temp_cliques_set.append(new_cliques_set[i].difference(new_cliques_set[0]))

            cliques_set = []

            for i in temp_cliques_set:

                clique_score = density_score(i, matrix)
                temp_list = []
                for j in i:
                    temp_list.append(j)

                temp_list.append(clique_score)
                cliques_set.append(temp_list)

            cliques_set.sort(key=f_key, reverse=True)

            new_cliques_set = []
            for i in range(len(cliques_set)):
                temp_set = set([])
                for j in range(len(cliques_set[i]) - 1):
                    temp_set.add(cliques_set[i][j])
                new_cliques_set.append(temp_set)

        elif len(new_cliques_set) == 1:
            seed_clique.append(new_cliques_set[0])
            break
        else:
            break

    return seed_clique


def expand_cluster(seed_clique, all_protein_set, matrix, expand_thres):
    expand_set = []
    complex_set = []

    for instance in seed_clique:
        avg_node_score = density_score(instance, matrix)

        temp_set = set([])
        for j in all_protein_set.difference(instance):
            temp_score = 0.
            for n in instance:
                temp_score += matrix[n, j]
            temp_score /= len(instance)

            if (temp_score ) >= expand_thres:
                temp_set.add(j)
        expand_set.append(temp_set)
    for i in range(len(seed_clique)):
        complex_set.append(seed_clique[i].union(expand_set[i]))

    return (complex_set)


protein_out_file="protein.temp"  #for each protein a index
cliques_file="cliques"
ppi_pair_file="ppi.pair"
ppi_matrix_file="ppi.matrix"

if __name__ == "__main__":
    f = open("qycdataset/krogan2006corenew_attr_sim.txt", "r")
    f_protein_out = open(protein_out_file, "w")
    Dic_map = {}
    index = 0
    Node1 = []
    Node2 = []
    Weight = []
    All_node = set([])
    All_node_index = set([])

    for line in f:
        line = line.strip().split()
        if len(line) == 3:
            Node1.append(line[0])
            All_node.add(line[0])
            Node2.append(line[1])
            All_node.add(line[1])
            Weight.append(float(line[2]))
            if line[0] not in Dic_map:
                Dic_map[line[0]] = index
                f_protein_out.write(line[0] + "\n")
                All_node_index.add(index)
                index += 1
            if line[1] not in Dic_map:
                Dic_map[line[1]] = index
                f_protein_out.write(line[1] + "\n")
                All_node_index.add(index)
                index += 1
    Node_count = index
    f.close()
    f_protein_out.close()
    #print Dic_map


    f.close()

    ######dic_map to map_dic###########
    Map_dic = {}
    for key in Dic_map.keys():
        Map_dic[Dic_map[key]] = key

    #print Map_dic

    ######bulid Adj_matrix###########

    Adj_Matrix = mat(zeros((Node_count, Node_count), dtype=float))

    if len(Node1) == len(Node2):

        for i in range(len(Node1)):
            if Node1[i] in Dic_map and Node2[i] in Dic_map:
                Adj_Matrix[Dic_map[Node1[i]], Dic_map[Node2[i]]] = Weight[i]
                Adj_Matrix[Dic_map[Node2[i]], Dic_map[Node1[i]]] = Weight[i]
    #print Adj_Matrix.shape[0]

    os.system(
        "ConvertPPI.exe " + "dataset/krogan2006core.txt" + " " + protein_out_file + " " + ppi_pair_file + " " + ppi_matrix_file)
    os.system(
        "Mining_Cliques.exe " + ppi_matrix_file + " " + "1" + " " + "3" + " " + str(Node_count) + " " + cliques_file)

    cliques_set = []
    f = open(cliques_file, "r")
    for line in f:
        temp_set = []
        line = line.strip().split()
        for i in range(1, len(line)):
            temp_set.append(int(line[i]))
        cliques_set.append(temp_set)

    f.close()
    avg_clique_score = 0.

    for instance in cliques_set:
        clique_score = density_score(instance, Adj_Matrix)
        avg_clique_score += clique_score
        instance.append(clique_score)
    avg_clique_score /= len(cliques_set)
    cliques_set.sort(key=f_key, reverse=True)
    #print cliques_set

    new_cliques_set = []
    for i in range(len(cliques_set)):
        temp_set = set([])
        for j in range(len(cliques_set[i]) - 1):
            temp_set.add(cliques_set[i][j])
        new_cliques_set.append(temp_set)
    #print new_cliques_set
    #print len(new_cliques_set)

    seed_clique = merge_cliques(new_cliques_set, Adj_Matrix)
    #print seed_clique
    #print len(seed_clique)

    expand_thres = 0.3
    complex_set = expand_cluster(seed_clique, All_node_index, Adj_Matrix, expand_thres)
    print("##########output predicted complexes##########\n")
    final_file = open("qycdataset/final_krogan2006corenew_attr_output", "w")

    for i in range(len(complex_set)):

        line = ""
        for m in complex_set[i]:
            line += Map_dic[m] + " "
        line += "\n"

        final_file.write(line)
    final_file.close()

    # print len(complex_set)

    print("##########COAN completes############")