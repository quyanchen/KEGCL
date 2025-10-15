import os
from collections import defaultdict
import numpy as np
import re

datasets = 'k06'
datasets2 = 'krogan2006core'

ppi_file = f"dataset/{datasets2}.txt"
go_attr_file = "protein_go_with_activity.txt"
output_prefix = f"{datasets}/{datasets2}_"


def extract_go_terms_from_file(file_path):
    go_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            go_set.update(re.findall(r"\bGO:\d{7}\b", line))
    return go_set

protein_cc = defaultdict(set)
protein_tt = defaultdict(set)
protein_go = defaultdict(set)

with open(go_attr_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        protein = parts[0].strip()


        terms_field = parts[1]
        go_terms = terms_field.strip().split() 

        tt_field = parts[-1]

        for term in go_terms:
            term = term.strip()
            if term.startswith("CC:"):
                protein_cc[protein].add(term)
            elif term.startswith("GO:"):
                protein_go[protein].add(term)

        if tt_field.startswith("TT:"):
            try:
                protein_tt[protein] = set(eval(tt_field[3:].strip()))
            except Exception as e:
                print(f"[!] TT ERROR: {protein} -> {tt_field}")


all_go_terms = extract_go_terms_from_file(go_attr_file)
go_list = sorted(all_go_terms)
go2idx = {go: i for i, go in enumerate(go_list)}


edge_subnets = defaultdict(list)
node_subnets = defaultdict(set)

with open(ppi_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if a not in protein_cc or b not in protein_cc:
            continue
        if a not in protein_tt or b not in protein_tt:
            continue
        common_cc = protein_cc[a] & protein_cc[b]
        common_tt = protein_tt[a] & protein_tt[b]
        for cc in common_cc:
            for t in common_tt:
                key = f"{cc}_t{t}"
                edge_subnets[key].append((a, b))
                node_subnets[key].update([a, b])


for idx, (key, edges) in enumerate(edge_subnets.items()):
    subname = f"{output_prefix}{idx}"
    node_file = f"{datasets}/{datasets}{idx}_node.txt"
    network_file = f"{datasets}/Network_{datasets}{idx}.txt"
    attr_file = f"{datasets}/Attribute_{datasets}{idx}.txt"

    nodes = sorted(list(node_subnets[key]))
    node2idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    with open(f"{subname}.txt", 'w') as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")

    with open(node_file, 'w') as f:
        for node in nodes:
            f.write(f"{node}\n")


    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        ui, vi = node2idx[u], node2idx[v]
        adj[ui][vi] = adj[vi][ui] = 1
    np.savetxt(network_file, adj, fmt='%d')


    attr = np.zeros((n, len(go2idx)), dtype=int)
    for j, node in enumerate(nodes):
        node = node.strip()
        terms = protein_go.get(node, set())
        for go_term in terms:
            go_term = go_term.strip()
            if go_term in go2idx:
                attr[j][go2idx[go_term]] = 1


    with open(attr_file, 'w') as f:
        for row in attr:
            f.write(' '.join(map(str, row)) + '\n')

