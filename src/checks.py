import pandas as pd
import sys


def check_edges_nodes(edges_path, nodes_path):
    nodes = set(pd.read_csv(nodes_path)['node_id'].astype(str).str.strip('"'))
    edges = pd.read_csv(edges_path)
    src = edges['node_1'].astype(str).str.strip('"')
    dst = edges['node_2'].astype(str).str.strip('"')
    missing = set(src.unique()).union(dst.unique()) - nodes
    if not missing:
        print('OK: all referenced nodes are present')
        return True
    print('MISSING:', len(missing), 'unique node ids')
    print(list(nodes)[0], list(src)[0], list(dst)[0])
    return False


if __name__ == '__main__':
    edges_path = sys.argv[1] if len(sys.argv) > 1 else 'data/edges.csv'
    nodes_path = sys.argv[2] if len(sys.argv) > 2 else 'data/nodes.csv'
    ok = check_edges_nodes(edges_path, nodes_path)
    if not ok:
        sys.exit(2)
