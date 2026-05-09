import math
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


# =========================
# TÍNH ENTROPY / GINI
# =========================

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)

    result = 0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)

    return result


def gini(labels):
    total = len(labels)
    counts = Counter(labels)

    result = 1
    for count in counts.values():
        p = count / total
        result -= p ** 2

    return result


def majority_class(labels):
    return Counter(labels).most_common(1)[0][0]


# =========================
# CHỌN THUỘC TÍNH TỐT NHẤT
# =========================

def best_attribute(data, attributes, target_col, criterion):
    base_value = entropy(data[target_col]) if criterion == "entropy" else gini(data[target_col])

    best_gain = -1
    best_attr = None

    for attr in attributes:
        weighted_value = 0

        for value in data[attr].unique():
            subset = data[data[attr] == value]
            weight = len(subset) / len(data)

            if criterion == "entropy":
                weighted_value += weight * entropy(subset[target_col])
            else:
                weighted_value += weight * gini(subset[target_col])

        gain = base_value - weighted_value

        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    return best_attr


# =========================
# XÂY CÂY
# =========================

def build_tree(data, attributes, target_col, criterion="entropy"):
    labels = data[target_col]

    if len(labels.unique()) == 1:
        return labels.iloc[0]

    if len(attributes) == 0:
        return majority_class(labels)

    best_attr = best_attribute(
        data,
        attributes,
        target_col,
        criterion
    )

    tree = {
        best_attr: {}
    }

    remaining_attrs = [
        attr for attr in attributes if attr != best_attr
    ]

    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]

        if subset.empty:
            tree[best_attr][value] = majority_class(labels)
        else:
            tree[best_attr][value] = build_tree(
                subset,
                remaining_attrs,
                target_col,
                criterion
            )

    return tree



# =========================
# VẼ CÂY ĐẸP
# =========================

def add_tree_to_graph(graph, tree, parent_id=None, edge_label="", counter=[0]):
    counter[0] += 1
    node_id = f"node_{counter[0]}"

    if isinstance(tree, dict):
        root_label = list(tree.keys())[0]

        graph.add_node(
            node_id,
            label=root_label,
            is_leaf=False
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                node_id,
                label=edge_label
            )

        for branch_value, subtree in tree[root_label].items():
            add_tree_to_graph(
                graph,
                subtree,
                node_id,
                str(branch_value),
                counter
            )

    else:
        graph.add_node(
            node_id,
            label=str(tree),
            is_leaf=True
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                node_id,
                label=edge_label
            )

    return node_id


def hierarchy_pos(graph, root, width=2.0, vert_gap=0.3, vert_loc=0, xcenter=0):
    children = list(graph.successors(root))

    if len(children) == 0:
        return {
            root: (xcenter, vert_loc)
        }

    pos = {
        root: (xcenter, vert_loc)
    }

    dx = width / len(children)
    next_x = xcenter - width / 2 - dx / 2

    for child in children:
        next_x += dx

        pos.update(
            hierarchy_pos(
                graph,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=next_x
            )
        )

    return pos


def draw_categorical_tree(tree, title="Cây quyết định"):
    graph = nx.DiGraph()

    counter = [0]
    root_id = add_tree_to_graph(
        graph,
        tree,
        counter=counter
    )

    pos = hierarchy_pos(
        graph,
        root_id,
        width=3.0,
        vert_gap=0.35,
        vert_loc=0,
        xcenter=0
    )

    fig, ax = plt.subplots(figsize=(16, 9))

    node_colors = []
    labels = {}

    for node in graph.nodes():
        is_leaf = graph.nodes[node].get("is_leaf", False)
        labels[node] = graph.nodes[node].get("label", node)

        if is_leaf:
            node_colors.append("#90EE90")
        else:
            node_colors.append("#ADD8E6")

    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_size=3500,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.5,
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowsize=15,
        ax=ax
    )

    edge_labels = nx.get_edge_attributes(graph, "label")

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=10,
        font_color="black",
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold"
    )

    ax.axis("off")

    return fig