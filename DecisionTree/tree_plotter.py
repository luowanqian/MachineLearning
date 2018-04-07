"""
Decision Tree Plotter
Reference:
    1. http://www.cnblogs.com/fantasy01/p/4595902.html
    2. http://whatbeg.com/2016/04/23/matplotlib-desiciontree.html
"""

import matplotlib.font_manager as font_manager
import os
import matplotlib.pyplot as plt


# Chinese font setting
font_path = os.path.abspath('../resources/font/msyh.ttf')
prop = font_manager.FontProperties(fname=font_path)

gAxis = None
gDecison_node = dict(boxstyle='sawtooth', fc='0.8')
gLeaf_node = dict(boxstyle='round4', fc="0.8")
gArrow_args = dict(arrowstyle='<-')
gNum_leaves = 0
gTree_depth = 0
gX_offset = 0
gY_offset = 0


def get_num_leafs(tree):
    """
    Identify the number of leaves in a tree
    """
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1

    return num_leafs


def get_tree_depth(tree):
    """
    Identiy the depth of a tree
    """
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            subtree_depth = 1 + get_tree_depth(second_dict[key])
        else:
            subtree_depth = 1

        if subtree_depth > max_depth:
            max_depth = subtree_depth

    return max_depth


def plot_node(node_text, center_point, parent_point, node_type):
    global gAxis

    gAxis.annotate(node_text, xy=parent_point,
                   xycoords='axes fraction',
                   xytext=center_point,
                   textcoords='axes fraction',
                   va='center', ha='center',
                   bbox=node_type, arrowprops=gArrow_args,
                   fontproperties=prop)


def plot_mid_text(current_point, parent_point, text_str):
    """
    Plot text between child and parent
    """
    global gAxis

    x_mid = (parent_point[0] - current_point[0])/2.0 + current_point[0]
    y_mid = (parent_point[1] - current_point[1])/2.0 + current_point[1]
    gAxis.text(x_mid, y_mid, text_str, fontproperties=prop)


def plot_tree(tree, parent_point, node_text):
    global gAxis, gDecison_node
    global gNum_leaves, gTree_depth, gX_offset, gY_offset

    num_leaves = get_num_leafs(tree)
    first_str = list(tree.keys())[0]
    current_point = (gX_offset+(1.0+float(num_leaves))/(2.0*gNum_leaves),
                     gY_offset)
    plot_mid_text(current_point, parent_point, node_text)
    plot_node(first_str, current_point, parent_point, gDecison_node)
    second_dict = tree[first_str]
    gY_offset = gY_offset - 1.0/gTree_depth
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], current_point, str(key))
        else:
            gX_offset = gX_offset + 1.0/gNum_leaves
            plot_node(second_dict[key], (gX_offset, gY_offset),
                      current_point, gLeaf_node)
            plot_mid_text((gX_offset, gY_offset), current_point, str(key))
    gY_offset = gY_offset + 1.0/gTree_depth


def create_plot(tree):
    global gAxis, gNum_leaves, gTree_depth, gX_offset, gY_offset

    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    gAxis = plt.subplot(111, frameon=False, **axprops)
    gNum_leaves = float(get_num_leafs(tree))
    gTree_depth = float(get_tree_depth(tree))
    gX_offset = -1/(2.0 * gNum_leaves)
    gY_offset = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()
