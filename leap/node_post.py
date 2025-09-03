import networkx as nx
import pickle
import os
from pronto import Ontology
import functools

from .utils import get_default_save_path


# 全局变量用于缓存图
_G = None
_save_path = ""
_hp_obo_path = ""

def set_config(save_path: str, hp_obo_path: str):
    """ 设置配置参数 """
    global _save_path, _hp_obo_path, _G
    _save_path = save_path
    _hp_obo_path = hp_obo_path
    # 重置图缓存，因为路径可能变化
    _G = None


# HPO图构建函数
def hpo_obo_extraction():
    """ 从HPO OBO文件中提取HPO节点间的关系，创建edge对 """
    hp_obo_path = _hp_obo_path
    if not hp_obo_path:
        raise ValueError("HP_OBO_PATH not configured. Please call set_config() first.")
    hpo_root = Ontology(hp_obo_path)["HP:0000118"]
    edge_list = []
    def generate_hpo_edges(hpo_node):
        if hpo_node.subclasses(distance=1, with_self=False) != []:
            for i in hpo_node.subclasses(distance=1, with_self=False):
                edge_list.append((hpo_node.id, i.id))
                generate_hpo_edges(i)     
    generate_hpo_edges(hpo_root)
    return edge_list

def build_hpo_graph(edge_list):
    """ nx从HPO节点间的关系创建有向图 """
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    return G

def get_hpo_graph():
    """ 懒加载HPO图 """
    global _G
    if _G is None:
        savepath = _save_path
        file_path = os.path.join(savepath, "hpo_nxgraph.pkl")
        
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                _G = pickle.load(f)
        else:
            _G = build_hpo_graph(hpo_obo_extraction())
            with open(file_path, 'wb') as f:
                pickle.dump(_G, f)
    return _G


# 添加缓存装饰器，缓存节点间的可达性检查
@functools.lru_cache(maxsize=1024)
def cached_has_path(source, target):
    G = get_hpo_graph()
    return nx.has_path(G, source, target)

def retain_furthest_nodes(nodes):
    """ 从HPO图中保留最远的节点 """
    G = get_hpo_graph()
    # 判断节点是否在图里
    nodes_deleted = []
    for node in nodes:
        if not G.has_node(node):
            print(f"WARNING: Node {node} not in graph")
        else:
            nodes_deleted.append(node)
            
    retained = []
    for node in nodes_deleted:
        # 检查当前节点是否有后代存在于列表中
        has_descendant_in_list = False
        for other_node in nodes_deleted:
            if node != other_node and cached_has_path(node, other_node):
                has_descendant_in_list = True
                break
        if not has_descendant_in_list:
            retained.append(node)
    return retained


def find_same_branch_groups(L):
    G = get_hpo_graph()
    # 检查所有节点存在
    for node in L:
        if not G.has_node(node):
            raise ValueError(f"Node {node} not in graph")
    
    remaining = set(L)
    groups = []
    
    while remaining:
        max_group = []
        # 遍历每个节点，寻找最长路径
        for start_node in remaining:
            # 正向搜索（沿后继方向）
            forward_path = []
            stack = [(start_node, [start_node])]
            while stack:
                current, path = stack.pop()
                successors = [n for n in G.successors(current) if n in remaining]
                if not successors:
                    if len(path) > len(forward_path):
                        forward_path = path.copy()
                else:
                    for succ in successors:
                        stack.append((succ, path + [succ]))
            
            # 反向搜索（沿前驱方向）
            reverse_path = []
            stack = [(start_node, [start_node])]
            while stack:
                current, path = stack.pop()
                predecessors = [n for n in G.predecessors(current) if n in remaining]
                if not predecessors:
                    if len(path) > len(reverse_path):
                        reverse_path = path.copy()
                else:
                    for pred in predecessors:
                        stack.append((pred, [pred] + path))
            
            # 选择更长的路径
            current_path = forward_path if len(forward_path) >= len(reverse_path) else reverse_path
            if len(current_path) > len(max_group):
                max_group = current_path
        
        # 提取路径中的剩余节点作为分组
        current_group = [n for n in max_group if n in remaining]
        if current_group:
            groups.append(tuple(current_group))
            remaining -= set(current_group)
        else:
            # 处理剩余孤立节点
            for node in remaining:
                groups.append((node,))
            remaining = set()
    
    # 去重并排序
    unique_groups = list({tuple(sorted(g)) for g in groups})
    unique_groups.sort(key=lambda x: (-len(x), x))
    return unique_groups

def merge_child_nodes(nodelist):
    Kyphosis = ['HP:0004633', 'HP:0005619', 'HP:0003423', 'HP:0002947', 'HP:0002942', 'HP:0008453', 'HP:0002751', 'HP:0004619']
    Scoliosis = ['HP:0100884', 'HP:0005659', 'HP:0008458', 'HP:0004626', 'HP:0002944', 'HP:0003423', 'HP:0002943', 'HP:0008453', 'HP:0002751', 'HP:0004619']
    temp = []
    for i in nodelist:
        if i in Kyphosis:
            temp.append('HP:0002808')
        elif i in Scoliosis:
            temp.append("HP:0002650")
        else:
            temp.append(i)
    
    return list(set(temp))
