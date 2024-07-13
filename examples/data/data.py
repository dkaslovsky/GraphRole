import json

import networkx as nx

# json serialized data representing the karate club graph from NetworkX
karate_club_graph_data = '{"0": {"1": {"weight": 4}, "2": {"weight": 5}, "3": {"weight": 3}, "4": {"weight": 3}, "5": {"weight": 3}, "6": {"weight": 3}, "7": {"weight": 2}, "8": {"weight": 2}, "10": {"weight": 2}, "11": {"weight": 3}, "12": {"weight": 1}, "13": {"weight": 3}, "17": {"weight": 2}, "19": {"weight": 2}, "21": {"weight": 2}, "31": {"weight": 2}}, "1": {"0": {"weight": 4}, "2": {"weight": 6}, "3": {"weight": 3}, "7": {"weight": 4}, "13": {"weight": 5}, "17": {"weight": 1}, "19": {"weight": 2}, "21": {"weight": 2}, "30": {"weight": 2}}, "2": {"0": {"weight": 5}, "1": {"weight": 6}, "3": {"weight": 3}, "7": {"weight": 4}, "8": {"weight": 5}, "9": {"weight": 1}, "13": {"weight": 3}, "27": {"weight": 2}, "28": {"weight": 2}, "32": {"weight": 2}}, "3": {"0": {"weight": 3}, "1": {"weight": 3}, "2": {"weight": 3}, "7": {"weight": 3}, "12": {"weight": 3}, "13": {"weight": 3}}, "4": {"0": {"weight": 3}, "6": {"weight": 2}, "10": {"weight": 3}}, "5": {"0": {"weight": 3}, "6": {"weight": 5}, "10": {"weight": 3}, "16": {"weight": 3}}, "6": {"0": {"weight": 3}, "4": {"weight": 2}, "5": {"weight": 5}, "16": {"weight": 3}}, "7": {"0": {"weight": 2}, "1": {"weight": 4}, "2": {"weight": 4}, "3": {"weight": 3}}, "8": {"0": {"weight": 2}, "2": {"weight": 5}, "30": {"weight": 3}, "32": {"weight": 3}, "33": {"weight": 4}}, "9": {"2": {"weight": 1}, "33": {"weight": 2}}, "10": {"0": {"weight": 2}, "4": {"weight": 3}, "5": {"weight": 3}}, "11": {"0": {"weight": 3}}, "12": {"0": {"weight": 1}, "3": {"weight": 3}}, "13": {"0": {"weight": 3}, "1": {"weight": 5}, "2": {"weight": 3}, "3": {"weight": 3}, "33": {"weight": 3}}, "14": {"32": {"weight": 3}, "33": {"weight": 2}}, "15": {"32": {"weight": 3}, "33": {"weight": 4}}, "16": {"5": {"weight": 3}, "6": {"weight": 3}}, "17": {"0": {"weight": 2}, "1": {"weight": 1}}, "18": {"32": {"weight": 1}, "33": {"weight": 2}}, "19": {"0": {"weight": 2}, "1": {"weight": 2}, "33": {"weight": 1}}, "20": {"32": {"weight": 3}, "33": {"weight": 1}}, "21": {"0": {"weight": 2}, "1": {"weight": 2}}, "22": {"32": {"weight": 2}, "33": {"weight": 3}}, "23": {"25": {"weight": 5}, "27": {"weight": 4}, "29": {"weight": 3}, "32": {"weight": 5}, "33": {"weight": 4}}, "24": {"25": {"weight": 2}, "27": {"weight": 3}, "31": {"weight": 2}}, "25": {"23": {"weight": 5}, "24": {"weight": 2}, "31": {"weight": 7}}, "26": {"29": {"weight": 4}, "33": {"weight": 2}}, "27": {"2": {"weight": 2}, "23": {"weight": 4}, "24": {"weight": 3}, "33": {"weight": 4}}, "28": {"2": {"weight": 2}, "31": {"weight": 2}, "33": {"weight": 2}}, "29": {"23": {"weight": 3}, "26": {"weight": 4}, "32": {"weight": 4}, "33": {"weight": 2}}, "30": {"1": {"weight": 2}, "8": {"weight": 3}, "32": {"weight": 3}, "33": {"weight": 3}}, "31": {"0": {"weight": 2}, "24": {"weight": 2}, "25": {"weight": 7}, "28": {"weight": 2}, "32": {"weight": 4}, "33": {"weight": 4}}, "32": {"2": {"weight": 2}, "8": {"weight": 3}, "14": {"weight": 3}, "15": {"weight": 3}, "18": {"weight": 1}, "20": {"weight": 3}, "22": {"weight": 2}, "23": {"weight": 5}, "29": {"weight": 4}, "30": {"weight": 3}, "31": {"weight": 4}, "33": {"weight": 5}}, "33": {"8": {"weight": 4}, "9": {"weight": 2}, "13": {"weight": 3}, "14": {"weight": 2}, "15": {"weight": 4}, "18": {"weight": 2}, "19": {"weight": 1}, "20": {"weight": 1}, "23": {"weight": 4}, "26": {"weight": 2}, "27": {"weight": 4}, "28": {"weight": 2}, "29": {"weight": 2}, "30": {"weight": 3}, "31": {"weight": 4}, "32": {"weight": 5}, "22": {"weight": 3}}}'


def load_nx_karate_club_graph(weighted=False) -> nx.Graph:
    graph_dict = json.loads(karate_club_graph_data, object_hook=int_keys)
    if not weighted:
        for edges in graph_dict.values():
            for attrs in edges.values():
                attrs.pop('weight', None)
    return nx.Graph(graph_dict)


def int_keys(dict_in: dict) -> dict:
    dict_out = dict()
    for key, val in dict_in.items():
        try:
            key_out = int(key)
        except ValueError:
            key_out = key
        dict_out[key_out] = val
    return dict_out
