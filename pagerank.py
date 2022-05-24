import os 



def pagerank(node_list, beta=0.85):
    node_list["A"] = (node_list["B"] * 1/3 + node_list["C"] * 1/2) * beta + (1-beta) * 1 / len(node_list)
    node_list["B"] = (node_list["C"] * 1/2) * beta + (1-beta) * 1 / len(node_list)
    node_list["C"] = (node_list["A"] + node_list["B"] * 1/3) * beta + (1-beta) * 1 / len(node_list)
    node_list["D"] = (node_list["B"] * 1/3) * beta + (1-beta) * 1 / len(node_list)
    print(node_list)
    return node_list


if __name__ == "__main__":
    node_list = {"A":1, "B":1, "C":1, "D":1}
    for i in range(20):
        node_list=pagerank(node_list)

