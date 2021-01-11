import matplotlib.pyplot as plt
import sys
import os

array_norm = [int]

if __name__ == "__main__":
    norm_path = sys.argv[1]
    cc_super_path = sys.argv[2]
    APL_path = sys.argv[3]
    cc_original = float(sys.argv[4])
    apl_original = float(sys.argv[5])


# Read arguments

    if os.path.exists(norm_path):
        # if file exist
        with open(norm_path) as f:
            content = f.readlines()
            # read each line
            content = [x.strip() for x in content]
            for line in content:
                # split name inside each line
                array_norm = line.split(",")
            array_norm = [int(i) for i in array_norm]


    if os.path.exists(cc_super_path):
        # if file exist
        with open(cc_super_path, mode='r', encoding='utf8') as f:

            array_temp = []
            array_k_super = []
            array_cc = []

            for line in f.readlines():
                array_temp = line[:-1].split(",")
                array_k_super.append(array_temp[0])
                array_cc.append(array_temp[1])

            array_k_super = [int(i) for i in array_k_super]
            array_cc = [float(i) for i in array_cc]
            #print("array_k = {}\narray cc = {}".format(array_k_super, array_cc))

    if os.path.exists(APL_path):
        # if file exist
        with open(APL_path, mode='r', encoding='utf8') as f:

            array_temp = []
            array_k_super_APL = []
            array_apl = []

            for line in f.readlines():
                array_temp = line[:-1].split(",")
                array_k_super_APL.append(array_temp[0])
                array_apl.append(array_temp[1])

            array_k_super_APL = [int(i) for i in array_k_super_APL]
            array_apl = [float(i) for i in array_apl]


# Create array with CC original graph ( CC = 0.07286909946469872 )
array_k = []
array_cc_original_graph = []
for i in range(1,50):
    array_k.append(i)
    array_cc_original_graph.append(cc_original)



    # Create graph Anonymization cost
    plt.title("Anonymization cost (graph_friend_10000_100_1000.csv)")
    plt.xlabel("K-degree")
    plt.ylabel("L(dA-d)")
    plt.plot(array_k, array_norm, 'g')
    plt.savefig("anonymization_cost.png")


    # Create graph Clustering Coefficient
    plt.title("Clustering Coefficient (graph_friend_10000_100_1000.csv)")
    plt.xlabel("K-degree")
    plt.ylabel("Clustering Coefficient")
    ax = plt.subplot(111)
    ax.plot(array_k, array_cc_original_graph, label = 'CC orignial graph', color="b", linestyle="--" )
    ax.plot(array_k_super, array_cc, label = 'CC supergraph', color= "g")
    ax.legend(shadow = "TRUE")
    plt.plot(array_k, array_cc_original_graph, 'b--', array_k_super,array_cc, 'g')
    plt.savefig("clustering_coefficient.png")


    # Create graph APL with dataset 1000 nodes

    # Create array with APL original graph ( APL = 2.0787367367367366)
    array = []
    array_avg_original = []
    for i in range(7,22):
        array.append(i)
        array_avg_original.append(apl_original)


    plt.title("Average Path Length (graph_friend_1000_10_100.csv)")
    plt.xlabel("K-degree")
    plt.ylabel("Average Path Length")
    plt.plot(array_k_super_APL,array_apl, 'g', array, array_avg_original, 'r--')

    ax = plt.subplot(111)
    ax.plot(array_k_super_APL, array_apl, label = 'APL supergraph', color="g" )
    ax.plot(array, array_avg_original, label = 'APL original', color ="r", linestyle="--")
    ax.legend(loc = "lower left", shadow="TRUE")
    plt.savefig("average_path_length.png")


