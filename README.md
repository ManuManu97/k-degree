## k-degree anonymity
### Install requirements
```
pip install -r requirements.txt
```

### Usage
```
usage: python k-degree.py k_value graph_to_anonymize.csv
usage: python print_metrics.py array_of_norm cc_supergraph apl_supergraph cc_original_graph apl_original_graph


```
#### Example
```
usage: python k-degree.py 3 Dataset/graph_friend_100_10_100.csv
usage: print_metrics ./Metrics/array_norm.csv ./Metrics/cc_supergraph.csv ./Metrics/avg_path_1000.csv 0.07286909946469872 2.0787367367367366
```

#### Dataset
In Dataset directory you can find 4 datasets:

- Dataset\graph_friend_6.csv
- Dataset\graph_friend_1000_10_100.csv
- Dataset\graph_friend_10000_100_1000.csv


#### Metrics
In Metrics directory you can find 3 files:

- Metrics/array_norm.csv
- Metrics/cc_supergraph.csv
- Metrics/avg_path_1000.csv

#### K for supergraph graph_friend_10000_100_1000.csv

3, 5, 6, 12, 13, 14, 16, 20, 24, 30, 33, 34, 38, 40, 48, 50

#### K for supergraph graph_friend_1000_10_100.csv

7, 9, 10, 12, 15, 16, 17, 20, 22


Where the first number is the number of nodes, instead, the second and the third number is the min and max link for each node.

Each line inside `.csv` file is an adjacency list. 