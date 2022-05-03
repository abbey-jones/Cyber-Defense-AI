from weka.core.converters import Loader
from weka.clusterers import Clusterer, ClusterEvaluation
import weka.core.converters as converters

def eval_clusterer(clusterer, eval_data, show_eval=True):
    # classes to clusters
    evl = ClusterEvaluation()
    evl.set_model(clusterer)
    evl.test_model(eval_data)
    if show_eval:
        print("Cluster results")
        print(evl.cluster_results)
        print("Classes to clusters")
        print(evl.classes_to_clusters)

def build_clusterer(algo, data, num_clusters=None):
    """
    https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/clusterers/cluster_data.py
    https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/clusterers/classes_to_clusters.py
    """
    clusterer_name = algo
    print(f"Training {clusterer_name} clusterer")
    if num_clusters:
        clusterer = Clusterer(classname=clusterer_name, options=["-N", f"{num_clusters}"])
    else:
        clusterer = Clusterer(classname=clusterer_name)
    clusterer.build_clusterer(data)
    print("done")

    return clusterer
