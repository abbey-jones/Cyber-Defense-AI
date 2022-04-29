import weka.core.jvm as jvm

from weka.core.converters import Loader
from weka.clusterers import Clusterer, ClusterEvaluation
import weka.core.converters as converters

from datetime import datetime
import traceback

data_dir = "/home/student/Cyber-Defense-AI/data/"
data_file = "kddcup.testdata.unlabeled_10_percent"
eval_file = "kddcup.data_10_percent_corrected"

def prepend_attributes_to_file(input_kdd, output_filename="tmp", labeled=False):
    labels = []
    with open(data_dir+ 'kddcup.names', 'r') as names:
        count = 0
        for line in names.readlines():
            count += 1
            if count == 1:
                continue
            else:
                labels.append(line.strip().split(':')[0])

    labelstring = ""
    for i, label in enumerate(labels):
        if i+1 == len(labels):
            labelstring += label
        else:
            labelstring += f"{label},"
    if labeled:
        labelstring += ",category"

    tmp_file = f"{output_filename}.csv"
    with open(data_dir+tmp_file, "w+") as f:
        f.write(labelstring+'\n')
        with open(data_dir+input_kdd, "r") as f_kdd:
            kdd_lines = f_kdd.readlines()
            for line in kdd_lines:
                f.write(line)
    
    return tmp_file

def load_data(input_kdd, output_filename="tmp", labeled=False):
    tmp_file = prepend_attributes_to_file(input_kdd, output_filename=output_filename, labeled=labeled)
    data = converters.load_any_file(data_dir + tmp_file)
    return data

def eval(clusterer, eval_data, show_eval=True):
    # classes to clusters
    evl = ClusterEvaluation()
    evl.set_model(clusterer)
    evl.test_model(eval_data)
    if show_eval:
        print("Cluster results")
        print(evl.cluster_results)
        print("Classes to clusters")
        print(evl.classes_to_clusters)

def build_clusterer(data, clusters=None):
    """
    https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/clusterers/cluster_data.py
    https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/clusterers/classes_to_clusters.py
    """
    print("Training SimpleKMeans clusterer")
    if clusters:
        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", f"{clusters}"])
    else:
        clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans")
    clusterer.build_clusterer(data)
    print("done")

    return clusterer

def main():
    start_time = datetime.now()
    do_eval = False
    
    data = load_data(data_file)
    load_time = datetime.now()
    
    eval_load_time = None
    if eval_file and do_eval:
        eval_data = load_data(eval_file, output_filename="eval", labeled=True)
        eval_data.class_is_last()
        eval_load_time = datetime.now()
    
    clusterer = build_clusterer(data, clusters=23)
    build_clusterer_time = datetime.now()
    # print(clusterer)

    # print("Results")
    # for i, inst in enumerate(data):
    #     cl = clusterer.cluster_instance(inst)
    #     dist = clusterer.distribution_for_instance(inst)
    #     print(str(i+1) + ": cluster=" + str(cl) + ", distribution=" + str(dist))

    eval_time = None
    if eval_file and do_eval:
        eval(clusterer, eval_data, show_eval=False)
        eval_time = datetime.now()

    print(f"elapsed time (load data): {load_time - start_time}")
    if eval_load_time:
        print(f"elapsed time (load eval data): {eval_load_time - load_time}")
        print(f"elapsed time (cluster): {build_clusterer_time - eval_load_time}")
    else:
        print(f"elapsed time (cluster): {build_clusterer_time - load_time}")
    if eval_time:
        print(f"elapsed time (eval): {eval_time - build_clusterer_time}")
    print(f"elapsed time (total): {datetime.now() - start_time}")

if __name__ == "__main__":
    try:
        jvm.start(max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
