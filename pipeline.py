import weka.core.jvm as jvm

from datetime import datetime
import traceback

from expansion import attribute_selection, get_nominal_to_numeric_mapping, insert_meta_feature
from clustering import build_clusterer, eval_clusterer
from classifiers import naive_bayes
from utilities import load_data

data_dir = "/home/student/Cyber-Defense-AI/data/"
data_file = "kddcup.testdata.unlabeled_10_percent"
eval_file = "kddcup.data_10_percent_corrected"
all_eval_file = "kddcup.data.corrected"

def main_meta_feature(dict_protocol_type, dict_service, dict_flag):
    data = load_data(data_dir, eval_file, prepend=True, output_filename="eval", labeled=True)
    # data = load_data(data_dir, data_file, prepend=False)
    insert_meta_feature(data, dict_protocol_type, dict_service, dict_flag)

def main_classify():
    data = load_data(data_dir, eval_file, prepend=True, output_filename="eval", labeled=True)
    naive_bayes(data)

def main_cluster():
    start_time = datetime.now()
    do_eval = False
    
    data = load_data(data_dir, data_file)
    load_time = datetime.now()
    
    eval_load_time = None
    if eval_file and do_eval:
        eval_data = load_data(data_dir, eval_file, prepend=True, output_filename="eval", labeled=True)
        eval_load_time = datetime.now()
    
    clusterer = build_clusterer(data, clusters=len(data/10))
    build_clusterer_time = datetime.now()
    # print(clusterer)

    # print("Results")
    # for i, inst in enumerate(data):
    #     cl = clusterer.cluster_instance(inst)
    #     dist = clusterer.distribution_for_instance(inst)
    #     print(str(i+1) + ": cluster=" + str(cl) + ", distribution=" + str(dist))

    eval_time = None
    if eval_file and do_eval:
        eval_clusterer(clusterer, eval_data, show_eval=False)
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
        jvm.start(max_heap_size="4096m")
        all_labeled_data = load_data(data_dir, all_eval_file, prepend=True, output_filename="eval", labeled=True)
        dict_protocol_type, dict_service, dict_flag = get_nominal_to_numeric_mapping(all_labeled_data)
        main_meta_feature(dict_protocol_type, dict_service, dict_flag)

        # main_cluster()
        # main_test_data_patching()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()