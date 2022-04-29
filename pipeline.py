import weka.core.jvm as jvm

from datetime import datetime
import traceback

from clustering import build_clusterer, eval_clusterer
from utilities import load_data

data_dir = "/home/student/Cyber-Defense-AI/data/"
data_file = "kddcup.testdata.unlabeled_10_percent"
eval_file = "kddcup.data_10_percent_corrected"

# def main():
#     pass

def main():
    start_time = datetime.now()
    do_eval = False
    
    data = load_data(data_dir, data_file)
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
        jvm.start(max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()