import weka.core.jvm as jvm
from weka.core.dataset import Attribute
from weka.filters import Filter

from datetime import datetime
import traceback

from expansion import attribute_selection, get_nominal_to_numeric_mapping, get_weights,  insert_meta_feature
from clustering import build_clusterer, eval_clusterer
from classifiers import build_classifier
from utilities import load_data

data_dir = "/home/student/Cyber-Defense-AI/data/"
data_file = "kddcup.testdata.unlabeled_10_percent"
eval_file = "kddcup.data_10_percent_corrected"
all_eval_file = "kddcup.data.corrected"

def main_build_classify(weights, dict_protocol_type, dict_service, dict_flag):
    data = load_data(data_dir, eval_file, prepend=True, output_filename="eval", labeled=True)
    data = insert_meta_feature(data, weights, dict_protocol_type, dict_service, dict_flag)
    classifier = build_classifier(data)
    return data, classifier

def main_build_cluster():
    # data = load_data(data_dir, data_file)
    data = load_data(data_dir, eval_file, prepend=True, output_filename="eval", labeled=True)
    data = remove_last_attribute(data)
    num_clusters = 10
    clusterer = build_clusterer(data, num_clusters=num_clusters)
    # clusterer = build_clusterer(data, num_clusters=int(data.num_instances/10))
    return data, clusterer, num_clusters

def main_metafeature_cluster(data, clusterer, num_clusters, weights, dict_protocol_type, dict_service, dict_flag):
    new_cluster = []
    for i in range(int(data.num_instances)):
        if i % 10000 == 0:
            print(f"getting cluster {i}/{data.num_instances}")
        instance = data.get_instance(i)
        cl = clusterer.cluster_instance(instance)
        new_cluster.append(cl)

    labels = []
    for i in range(num_clusters):
        labels.append(str(i))

    data.insert_attribute(Attribute.create_nominal("cluster", labels), data.num_attributes)
    for index, cl in enumerate(new_cluster):
        instance = data.get_instance(index)
        instance.set_value(data.num_attributes-1, cl)
    data.class_is_last()

    data = insert_meta_feature(data, weights, dict_protocol_type, dict_service, dict_flag)

    return data

def remove_last_attribute(data):
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "last"])
    remove.inputformat(data)
    filtered = remove.filter(data)
    return filtered

def main_run_classifier(data, classifier, attack_types):
    new_classes = []
    for i in range(int(data.num_instances)):
        if i % 10000 == 0:
            print(f"getting classification {i}/{data.num_instances}")
        instance = data.get_instance(i)
        cl = classifier.classify_instance(instance)
        new_classes.append(cl)

    data.insert_attribute(Attribute.create_nominal("category", attack_types), data.num_attributes)
    for index, cl in enumerate(new_classes):
        instance = data.get_instance(index)
        instance.set_value(data.num_attributes-1, cl)
    data.class_is_last()
    return data


if __name__ == "__main__":
    try:
        jvm.start(max_heap_size="4096m")
        print()
        # use full labeled set to get weighting function for metascore, all nominal variables
        all_labeled_data = load_data(data_dir, all_eval_file, prepend=True, output_filename="eval", labeled=True)
        dict_protocol_type, dict_service, dict_flag, attack_types = get_nominal_to_numeric_mapping(all_labeled_data)
        weights = get_weights(all_labeled_data)
        # build classifier using labeled data and metascore
        eval, classifier = main_build_classify(weights, dict_protocol_type, dict_service, dict_flag)
        # build clusterer; strips class attribute from labeled data used in above classifier
        # for better verification
        data, clusterer, num_clusters = main_build_cluster()
        # label unlabeled data and calculate metascore
        data = main_metafeature_cluster(data, clusterer, num_clusters, weights, dict_protocol_type, dict_service, dict_flag)
        # use classifier
        data = main_run_classifier(data, classifier, attack_types)

    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()