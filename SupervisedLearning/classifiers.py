import weka.core.jvm as jvm

import weka.core.converters as converters 
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

from datetime import datetime
import traceback

data_dir = "data/"
data_file = "kddcup_test_data_unlabeled_10_percent.csv"
eval_file = "kddcup_data_10_percent_corrected.csv"

names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

def prepend_attributes_to_file(input_kdd, output_filename="tmp", labelled=False):
    
    labelstring = ",".join(names)
    if labelled:
        labelstring += ",category\n"
        # output_filename += "_unlabelled"

    count = 0

    tmp_file = f"{output_filename}.csv"
    with open(data_dir + tmp_file, "w+") as f:
        f.write(labelstring)
        with open(data_dir+input_kdd, "r") as f_kdd:
            kdd_lines = f_kdd.readlines()
            for line in kdd_lines:
                if not labelled:
                    line = line.strip("\n")
                    line += ",?\n"
                f.write(line)
                count += 1
    return tmp_file

def naive_bayes(data):
    classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(classifier, data, 10, Random(42))
    print(evaluation.summary())
    print("pctCorrect: " + str(evaluation.percent_correct))
    print("incorrect: " + str(evaluation.incorrect))

def load_data(file):
    data = converters.load_any_file(data_dir + file)
    data.class_is_last()
    return data
    
def main():
    
    tmp_file = prepend_attributes_to_file(eval_file, labelled=True)
    data = load_data(tmp_file)
    naive_bayes(data)

if __name__ == "__main__":
    
    try:
        jvm.start(max_heap_size="512m")
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()