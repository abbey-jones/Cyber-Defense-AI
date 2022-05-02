import weka.core.converters as converters 
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Attribute

from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection

def attribute_selection(data):
    # search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
    # evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    as_name = "weka.attributeSelection.InfoGainAttributeEval"
    print(f"Performing {as_name} Attribute Evaluation")
    search = ASSearch(classname="weka.attributeSelection.Ranker")
    evaluator = ASEvaluation(classname=as_name)
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)

    weights = [None]*attsel.number_attributes_selected
    for rank in attsel.ranked_attributes:
        weights[int(rank[0])] = rank[1]
    
    return weights

def get_nominal_to_numeric_mapping(data):
    print("Converting protocol_type, service, flag to numeric")
    # score nominal attributes (protocol_type, service, flag)
    # protocol_type
    dict_protocol_type = {}
    attribute_idx = 1
    for index, value in enumerate(data.attribute(attribute_idx).values):
        dict_protocol_type[value] = data.attribute_stats(attribute_idx).nominal_weights[index] / data.num_instances
    # service
    dict_service = {}
    attribute_idx = 2
    for index, value in enumerate(data.attribute(attribute_idx).values):
        dict_service[value] = data.attribute_stats(attribute_idx).nominal_weights[index] / data.num_instances
    # flag
    dict_flag = {}
    attribute_idx = 3
    for index, value in enumerate(data.attribute(attribute_idx).values):
        dict_flag[value] = data.attribute_stats(attribute_idx).nominal_weights[index] / data.num_instances
    # attack types
    attack_types = []
    for attack_type in data.attribute(data.class_index).values:
        attack_types.append(attack_type)

    return dict_protocol_type, dict_service, dict_flag, attack_types

def get_weights(data):
    # get info gain for each attribute
    weights = attribute_selection(data)
    return weights

def insert_meta_feature(data, weights, dict_protocol_type, dict_service, dict_flag):   
    # TODO: calculate average attribute via grouping instances by class/cluster
    meta_dict = {}
    count_dict = {}
    for class_attribute in data.attribute(data.class_index).values:
        meta_dict[class_attribute] = 0
        count_dict[class_attribute] = 0
    # TODO: limit range to decrease time
    for i in range(int(data.num_instances/20)):
        if i % 10000 == 0:
            print(f"calculating metascore {i}/{data.num_instances}")
        for class_attribute in data.attribute(data.class_index).values:
            instance = data.get_instance(i)
            if class_attribute == instance.get_string_value(data.class_index):
                # use info gain weight to calculate metascore
                metascore = 0
                # multiply weight by attribute value, sum metascores
                for index, weight in enumerate(weights):
                    # protocol_type
                    if index == 1:
                        value = instance.get_string_value(index)
                        value = dict_protocol_type[value] / len(dict_protocol_type)
                    # service
                    elif index == 2:
                        value = instance.get_string_value(index)
                        value = dict_service[value] / len(dict_service)
                    # flag
                    elif index == 3:
                        value = instance.get_string_value(index)
                        value = dict_flag[value] / len(dict_flag)
                    # src_bytes
                    elif index == 4 :
                        value = instance.get_value(index)
                        continue
                    # dst_bytes
                    elif index == 5:
                        src_bytes = value
                        dst_bytes = instance.get_value(index)
                        sum_bytes = dst_bytes + src_bytes
                        if sum_bytes > 0:
                            value = src_bytes / sum_bytes
                            metascore += weights[index-1]*value
                            value = dst_bytes / sum_bytes
                        else:
                            value = 0
                    else:
                        if data.attribute(index).upper_numeric_bound > 1:
                            value = instance.get_value(index) / data.attribute(index).upper_numeric_bound
                        else:
                            value = instance.get_value(index)
                    metascore += weight*value
                # normalize against number of attributes
                meta_dict[class_attribute] += metascore / len(weights)
                count_dict[class_attribute] += 1

    # weight metascore against class/cluster weight
    for class_attribute in data.attribute(data.class_index).values:
        meta_dict[class_attribute] = meta_dict[class_attribute] * (count_dict[class_attribute] / data.num_instances)

    # insert metascore as first attribute
    data.insert_attribute(Attribute.create_numeric("metascore"), 0)
    for i in range(data.num_instances):
        if i % 10000 == 0:
            print(f"inserting metascore {i}/{data.num_instances}")
        data.get_instance(i).set_value(0, meta_dict[data.get_instance(i).get_string_value(data.class_index)])

    return data
