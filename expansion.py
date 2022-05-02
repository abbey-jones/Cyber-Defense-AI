import weka.core.converters as converters 
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.dataset import Attribute

from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection

def attribute_selection(data):
    # search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
    # evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    search = ASSearch(classname="weka.attributeSelection.Ranker")
    evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)

    weights = [None]*attsel.number_attributes_selected
    for rank in attsel.ranked_attributes:
        weights[int(rank[0])] = rank[1]
    
    return weights

def get_nominal_to_numeric_mapping(data):
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

    return dict_protocol_type, dict_service, dict_flag

def insert_meta_feature(data, dict_protocol_type, dict_service, dict_flag):
    # get info gain for each attribute
    weights = attribute_selection(data)
    
    # TODO: calculate average attribute via grouping instances by class/cluster
    meta_dict = {}
    count_dict = {}
    for class_attribute in data.attribute(data.class_index).values:
        meta_dict[class_attribute] = 0
        count_dict[class_attribute] = 0
        for i in range(data.num_instances):
            instance = data.get_instance(i)
            if class_attribute == instance.get_string_value(data.class_index):
                # use info gain weight to calculate metascore
                metascore = 0
                for index, weight in enumerate(weights):
                    if index == 1:
                        value = data.get_instance(0).get_string_value(index)
                        value = dict_protocol_type[value]
                    elif index == 2:
                        value = data.get_instance(0).get_string_value(index)
                        value = dict_service[value]
                    elif index == 3:
                        value = data.get_instance(0).get_string_value(index)
                        value = dict_flag[value]
                    else:
                        value = data.get_instance(0).get_value(index)
                    metascore += weight*value
                meta_dict[class_attribute] += metascore
                count_dict[class_attribute] += 1
    print(meta_dict)
    print(count_dict)

    # TODO: weight metascore against class/cluster weight
    for class_attribute in data.attribute(data.class_index).values:
        meta_dict[class_attribute] = meta_dict[class_attribute] * count_dict[class_attribute] / data.num_instances
    print(meta_dict)

    # TODO: insert metascore as first attribute
    # data.insert_attribute(Attribute.create_numeric("metascore"), 0)
    # data.get_instance(0).set_value(0, metascore)
