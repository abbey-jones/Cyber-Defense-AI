def get_percent_correct(data, eval):
    print("evaluating")
    num_correct = 0
    if data.num_instances == eval.num_instances:
        for i in range(data.num_instances):
            data_instance = data.get_instance(i)
            eval_instance = eval.get_instance(i)
            if eval_instance.get_string_value(eval.class_index) == data_instance.get_string_value(data.class_index):
                num_correct += 1
    return num_correct/data.num_instances
