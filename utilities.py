import weka.core.converters as converters

def prepend_attributes_to_file(data_dir, input_kdd, output_filename="tmp", labeled=False):
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

def load_data(data_dir, input_kdd, prepend=True, output_filename="tmp", labeled=False):
    if prepend:
        tmp_file = prepend_attributes_to_file(data_dir, input_kdd, output_filename=output_filename, labeled=labeled)
    data = converters.load_any_file(data_dir + tmp_file)
    if labeled:
        data.class_is_last()
    return data