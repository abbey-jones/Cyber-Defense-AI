

def get_labeled_values(file):
    count = 0
    value_array = []
    normal_count = 0
    smurf_count = 0
    neptune_count = 0
    dict_vals = {}
    with open(file, 'r') as data_set:
        line_count = len(data_set.readlines())
        data_set.close()
    with open(file, 'r') as data_set:
        value_array = [0 for i in range(line_count)]
        while True:
            line = data_set.readline()
            if(line == ''):
                break
            else:
                values = line.split(',')
                value_array[count] = values
                if value_array[count][41] in dict_vals:
                    dict_vals[value_array[count][41]] += 1
                else:
                    dict_vals[value_array[count][41]] = 1
                count += 1
    return dict_vals

def get_correct_values(corrected_file, supervised_file):
    count = 0
    compared_vals = {}
    with open(corrected_file, 'r') as correct_set:
        line_count = len(correct_set.readlines())
        correct_set.close()
    with open(corrected_file, 'r') as correct_set:
        correct_array = [0 for i in range(line_count)]
        while True:
            line = correct_set.readline()
            if(line == ''):
                break
            else:
                values = line.split(',')
                correct_array[count] = values
            count += 1
            
    count = 0
    with open(supervised_file, 'r') as super_set:
        super_array = [0 for i in range(line_count)]
        while True:
            line = super_set.readline()
            if(line == ''):
                break
            else:
                values = line.split(',')
                super_array[count] = values
            count += 1
    
    for val in range(line_count):
        if(correct_array[val][41] == super_array[val][41]):
            if(correct_array[val][41] in compared_vals):
                compared_vals[correct_array[val][41]]['correct'] += 1
            else:
                compared_vals[correct_array[val][41]] = {'correct': 1, 'incorrect' : 0}
        else:
            if(correct_array[val][41] in compared_vals):
                compared_vals[correct_array[val][41]]['incorrect'] += 1
            else:
                compared_vals[correct_array[val][41]] = {'correct': 0, 'incorrect' : 1}
            if(super_array[val][41] in compared_vals):
                compared_vals[super_array[val][41]]['incorrect'] += 1
            else:
                compared_vals[super_array[val][41]] = {'correct': 0, 'incorrect' : 1}
    return compared_vals
    

def main():
    correct_total = 0
    incorrect_total = 0
    correct_vals = get_labeled_values('kddcup.data_10_percent_corrected')
    supervised_vals = get_labeled_values('kddcup.data_10_percent_wrong')
    compared_vals = get_correct_values('kddcup.data_10_percent_corrected','kddcup.data_10_percent_wrong')
    for key in correct_vals:
        stripped_key = key.split('.')
        if(supervised_vals[key] != correct_vals[key]):
            incorrect_no = supervised_vals[key]-correct_vals[key]
            incorrect_no_abs = abs(incorrect_no)
            incorrect_total += incorrect_no_abs
            correct_total += correct_vals[key]
        else:
            incorrect_no = 0
            correct_total += correct_vals[key]
        print('Correct values for: ' + stripped_key[0] + ': ' + str(compared_vals[key]['correct']) + '/' + str(correct_vals[key]) )
        print('Incorrect values for: ' + stripped_key[0] + ': ' + str(compared_vals[key]['incorrect']) + '/' + str(correct_vals[key]) )
    print('Total incorrect values: ' + str(int(incorrect_total/2)) + '/' + str(correct_total))


if __name__ == '__main__':
    main()