import argparse
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--test_file', type=str, default= 'data/today/test_with_explanation.txt')
parser.add_argument('--output_dir', type=str, default= 'experiment_result/best_model')
parser.add_argument('--name', type=str, default= 'test_with_explanation')

args = parser.parse_args()
print(args)


def calculate_today_accruacy():
    # load the original test data
    test = []
    with open(args.test_file) as f:
        for i,line in enumerate(f):
            test.append(line)

    test_sample = []
    for i in range(0,len(test), 4):
        test_sample.append(test[i:i+4])
    
    # load the model prediction scores
    modify_correct_correct_prob_list = []
    modify_correct_wrong_prob_list = []
    origin_correct_prob_list = []
    origin_wrong_prob_list = []
    with open(args.output_dir+'/'+args.name+'.txt') as f:
      for i,line in enumerate(f):
        data = line.strip().split('\t')
        positive_score = float(data[1])
        negative_score = float(data[2])
        positive_score = positive_score/(positive_score+ negative_score)

        if i%4 == 0:
          origin_correct_prob_list.append(positive_score)
        elif i%4 == 1:
          origin_wrong_prob_list.append(positive_score)
        elif i%4 == 2:
          modify_correct_correct_prob_list.append(positive_score)
        else:
          modify_correct_wrong_prob_list.append(positive_score)
    
    # calculate accuracy 
    count_1 = 0
    general_count = 0
    for i in range(len(origin_correct_prob_list)):
        origin_correct_prob = origin_correct_prob_list[i]
        origin_wrong_prob = origin_wrong_prob_list[i]
        modify_correct_correct_prob = modify_correct_correct_prob_list[i]
        modify_correct_wrong_prob = modify_correct_wrong_prob_list[i]

        temp1 = modify_correct_correct_prob - origin_correct_prob
        temp2 = modify_correct_wrong_prob - origin_wrong_prob
        
        # Binary classfication: Probability change in the right direction should be larger than the wrong direction
        if temp1 > temp2:
            count_1 += 1

        general_count += 1

    print('Final accuracy for today: '+ str(count_1/general_count))


calculate_today_accruacy()