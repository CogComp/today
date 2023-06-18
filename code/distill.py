
from collections import defaultdict
import time
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default= 'explanation_verifier_ce_finetuned_ptntime_5_epoch_')

parser.add_argument('--general_name', type=str, default= 'gpt3_train_matres_gpt3_generated_with_explanation_0_old')

parser.add_argument('--as_name', type=str, default= 'gpt3_train_matres_gpt3_generated_without_explanation_0')

parser.add_argument('--general_model', type=str, default='finetune_ptntime_annotated_without_explanation50_with_explanation10')

parser.add_argument('--as_model', type=str, default='finetune_ptntime_annotated_without_explanation')

parser.add_argument('--exp_model', type=str, default='finetune_ptntime_annotated_without_explanation')

parser.add_argument('--folder', type=str, default='gpt3_train_matres_gpt3_generated_with_or_without_explanation')

parser.add_argument('--combined_train', type=str, default='../data/today/today_train_with_explanation.txt')

parser.add_argument('--extra_name', type=str, default='')

parser.add_argument('--output_dir', type=str, default='../experiment')




args = parser.parse_args()
print(args)


# Explanation sentence verifier
exp_data_list = []
with open(args.folder+'/'+args.exp_name+'.txt') as f:
    for i, line in enumerate(f):
        exp_data_list.append(line)


exp_accept_list = []
score_temp = []
with open(args.output_dir+'/'+args.exp_model+'_'+args.exp_name+'.txt') as f:
  for i,line in enumerate(f):
    data = line.strip().split('\t')
    positive_score = float(data[1])
    negative_score = float(data[2])
    positive_score = positive_score / (positive_score + negative_score)
    if positive_score > negative_score:
        exp_accept_list.append(i)



# General verifier
origin_data_list = []
origin_data_temp = []
with open(args.folder+'/'+args.general_name+'.txt') as f:
    for i, line in enumerate(f):
        origin_data_temp.append(line)

for i in range(0,len(origin_data_temp), 4):
    origin_data_list.append(origin_data_temp[i:i+4])


final_score_list = []
score_temp = []
with open(args.output_dir+'/'+args.general_model+'_'+args.general_name+'.txt') as f:
  for i,line in enumerate(f):
    data = line.strip().split('\t')
    positive_score = float(data[1])
    negative_score = float(data[2])
    positive_score = positive_score / (positive_score + negative_score)
    score_temp.append(positive_score)

for i in range(0,len(score_temp), 4):
    final_score_list.append(score_temp[i:i+4])



# Additional sentence verifier
origin_data_list_1 = []
origin_data_temp_1 = []
with open(args.folder+'/'+args.as_name+'.txt') as f:
    for i, line in enumerate(f):
        origin_data_temp_1.append(line)

for i in range(0,len(origin_data_temp_1), 4):
    origin_data_list_1.append(origin_data_temp_1[i:i+4])


final_score_list_1 = []
score_temp_1 = []
with open(args.output_dir+'/'+args.as_model+'_'+args.as_name+'.txt') as f:
  for i,line in enumerate(f):
    data = line.strip().split('\t')
    positive_score = float(data[1])
    negative_score = float(data[2])
    positive_score = positive_score / (positive_score + negative_score)
    score_temp_1.append(positive_score)

for i in range(0,len(score_temp_1), 4):
    final_score_list_1.append(score_temp_1[i:i+4])



f_1 = open(args.output_dir+'/'+args.extra_name,'w')

# first write down today train data
with open(args.combined_train) as f:
    for i,line in enumerate(f):
        f_1.write(line)

f_1.write('\n')

# Distill GPT-3 data
for i,score in enumerate(final_score_list):

    if i in exp_accept_list:
        origin_correct_prob = score[0]
        origin_wrong_prob = score[1]
        modify_correct_prob = score[2]
        modify_wrong_prob = score[3]

        temp1 = modify_correct_prob - origin_correct_prob
        temp2 = modify_wrong_prob - origin_wrong_prob

        origin_correct_prob_1 = final_score_list_1[i][0]
        origin_wrong_prob_1 = final_score_list_1[i][1]
        modify_correct_prob_1 = final_score_list_1[i][2]
        modify_wrong_prob_1 = final_score_list_1[i][3]

        temp1_1 = modify_correct_prob_1 - origin_correct_prob_1
        temp2_1 = modify_wrong_prob_1 - origin_wrong_prob_1


        if temp1 > 0 and temp2 < 0 and temp1_1 > temp2_1: # stricter rule for general explaination verifer
            for j in range(4):
                f_1.write(origin_data_list[i][j])

print("Distilled data is saved to: ",args.output_dir+'/'+args.extra_name)