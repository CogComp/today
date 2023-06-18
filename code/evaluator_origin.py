import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='data/tracie/tracie_test.txt')
parser.add_argument('--output_dir', type=str, default= 'experiment_result/best_model')
parser.add_argument('--name', type=str, default='tracie_test')

args = parser.parse_args()
print(args)


def evaluate_tracie_style():
    glines = [x.strip() for x in open(args.test_file).readlines()]
    plines = [x.strip() for x in open(args.output_dir + "/" + args.name + ".txt").readlines()]
    assert len(glines) == len(plines)
    total = 0
    correct = 0
    total_start = 0
    correct_start = 0
    total_end = 0
    correct_end = 0
    story_prediction_map = {}
    for i, l in enumerate(glines):
        if "story:" in l.split("\t")[0]:
            story = l.split("\t")[0].split("story:")[1]
        else:
            story = "no story"
        if story not in story_prediction_map:
            story_prediction_map[story] = []
        label = l.split("\t")[1].split()[1]
        data = plines[i].strip().split('\t')

        positive_score = float(data[1]) / (float(data[1]) + float(data[2]))
        negative_score = float(data[2]) / (float(data[1]) + float(data[2]))
        if positive_score >= negative_score:
            p = 'positive'
        else:
            p = 'negative'
        total += 1
        if label == p:
            correct += 1
            story_prediction_map[story].append(True)
        else:
            story_prediction_map[story].append(False)
        if "starts before" in l or "starts after" in l:
            total_start += 1
            if label == p:
                correct_start += 1
        else:
            total_end += 1
            if label == p:
                correct_end += 1
    s_total = 0
    s_correct = 0
    for key in story_prediction_map:
        s_total += 1
        cv = True
        for v in story_prediction_map[key]:
            cv = cv and v
        if cv:
            s_correct += 1
    print("Overall Acc: {}".format(str(np.round(float(correct) / float(total), 3))))
    print("Start Acc: {}".format(str(np.round(float(correct_start) / float(total_start), 3))))

    return np.round(float(correct_start) / float(total_start), 3)


evaluate_tracie_style()