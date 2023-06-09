Data
===
To facilitate relative label evaluation, we extend each instance in Today to four lines with the following structure:

Original instance:

    event:(gold relation) story:([Additional sentence for gold relation]+Context) explanation:(explanation for gold relation)

We present the dataset in modified final instance version with and without explanation:  

    Line1: event:(gold relation) story:(Context) \n
    Line2: event:(opposite relation) story:(Context) \n
    Line3: event:(gold relation) story:([Additional sentence for gold relation]+Context) explanation:(human annotated explanation for gold relation) \n
    Line4: event:(opposite relation) story:([Additional sentence for gold relation]+Context) explanation:(human annotated explanation for gold relation) \n

Inference
===
The model should output a probability for entailment for each line and please refer to ../code/evaluator_today.py for dataset evaluation. For each instance:

    Line1: origin_correct_prob
    Line2: origin_wrong_prob
    Line3: modify_correct_correct_prob
    Line4: modify_correct_wrong_prob

If the probability change in the right direction is larger than the wrong direction, i.e, if (modify_correct_correct_prob - origin_correct_prob) > (modify_correct_wrong_prob - origin_wrong_prob), we denote this as a correct instance. 
