import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import MFCC
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk

CLASS_LABELS = {"vietnam", "khach", "khong", "toi", "ban"}
# Directory name of sound files must match labels

def get_label_data(label):
    files = os.listdir(label)
    test_mfcc = [MFCC.get_mfcc(os.path.join(label, f)) for f in files if f.endswith("wav")]
    return test_mfcc

if __name__ == "__main__":
    test_dataset = {}
    for label in CLASS_LABELS:
        test_dataset[label] = get_label_data(os.path.join("SoundFiles", label))

    models = {}
    for label in CLASS_LABELS:
        with open(os.path.join("Models", label + ".pkl"), "rb") as file: models[label] = pk.load(file)
    with open("Models/kmeans.pkl", "rb") as file: kmeans = pk.load(file)
    
    for labels in CLASS_LABELS:
        test_dataset[labels] = list([kmeans.predict(v).reshape(-1,1) for v in test_dataset[labels]])

    print("Testing (Higher is better)")
    model_acc_test = {}

    for true_cname in CLASS_LABELS:
        hits = 0
        for O in test_dataset[true_cname]:
            evals = {cname : model.score(O, [len(O)]) for cname, model in models.items()}
            print(true_cname, evals)
            if max(evals.keys(), key=(lambda k: evals[k])) == true_cname:
                print("Hit")
                hits += 1
            else:
                print("Miss")
        model_acc_test[true_cname] = hits

    print(model_acc_test)