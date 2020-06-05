import os
import MFCC
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk
import argparse

CLASS_LABELS = {"vietnam", "khach", "khong", "toi", "ban"}

def runHMM(file_path):
    models = {}
    for label in CLASS_LABELS:
        with open(os.path.join("Models", label + ".pkl"), "rb") as file: models[label] = pk.load(file)

    with open("Models/kmeans.pkl", "rb") as file: kmeans = pk.load(file)

    sound_mfcc = MFCC.get_mfcc(file_path)
    sound_mfcc = kmeans.predict(sound_mfcc).reshape(-1,1)

    evals = {cname : model.score(sound_mfcc, [len(sound_mfcc)]) for cname, model in models.items()}
    conclusion = max(evals.keys(), key=(lambda k: evals[k]))

    return evals, conclusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="Image input", required=True)
    args = parser.parse_args()

    evals, conclusion = runHMM(args.test)
    print(evals)
    print("Conclusion: " + conclusion)