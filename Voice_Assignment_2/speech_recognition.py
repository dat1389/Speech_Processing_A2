import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import MFCC
import math
import random as rand
from sklearn.cluster import KMeans
import hmmlearn.hmm
import numpy as np
import pickle as pk

CLASS_LABELS = {"vietnam", "khach", "khong", "toi", "ban"}
# Directory name of sound files must match labels

TEST_SIZE = 20
# Number of sound files reserved for testing (<100)

def get_label_data(directory):
    files = os.listdir(directory)
    # Randomized Test and Train set
    test_files = rand.sample(files, TEST_SIZE)

    train_mfcc = []
    test_mfcc = []

    for f in test_files:
        if f.endswith("wav"):
            test_mfcc.append(MFCC.get_mfcc(os.path.join(directory, f)))
            files.pop(files.index(f))
    
    train_mfcc = [MFCC.get_mfcc(os.path.join(directory, f)) for f in files if f.endswith("wav")]

    return train_mfcc, test_mfcc

def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    return kmeans  

if __name__ == "__main__":
    train_dataset = {}
    test_dataset = {}
    X = {}

    # Loading MFCC
    for label in CLASS_LABELS:
        train_dataset[label], test_dataset[label] = get_label_data(os.path.join("SoundFiles", label))
        print(label)
        print("Loaded test: ", len(test_dataset[label]), " files")
        print("Loaded train: ", len(train_dataset[label]), " files")

    # Get all vectors in the datasets
    all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in train_dataset.items()], axis=0)
    #print("Vectors", all_vectors.shape)a
    # Run K-Means algorithm to get clusters
    kmeans = clustering(all_vectors)
    #print("Centers", kmeans.cluster_centers_.shape)

    # KMeans
    for label in CLASS_LABELS:
        train_dataset[label] = list([kmeans.predict(v).reshape(-1,1) for v in train_dataset[label]])
        test_dataset[label] = list([kmeans.predict(v).reshape(-1,1) for v in test_dataset[label]])
        X[label] = np.concatenate(train_dataset[label])

    models = {}
    # HMM Models
    # vietnam
    vietnam_model = hmmlearn.hmm.MultinomialHMM(
        n_components=12, random_state=0, n_iter=1000, verbose=True,params='te',init_params='e'
    )
    vietnam_model.startprob_ = np.array([0.5,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    vietnam_model.transmat_ = np.array([
        [0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    ])
    length = list([len(x) for x in train_dataset["vietnam"]])
    print("Training class", label)
    print(X["vietnam"].shape, length, len(length))
    vietnam_model.fit(X["vietnam"], lengths=length)
    models["vietnam"] = vietnam_model

     # Khach
    khach_model = hmmlearn.hmm.MultinomialHMM(
        n_components=9, random_state=0, n_iter=1000, verbose=True,params='te',init_params='e'
    )
    khach_model.startprob_ = np.array([0.5,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0])
    khach_model.transmat_ = np.array([
        [0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    ])
    length = list([len(x) for x in train_dataset["khach"]])
    print("Training class", label)
    print(X["khach"].shape, length, len(length))
    khach_model.fit(X["khach"], lengths=length)
    models["khach"] = khach_model

    # Toi
    toi_model = hmmlearn.hmm.MultinomialHMM(
        n_components=9, random_state=0, n_iter=1000, verbose=True,params='te',init_params='e'
    )
    toi_model.startprob_ = np.array([0.5,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0])
    toi_model.transmat_ = np.array([
        [0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    ])
    length = list([len(x) for x in train_dataset["toi"]])
    print("Training class", label)
    print(X["toi"].shape, length, len(length))
    toi_model.fit(X["toi"], lengths=length)
    models["toi"] = toi_model

# ban
    ban_model = hmmlearn.hmm.MultinomialHMM(
        n_components=9, random_state=0, n_iter=1000, verbose=True,params='te',init_params='e'
    )
    ban_model.startprob_ = np.array([0.5,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0])
    ban_model.transmat_ = np.array([
        [0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    ])
    length = list([len(x) for x in train_dataset["ban"]])
    print("Training class", label)
    print(X["ban"].shape, length, len(length))
    ban_model.fit(X["ban"], lengths=length)
    models["ban"] = ban_model

    # Khong
    khong_model = hmmlearn.hmm.MultinomialHMM(
        n_components=9, random_state=0, n_iter=1000, verbose=True,params='te',init_params='e'
    )
    khong_model.startprob_ = np.array([0.5,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0])
    khong_model.transmat_ = np.array([
        [0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.1,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.7,0.3,0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.5],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    ])
    length = list([len(x) for x in train_dataset["khong"]])
    print("Training class", label)
    print(X["khong"].shape, length, len(length))
    khong_model.fit(X["khong"], lengths=length)
    models["khong"] = khong_model

    
        
    print("Training done")

    print("Testing (Higher is better)")
    model_acc_train = {}
    model_acc_test = {}
    for true_cname in CLASS_LABELS:
        hits = 0
        for t in train_dataset[true_cname]:
            evals = {cname : model.score(t, [len(t)]) for cname, model in models.items()}
            print(true_cname, evals)
            if max(evals.keys(), key=(lambda k: evals[k])) == true_cname:
                print("Hit")
                hits += 1
            else:
                print("Miss")
        model_acc_train[true_cname] = hits

    for true_cname in CLASS_LABELS:
        hits = 0
        for t in test_dataset[true_cname]:
            evals = {cname : model.score(t, [len(t)]) for cname, model in models.items()}
            print(true_cname, evals)
            if max(evals.keys(), key=(lambda k: evals[k])) == true_cname:
                print("Hit")
                hits += 1
            else:
                print("Miss")
        model_acc_test[true_cname] = hits

    print(model_acc_train)
    print(model_acc_test)

    print("Exporting models")
    for label in CLASS_LABELS:
        with open(os.path.join("Models", label + ".pkl"), "wb") as file: pk.dump(models[label], file)
    with open("Models/kmeans.pkl", "wb") as file: pk.dump(kmeans, file)