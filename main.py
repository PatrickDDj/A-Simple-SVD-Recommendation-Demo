import os
import re

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def init_P_Q_set(K = 5):

    df = pd.read_table("ml-1m/ratings.dat", encoding='utf-8', header=None, sep='::', engine='python')
    df.columns = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

    # 打乱顺序
    df = df.sample(frac=1)

    n_User = df['UserID'].max()
    n_Movie = df['MovieID'].max()

    P = np.random.randint(1, 3, (n_User+1, K)) / (K**0.5)
    Q = np.random.randint(1, 3, (K, n_Movie+1)) / (K**0.5)

    grouped = df.groupby('UserID')

    train_set = {}
    val_set = {}
    test_set = {}

    for u, group in grouped:
        train_set[u] = {}
        val_set[u] = {}
        test_set[u] = {}

        i=0
        for v, w in zip(group['MovieID'], group['Rating']):
            if i<0.8*len(group):
                train_set[u][v] = w
            elif i<0.9*len(group):
                val_set[u][v] = w
            else:
                test_set[u][v] = w
            i += 1
    return P, Q, train_set, val_set, test_set


def optimize(P, Q, learning_rate, train_set):
    Pn = np.array(P, copy=True)
    Qn = np.array(Q, copy=True)

    U, I, K = len(P), len(Q[0]), len(P[0])

    cost = 0
    n = 0

    e_ui_map = {}

    for u in range(1, U):

        e_ui_map[u] = {}

        for i in range(1, I):
            if u in train_set.keys() and i in train_set[u].keys():
                e_ui = train_set[u][i] - np.dot(P[u, :], Q[:, i])

                e_ui_map[u][i] = e_ui

                cost += e_ui**2
                n += 1
                Pn[u, :] = Pn[u, :] + learning_rate * e_ui * Q[:, i]
                Qn[:, i] = Qn[:, i] + learning_rate * e_ui * P[u, :]

    return Pn, Qn, (cost / n ) ** 0.5


def train(P, Q, train_set, iterations=50, learning_rate=0.1):
    last_cost = 0
    for i in range(iterations):
        print("iteration %i" % i)
        P, Q, cost = optimize(P, Q, learning_rate, train_set)
        if abs(last_cost-cost) < 0.005:
            break
        last_cost = cost
        print("Cost after iteration %i: %f" % (i, cost))
    return P, Q


def evaluate(P, Q, val_set, Set="Val_Set"):
    R = np.dot(P, Q)
    cost = 0
    n = 0
    for u in val_set.keys():
        for i in val_set[u].keys():
            cost += (abs(val_set[u][i]-R[u][i]))**2
            n += 1
    cost /= n
    cost = cost ** 0.5
    print("Error on %s : %f" % (Set, cost))
    return cost


def save_P_Q(P, Q, K, learning_rate, cost):
    P_txt = "PQ/P(K=%d,learning_rate=%f, cost=%.6f).txt" % (K, learning_rate, cost)
    Q_txt = "PQ/Q(K=%d,learning_rate=%f, cost=%.6f).txt" % (K, learning_rate, cost)
    np.savetxt(P_txt, P)
    np.savetxt(Q_txt, Q)


def train_P_Q(K, learning_rate):
    P, Q, train_set, val_set, test_set = init_P_Q_set(K=K)
    P, Q = train(P, Q, train_set, iterations=50, learning_rate=learning_rate)
    cost = evaluate(P, Q, val_set)
    save_P_Q(P, Q, K=K, learning_rate=learning_rate, cost=cost)


def select_best_P_Q():
    for learning_rate in [0.00018, 0.00020, 0.00022]:
        for K in range(2, 6):
            train_P_Q(K=K, learning_rate=learning_rate)


def get_P_Q(K, learning_rate):
    P_prefix = "P(K=%d,learning_rate=%f" % (K, learning_rate)
    P_txt = "PQ/"+[i for i in os.listdir('./PQ') if i.startswith(P_prefix)][0]
    Q_prefix = "Q(K=%d,learning_rate=%f" % (K, learning_rate)
    Q_txt = "PQ/"+[i for i in os.listdir('./PQ') if i.startswith(Q_prefix)][0]
    P = np.loadtxt(P_txt)
    Q = np.loadtxt(Q_txt)

    return P, Q


def get_T(test_set):
    T = []
    n = 1
    for u in test_set.keys():
        for i in test_set[u].keys():
            n += 1
            if test_set[u][i] >= 4:
                T.append((u, i))

    return T


def get_R_PQ(P, Q, test_set):
    R = []
    for u in test_set.keys():
        for i in test_set[u].keys():
            if np.dot(P[u, :], Q[:, i]) >= 3.5:
                R.append((u,i))

    return R


def cal_Precision_Recall(R, T):
    RT = [i for i in R if i in T]
    # print(len(R))
    Precision = len(RT) / len(R)
    Recall = len(RT) / len(T)

    return Precision, Recall


def get_movie_rating(train_set):
    movie_rating = {}
    for u in train_set.keys():
        for i in train_set[u].keys():
            if i not in movie_rating.keys():
                movie_rating[i] = []
            movie_rating[i].append(train_set[u][i])
    return movie_rating


def get_R_math(movie_rating, test_set, f=np.average):
    movie_rating = {i: f(j) for i, j in movie_rating.items()}

    R = []
    for u in test_set.keys():
        for i in test_set[u].keys():
            if i in movie_rating.keys() and movie_rating[i] >= 3.5:
                R.append((u,i))
    return R


def plot_cost():
    fig = plt.figure()
    ax = Axes3D(fig)

    X = []
    Y = []
    Z = []

    for i in os.listdir("./PQ"):
        r = re.findall(r"[-+]?\d*\.\d+|\d+", i)
        if len(r) == 3:
            X.append(int(r[0]))
            Y.append(float(r[1]))
            Z.append(float(r[2]))

    ax = plt.subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, color='grey')

    ax.set_zlabel('cost')
    ax.set_ylabel('learning_rate')
    ax.set_xlabel('K')
    plt.show()


if __name__ == '__main__':

    # select_best_P_Q()

    K = 5
    learning_rate = 0.000220

    _, _, train_set, val_set, test_set = init_P_Q_set(K=K)
    T = get_T(test_set)

    P, Q = get_P_Q(K=K, learning_rate=learning_rate)

    evaluate(P, Q, test_set, Set="Test Set")

    plot_cost()

    R_PQ = get_R_PQ(P, Q, test_set)
    print("(1) P-Q Recommendation : Precision(%f), Recall(%f)" %cal_Precision_Recall(R_PQ, T))

    movie_rating = get_movie_rating(train_set)

    R_average = get_R_math(movie_rating, test_set, np.average)
    print("(2) Movie Average Rating Recommendation : Precision(%f), Recall(%f)" % cal_Precision_Recall(R_average, T))

    R_median = get_R_math(movie_rating, test_set, np.median)
    print("(3) Movie Median Rating Recommendation : Precision(%f), Recall(%f)" % cal_Precision_Recall(R_median, T))



