import sqlite3
import math
import numpy as np
from scipy import stats

class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        value = float(value)
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None

class RFunc:
    def __init__(self):
        self.k = 0
        self.v1 = np.array([])
        self.v2 = np.array([])

    def step(self, v1, v2):
        if v1 is None or v2 is None:
            return

        self.v1 = np.append(self.v1, [v1], axis=0)
        self.v2 = np.append(self.v2, [v2], axis=0)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        try:
            rv = stats.pearsonr(self.v1, self.v2)
        except ex:
            print(ex)
        print(rv)
        return rv[0]

with sqlite3.connect('output-bigfeatures-bacherlorvm-22.11.2018.sqlite') as con:

    con.create_aggregate("stdev", 1, StdevFunc)
    con.create_aggregate("pr", 2, RFunc)

    cur = con.cursor()

    def stdev(col):
        cur.execute("select avg(" + col + ") from results")
        avg = cur.fetchone()[0]
        print("avg %s: %f" % (col, avg))
        cur.execute("select stdev(" + col + ") from results")
        std = cur.fetchone()[0]
        print("stdev %s: %f" % (col, std))
        return avg, std

    def pr(col, col2):
        cur.execute("select pr(" + col + ", " + col2 + ") from results")
        r = cur.fetchone()[0]
        print("r %s <-> %s: %f" % (col, col2, r))
