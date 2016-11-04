from os import listdir
from numpy import array, multiply, divide, add, subtract, abs
import matplotlib.pyplot as plt

def myfun(directory):
    itera=None
    error=None
    res=list()
    for filename in listdir(directory):
        if not filename.endswith(".txt"): continue
        with open("%s/%s"%(directory,filename)) as f:
            itera=None
            error=None
            for line in f:
                line=line.strip()
                if not line.startswith("Current iteration: ") and not line.startswith("Error: "): continue
                if line.startswith("Current iteration: "):
                    line=line.partition(": ")
                    itera=float(line[2])
                    continue
                if line.startswith("Error: "):
                    line=line.partition(": ")
                    error=float(line[2])
                    continue
                if itera and error: break
        if not itera and not error: raise Exception("Failed to find data in file")
        if (itera and not error) or (error and not itera): raise Exception("Only one, not both")
        if itera and error:
            res.append((itera, error))

    print len(res)
    res.sort()
    X, Y = zip(*res)
    return X,Y

X1,Y1=myfun("outs/tm02/tn")
X2,Y2=myfun("outs/tm02/tr")
X2=multiply(X2,5)
Y2=multiply(Y2,5)
X1s,Y1s=myfun("outs/tm02/tns")
X2s,Y2s=myfun("outs/tm02/trs")
X2s=multiply(X2s,5)
Y2s=multiply(Y2s,5)
plt.plot(X1,Y1, "g", X2,Y2, "r", X1,Y1, "go", X2,Y2, "ro", X1s,Y1s, "g--", X2s,Y2s, "r--", X1s,Y1s, "gs", X2s,Y2s, "rs")
######################################################################################################################
X1,Y1=myfun("outs/tm02/cr")
X1s,Y1s=myfun("outs/tm02/crs")
X1=multiply(X1,5)
Y1=multiply(Y1,5)
X1s=multiply(X1s,5)
Y1s=multiply(Y1s,5)
plt.plot(X1,Y1, "b", X1,Y1, "bo")
plt.plot(X1s,Y1s, "b-", X1s,Y1s, "bs")
######################################################################################################################
# X1,Y1=myfun("outs/tm02/tn")
# X2,Y2=myfun("outs/tm02/tr")
# Y2=multiply(Y2,5)
# X=X1
# Y=subtract(Y1,Y2)
# Y=abs(Y)
# plt.plot(X,Y, "k", X,Y, "ko")



# plt.axis([0, 75000, .4, .6])
# plt.axis([0, 70000, 0, 1])
plt.grid(True)
# plt.savefig("a.png")#, dpi=80
plt.show()