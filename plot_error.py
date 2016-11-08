from os import listdir
from numpy import array, multiply, divide#, add, subtract, abs
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

# X1,Y1=myfun("outs/tm02/tn")
# X2,Y2=myfun("outs/tm02/tr")
# Y1=divide(Y1, 46)
# Y2=divide(Y2, 46)
# X2=multiply(X2,5)
# Y2=multiply(Y2,5)
# X1s,Y1s=myfun("outs/tm02/tns")
# X2s,Y2s=myfun("outs/tm02/trs")
# Y1s=divide(Y1s,46)
# Y2s=divide(Y2s,46)
# X2s=multiply(X2s,5)
# Y2s=multiply(Y2s,5)
# plt.plot(X1,Y1, "g", X2,Y2, "r", X1,Y1, "go", X2,Y2, "ro", X1s,Y1s, "g--", X2s,Y2s, "r--", X1s,Y1s, "gs", X2s,Y2s, "rs")

# X2m,Y2m=myfun("outs/tm02/trm")
# Y2m=divide(Y2m,46)
# X2m=multiply(X2m,5*2)
# Y2m=multiply(Y2m,5)
# plt.plot(X2m,Y2m, "r", X2m,Y2m, "r^")
######################################################################################################################
# X1,Y1=myfun("outs/tm02/cr")
# X1s,Y1s=myfun("outs/tm02/crs")
# Y1=divide(Y1,72)
# Y1s=divide(Y1s,72)
# X1=multiply(X1,5*2)
# Y1=multiply(Y1,5)
# X1s=multiply(X1s,5*2)
# Y1s=multiply(Y1s,5)
# plt.plot(X1,Y1, "b", X1,Y1, "bo")
# plt.plot(X1s,Y1s, "b--", X1s,Y1s, "bs")

# X2m,Y2m=myfun("outs/tm02/crb")
# Y2m=divide(Y2m,46)
# X2m=multiply(X2m,5*2*2)
# Y2m=multiply(Y2m,5)
# plt.plot(X2m,Y2m, "b", X2m,Y2m, "b^")

# X2b,Y2b=myfun("outs/tm02/crm")
# Y2b=divide(Y2b,46)
# X2b=multiply(X2b,5*2*2)
# Y2b=multiply(Y2b,5)
# plt.plot(X2b,Y2b, "b", X2b,Y2b, "bD")
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
# plt.grid(True)
# plt.savefig("a.png")#, dpi=80
# plt.show()




# Text NAG
X,Y=myfun("outs/tm02/tn")
Y=divide(Y, 46)
plt.plot(X,Y, "g", X,Y, "go")

# Text RMSProp
X,Y=myfun("outs/tm02/tr")
Y=divide(Y, 46)
X=multiply(X,5)
Y=multiply(Y,5)
plt.plot(X,Y, "r", X,Y, "ro")

# Text NAG with larger neural network
X,Y=myfun("outs/tm02/tnl")
Y=divide(Y, 46)
plt.plot(X[1:],Y[1:], "g--", X[1:],Y[1:], "g*")

# Text RMSProp with larger neural network
X,Y=myfun("outs/tm02/trl")
Y=divide(Y, 46)
X=multiply(X,5)
Y=multiply(Y,5)
plt.plot(X,Y, "r--", X,Y, "r*")

# Text NAG higher learning rate
# X,Y=myfun("outs/tm02/tns")
# Y=divide(Y, 46)
# plt.plot(X,Y, "g", X,Y, "gs")

# Text RMSProp higher learning rate
# X,Y=myfun("outs/tm02/trs")
# Y=divide(Y, 46)
# X=multiply(X,5)
# Y=multiply(Y,5)
# plt.plot(X,Y, "r", X,Y, "rs")

# Text RMSProp more data
X,Y=myfun("outs/tm02/trm")
Y=divide(Y, 46)
X=multiply(X,5*2)
Y=multiply(Y,5)
plt.plot(X,Y, "r", X,Y, "r^")

# CSS RMSProp
X,Y=myfun("outs/tm02/cr")
Y=divide(Y, 72)
X=multiply(X,5*2)
Y=multiply(Y,5)
plt.plot(X,Y, "b", X,Y, "bo")

# CSS RMSProp higher learning rate
# X,Y=myfun("outs/tm02/crs")
# Y=divide(Y, 72)
# X=multiply(X,5*2)
# Y=multiply(Y,5)
# plt.plot(X,Y, "b", X,Y, "bs")

# CSS RMSProp more data
# X,Y=myfun("outs/tm02/crm")
# Y=divide(Y, 72)
# X=multiply(X,5*2*2)
# Y=multiply(Y,5)
# plt.plot(X,Y, "b", X,Y, "b^")

# CSS RMSProp larger batch size
# X,Y=myfun("outs/tm02/crb")
# Y=divide(Y, 72)
# X=multiply(X,5*2*2)
# Y=multiply(Y,5*2)
# plt.plot(X,Y, "b", X,Y, "bD")

plt.grid(True)
# plt.savefig("a.png")#, dpi=80
plt.show()