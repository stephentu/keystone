import itertools as it
from random import randint

#gammas = [8e-6, 4e-6, 1e-6] # [8e-4, 4e-4, 2e-4, 1e-4, 8e-5, 4e-5, 1e-5]
#lambdas = [1e-1, 1e-2, 1e-3, 1e-4, 1e1]
#numModels = [32, 64, 128]
#
#for gamma, numModel in it.product(gammas, numModels):
#    seed = randint(0, 0xffffffff)
#    print "./bin/run-cifar-augmented.sh {0} {1} {2} {3}".format(gamma, ",".join(map(str, lambdas)), numModel, seed)
#    print "/root/spark/sbin/stop-all.sh"
#    print "sleep 5"
#    print "/root/spark/sbin/start-all.sh"
#    print "sleep 5"

solver = "dcsvm"
numModel = 256
gammas= [0.00308025, 8.00E-04, 4.00E-04, 2.00E-04]
lam = "0.1,1.00E-02,1.00E-03,1.00E-04,1.00E-05"

pairs = [(0.00308025, 1e-3),                   (0.00308025, 1e-5),
         (4e-4, 1e-3),                         (4e-4, 1e-5)]

for (gamma, lamb) in pairs:
    seed = randint(0, 0xffffffff)
    print "./bin/run-timit.sh {0} {1} {2} {3} {4}".format(solver, gamma, lamb, numModel, seed)
    print "/root/spark/sbin/stop-all.sh"
    print "sleep 5"
    print "/root/spark/sbin/start-all.sh"
    print "sleep 5"
