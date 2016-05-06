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

numModel = 16
pairs = [(1e-4, 1e-3), (1e-4, 1e-4), (1e-4, 1e-5), (1e-4, 1e-6), (1e-4, 1e-7),
         (1e-5, 1e-3),                                           (1e-5, 1e-7),
         (1e-6, 1e-3), (1e-6, 1e-4), (1e-6, 1e-5), (1e-6, 1e-6), (1e-6, 1e-7)]

for gamma, lam in pairs:
    seed = randint(0, 0xffffffff)
    print "./bin/run-cifar-augmented.sh {0} {1} {2} {3}".format(gamma, lam, numModel, seed)
    print "/root/spark/sbin/stop-all.sh"
    print "sleep 5"
    print "/root/spark/sbin/start-all.sh"
    print "sleep 5"
