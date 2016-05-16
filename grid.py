import itertools as it
from random import randint

#gammas = [4e-5] #[8e-5, 4e-5] # [8e-4, 4e-4, 2e-4, 1e-4, 8e-5, 4e-5, 1e-5]
#lambdas = [1e-3]
#numModels = [32] #[32, 64, 128]

#for gamma, numModel in it.product(gammas, numModels):
#    seed = randint(0, 0xffffffff)
#    print "./bin/run-cifar-augmented.sh {0} {1} {2} {3}".format(gamma, ",".join(map(str, lambdas)), numModel, seed)
#    print "/root/spark/sbin/stop-all.sh"
#    print "sleep 5"
#    print "/root/spark/sbin/start-all.sh"
#    print "sleep 5"

#solver = "dcsvm"
#numModel = 256
#gammas= [0.00308025, 8.00E-04, 4.00E-04, 2.00E-04]
#lam = "0.1,1.00E-02,1.00E-03,1.00E-04,1.00E-05"

TIMIT_DCSVM_PAIRS=[(384,      8.00E-04,       1E-04),
(512,     8.00E-04,       1E-04),
(640,      8.00E-04,       1E-04)]
CIFAR_DCYUCHEN_PAIRS=[(64,     1.00E-05,       1.00E-04),
(96  , 1.00E-05,       1.00E-04),
(128  , 1.00E-05,       1.00E-04),
(256 , 1.00E-05,       1.00E-04),
(512  , 1.00E-05,       1.00E-04)]
TIMIT_DCYUCHEN_PAIRS=[(384,      8.00E-04,       1.00E-04),
(512,      8.00E-04,       1.00E-04),
(640,      8.00E-04,       1.00E-04)]
YELP_DCYUCHEN_PAIRS=[(128, 1.00E+01),
(256, 1.00E+01)]

for (numModel, gamma, lamb) in TIMIT_DCSVM_PAIRS:
    seed = randint(0, 0xffffffff)
    solver="dcsvm"
    print "./bin/run-timit-dc.sh {0} {1} {2} {3} {4}".format(solver, gamma, lamb, numModel, seed)
    print "/root/spark/sbin/stop-all.sh"
    print "sleep 5"
    print "/root/spark/sbin/start-all.sh"
    print "sleep 5"
for (numModel, gamma, lamb) in CIFAR_DCYUCHEN_PAIRS:
    seed = randint(0, 0xffffffff)
    solver="dcyuchen"
    print "./bin/run-cifar-augmented.sh {0} {1} {2} {3} {4}".format(solver, gamma, lamb, numModel, seed)
    print "/root/spark/sbin/stop-all.sh"
    print "sleep 5"
    print "/root/spark/sbin/start-all.sh"
    print "sleep 5"
for (numModel, gamma, lamb) in TIMIT_DCYUCHEN_PAIRS:
    seed = randint(0, 0xffffffff)
    solver="dcyuchen"
    print "./bin/run-timit-dc.sh {0} {1} {2} {3} {4}".format(solver, gamma, lamb, numModel, seed)
    print "/root/spark/sbin/stop-all.sh"
    print "sleep 5"
    print "/root/spark/sbin/start-all.sh"
    print "sleep 5"
for (numModel, lamb) in YELP_DCYUCHEN_PAIRS:
    seed = randint(0, 0xffffffff)
    solver="dcyuchen"
    print "./bin/run-yelp-dc.sh {0} {1} {2} {3} 1024".format(solver, lamb, numModel, seed)
    print "/root/spark/sbin/stop-all.sh"
    print "sleep 5"
    print "/root/spark/sbin/start-all.sh"
    print "sleep 5"
