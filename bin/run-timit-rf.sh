#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=TimitRandomFeatLBFGS

TRAIN_FEATURES="s3n://timit-data/timit-train-features.csv"
TRAIN_LABELS="/root/stephentu-keystone/timit-train-labels.sparse"
TEST_FEATURES="s3n://timit-data/timit-test-features.csv"
TEST_LABELS="/root/stephentu-keystone/timit-test-labels.sparse"

NUM_PARTITIONS=512
LAMBDAS=1e-6
NUM_ITERS=15
NUM_COSINES=202752
BLOCK_SIZE=4096
GAMMA=0.0555
SEED=213769821231
SOLVER="lbfgs"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

export EXECUTOR_OMP_NUM_THREADS=1

OMP_NUM_THREADS=1 KEYSTONE_MEM=180g ./bin/run-pipeline.sh \
  pipelines.speech.timit.$CLASS \
  --trainDataLocation $TRAIN_FEATURES \
  --trainLabelsLocation $TRAIN_LABELS \
  --testDataLocation $TEST_FEATURES \
  --testLabelsLocation $TEST_LABELS \
  --trainParts $NUM_PARTITIONS \
  --lambda $LAMBDAS \
  --cosineGamma $GAMMA \
  --blockSize $BLOCK_SIZE \
  --numCosineFeatures $NUM_COSINES \
  --numIters $NUM_ITERS \
  --solver $SOLVER \
  --seed $SEED 2>&1 | tee /root/logs/timit-512-solver-$SOLVER-gamma-$GAMMA-lambda-$LAMBDAS-numIter-$NUM_ITERS-numCosine-$NUM_COSINES-logs-"$LOG_SUFFIX".log

