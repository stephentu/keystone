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

NUM_PARTITIONS=1024
LAMBDAS=1e5
NUM_ITERS=100
NUM_COSINES=202752
BLOCK_SIZE=6144
GAMMA=0.0555
SEED=213769821231
SOLVER="cocoa"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`
STEP_SIZE=0.1
MINI_BATCH_FRACTION=1.0
COCOA_BETA=1.0
COCOA_LOCAL_ITERS=0.1
NORM_ROWS="false"

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
  --stepSize $STEP_SIZE \
  --cocoaBeta $COCOA_BETA \
  --cocoaLocalItersFraction $COCOA_LOCAL_ITERS \
  --miniBatchFraction $MINI_BATCH_FRACTION \
  --seed $SEED 2>&1 | tee /root/logs/timit-"$NUM_PARTITIONS"-solver-$SOLVER-gamma-$GAMMA-lambda-$LAMBDAS-numIter-$NUM_ITERS-numCosine-$NUM_COSINES-cocoaLocal-"$COCOA_LOCAL_ITERS"-logs-"$LOG_SUFFIX".log

