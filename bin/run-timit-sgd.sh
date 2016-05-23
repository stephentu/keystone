#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=TimitRandomFeatLBFGS

#CIFAR_TRAIN_DIR="s3n://cifar-augmented/cifar_train_featurized_augmented_512_flip"
#CIFAR_TEST_DIR="s3n://cifar-augmented/cifar_test_featurized_augmented_512_flip"

TRAIN_FEATURES="s3n://timit-data/timit-train-features.csv"
TRAIN_LABELS="/root/stephentu-keystone/timit-train-labels.sparse"
TEST_FEATURES="s3n://timit-data/timit-test-features.csv"
TEST_LABELS="/root/stephentu-keystone/timit-test-labels.sparse"

NUM_PARTITIONS=1024
LAMBDAS=1e-6
NUM_ITERS=500
NUM_COSINES=202752
BLOCK_SIZE=6144
GAMMA=0.0555
SEED=123123
SOLVER="sgd"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`
#MAX ROW NORM IS 311
# 1e-3 diverged
# 1e-4
STEP_SIZE=3.75e-4
SGD_DAMPEN=0.95
MINI_BATCH_FRACTION=0.1
COCOA_BETA=1.0
COCOA_LOCAL_ITERS=1.0

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
  --sgdDampen $SGD_DAMPEN \
  --stepSize $STEP_SIZE \
  --cocoaBeta $COCOA_BETA \
  --cocoaLocalItersFraction $COCOA_LOCAL_ITERS \
  --miniBatchFraction $MINI_BATCH_FRACTION \
  --seed $SEED 2>&1 | tee /root/logs/timit-"$NUM_PARTITIONS"-solver-$SOLVER-gamma-$GAMMA-lambda-$LAMBDAS-numIter-$NUM_ITERS-numCosine-$NUM_COSINES-step-$STEP_SIZE-mini-$MINI_BATCH_FRACTION-dampen-$SGD_DAMPEN-logs-"$LOG_SUFFIX".log

