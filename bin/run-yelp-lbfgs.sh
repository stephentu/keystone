#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=YelpHashLBFGS

TRAIN_FEATURES="s3n://yelp-rating-reviews/yelp_academic_dataset_review-train-tokens.txt"
#TRAIN_LABELS="/root/stephentu-keystone/yelp_academic_dataset_review-train-labels.txt"
TRAIN_LABELS="s3n://yelp-rating-reviews/yelp_academic_dataset_review-train-labels.txt"
TEST_FEATURES="s3n://yelp-rating-reviews/yelp_academic_dataset_review-test-tokens.txt"
#TEST_LABELS="/root/stephentu-keystone/yelp_academic_dataset_review-test-labels.txt"
TEST_LABELS="s3n://yelp-rating-reviews/yelp_academic_dataset_review-test-labels.txt"

SOLVER=lbfgs
LAMBDA=7.9655e-8
NUM_HASH_FEATURES=122880
BLOCK_SIZE=6144
NUM_ITERS=200
SEED=1213279863
NUM_PARTITIONS=512
TEST_PARTITIONS=512
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`
STEP_SIZE=0.1
MINI_BATCH_FRACTION=1.0

export EXECUTOR_OMP_NUM_THREADS=1

OMP_NUM_THREADS=1 KEYSTONE_MEM=180g ./bin/run-pipeline.sh \
  pipelines.text.$CLASS \
  --trainDataLocation $TRAIN_FEATURES \
  --trainLabelsLocation $TRAIN_LABELS \
  --testDataLocation $TEST_FEATURES \
  --testLabelsLocation $TEST_LABELS \
  --trainParts $NUM_PARTITIONS \
  --testParts $TEST_PARTITIONS \
  --numHashFeatures $NUM_HASH_FEATURES \
  --numIters $NUM_ITERS \
  --blockSize $BLOCK_SIZE \
  --lambda $LAMBDA \
  --solver $SOLVER \
  --stepSize $STEP_SIZE \
  --miniBatchFraction $MINI_BATCH_FRACTION \
  --seed $SEED 2>&1 | tee /root/logs/yelp-solver-$SOLVER-lambda-$LAMBDA-numHash-$NUM_HASH_FEATURES-numIters-$NUM_ITERS-logs-"$LOG_SUFFIX".log

