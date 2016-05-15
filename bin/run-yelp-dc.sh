#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=YelpDCSolver

TRAIN_FEATURES="s3n://yelp-rating-reviews/yelp_academic_dataset_review-train-tokens.txt"
#TRAIN_LABELS="/root/stephentu-keystone/yelp_academic_dataset_review-train-labels.txt"
TRAIN_LABELS="s3n://yelp-rating-reviews/yelp_academic_dataset_review-train-labels.txt"
TEST_FEATURES="s3n://yelp-rating-reviews/yelp_academic_dataset_review-test-tokens.txt"
#TEST_LABELS="/root/stephentu-keystone/yelp_academic_dataset_review-test-labels.txt"
TEST_LABELS="s3n://yelp-rating-reviews/yelp_academic_dataset_review-test-labels.txt"

SOLVER=$1 
LAMBDAS=$2
NUM_PARTITIONS=$3
SEED=$4
TEST_PARTITIONS=$5
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

export EXECUTOR_OMP_NUM_THREADS=8

OMP_NUM_THREADS=8 KEYSTONE_MEM=180g ./bin/run-pipeline.sh \
  pipelines.text.$CLASS \
  --trainDataLocation $TRAIN_FEATURES \
  --trainLabelsLocation $TRAIN_LABELS \
  --testDataLocation $TEST_FEATURES \
  --testLabelsLocation $TEST_LABELS \
  --trainParts $NUM_PARTITIONS \
  --testParts $TEST_PARTITIONS \
  --numPartitions $NUM_PARTITIONS \
  --numModels $NUM_PARTITIONS \
  --lambdas $LAMBDAS \
  --solver $SOLVER \
  --seed $SEED 2>&1 | tee /root/logs/yelp-solver-$SOLVER-lambdas-$LAMBDAS-nummodels-$NUM_PARTITIONS-logs-"$LOG_SUFFIX".log

