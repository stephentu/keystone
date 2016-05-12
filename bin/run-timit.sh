#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=TimitDCSolver

TRAIN_FEATURES=s3n://timit-data/timit-train-cosine-gaussian-features
TRAIN_LABELS=/root/stephentu-keystone/timit-train-labels.sparse
TEST_FEATURES=s3n://timit-data/timit-test-cosine-gaussian-features
TEST_LABELS=/root/stephentu-keystone/timit-test-labels.sparse

USE_YUCHEN=$6
NUM_MODELS=$4
NUM_PARTITIONS=$4
LAMBDAS=$3
GAMMA=$2
SEED=$5
SOLVER=$1 
#"dcyuchen"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

export EXECUTOR_OMP_NUM_THREADS=8

OMP_NUM_THREADS=8 KEYSTONE_MEM=180g ./bin/run-pipeline.sh \
  pipelines.speech.timit.$CLASS \
  --trainDataLocation $TRAIN_FEATURES \
  --trainLabelsLocation $TRAIN_LABELS \
  --testDataLocation $TEST_FEATURES \
  --testLabelsLocation $TEST_LABELS \
  --trainParts $NUM_PARTITIONS \
  --numPartitions $NUM_PARTITIONS \
  --lambdas $LAMBDAS \
  --gamma $GAMMA \
  --solver $SOLVER \
  --seed $SEED 2>&1 | tee /root/logs/timit-solver-$SOLVER-gamma-$GAMMA-lambdas-$LAMBDAS-nummodels-$NUM_MODELS-logs-"$LOG_SUFFIX".log

