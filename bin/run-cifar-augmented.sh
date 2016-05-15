#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=CifarDCSolver

CIFAR_TRAIN_DIR="s3n://cifar-augmented/cifar_train_featurized_augmented_512_flip"
CIFAR_TEST_DIR="s3n://cifar-augmented/cifar_test_featurized_augmented_512_flip"

NUM_MODELS=$3
NUM_PARTITIONS=$3
LAMBDAS=$2
GAMMA=$1
SEED=$4
SOLVER="dcsvm"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

export EXECUTOR_OMP_NUM_THREADS=8

OMP_NUM_THREADS=8 KEYSTONE_MEM=180g ./bin/run-pipeline.sh \
  pipelines.images.cifar.$CLASS \
  --trainLocation $CIFAR_TRAIN_DIR \
  --testLocation $CIFAR_TEST_DIR \
  --trainParts $NUM_PARTITIONS \
  --testParts $NUM_PARTITIONS \
  --numPartitions $NUM_PARTITIONS \
  --numModels $NUM_PARTITIONS \
  --lambdas $LAMBDAS \
  --gamma $GAMMA \
  --solver $SOLVER \
  --seed $SEED 2>&1 | tee /root/logs/cifar-512-solver-$SOLVER-gamma-$GAMMA-lambdas-$LAMBDAS-nummodels-$NUM_MODELS-logs-"$LOG_SUFFIX".log

