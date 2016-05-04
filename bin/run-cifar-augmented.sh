#!/bin/bash

FWDIR="$(cd `dirname $0`; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=CifarDCSolver

CIFAR_TRAIN_DIR="s3n://cifar-augmented/cifar_train_featurized_augmented_512_flip"
CIFAR_TEST_DIR="s3n://cifar-augmented/cifar_test_featurized_augmented_512_flip"
NUM_PARTITIONS=128
LAMBDAS=0.1
GAMMA=0.0073
SEED=8975323
METHOD="dcsvm"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

OMP_NUM_THREADS=8 KEYSTONE_MEM=200g /root/stephentu-keystone/bin/run-pipeline.sh \
  pipelines.images.cifar.$CLASS \
  --trainLocation $CIFAR_TRAIN_DIR \
  --testLocation $CIFAR_TEST_DIR \
  --trainParts 128 \
  --testParts 128 \
  --numModels 64 \
  --lambdas $LAMBDAS \
  --gamma $GAMMA \
  --seed $SEED 2>&1 | tee /mnt/cifar-512-solver-$SOLVER-gamma-$GAMMA-lambda-$LAMBDA-logs-"$LOG_SUFFIX".log

