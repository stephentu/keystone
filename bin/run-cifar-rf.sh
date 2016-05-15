#!/bin/bash

set -x

FWDIR="$(cd `dirname $0`/..; pwd)"
pushd $FWDIR

export SPARK_HOME="/root/spark"
source /root/spark/conf/spark-env.sh

export MEM=180g

MASTER=`cat /root/spark-ec2/cluster-url`
CLASS=CifarRandomFeatLBFGS

CIFAR_TRAIN_DIR="s3n://cifar-augmented/cifar_train_featurized_augmented_512_flip"
CIFAR_TEST_DIR="s3n://cifar-augmented/cifar_test_featurized_augmented_512_flip"

NUM_PARTITIONS=1024
LAMBDAS=1e-6
NUM_ITERS=300
NUM_COSINES=202752
BLOCK_SIZE=6144
GAMMA=0.00707
SEED=123123
SOLVER="lbfgs"
NORM_STD="false"
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

export EXECUTOR_OMP_NUM_THREADS=1

OMP_NUM_THREADS=1 KEYSTONE_MEM=180g ./bin/run-pipeline.sh \
  pipelines.images.cifar.$CLASS \
  --trainLocation $CIFAR_TRAIN_DIR \
  --testLocation $CIFAR_TEST_DIR \
  --trainParts $NUM_PARTITIONS \
  --testParts $NUM_PARTITIONS \
  --lambda $LAMBDAS \
  --cosineGamma $GAMMA \
  --blockSize $BLOCK_SIZE \
  --numCosineFeatures $NUM_COSINES \
  --numIters $NUM_ITERS \
  --solver $SOLVER \
  --normStd $NORM_STD \
  --seed $SEED 2>&1 | tee /root/logs/cifar-512-solver-$SOLVER-gamma-$GAMMA-lambda-$LAMBDAS-numIter-$NUM_ITERS-numCosine-$NUM_COSINES-normStd-$NORM_STD-logs-"$LOG_SUFFIX".log

