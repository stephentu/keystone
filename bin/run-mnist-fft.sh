#!/bin/bash

SPARK_JAVA_OPTS="-Dspark.master=local[5]" KEYSTONE_MEM=4g ./bin/run-pipeline.sh \
  pipelines.images.mnist.MnistRandomFFT \
  --trainLocation ./train-mnist-dense-with-labels.data \
  --testLocation ./test-mnist-dense-with-labels.data \
  --numPartitions 10 \
  --lambda 1e-6 \
  --cocoaNumLocalItersFraction 0.5 \
  --cocoaBeta 1.0 \
  --sgdMiniBatchFraction 0.1 \
  --sgdStepSize 50.0 \
  --sgdDampen 0.95 \
  --numIters 1000 \
  --solver sgd \
  --numFFTs 4 \
  --blockSize 2048
