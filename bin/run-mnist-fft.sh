#!/bin/bash

SPARK_JAVA_OPTS="-Dspark.master=local[5]" KEYSTONE_MEM=4g ./bin/run-pipeline.sh \
  pipelines.images.mnist.MnistRandomFFT \
  --trainLocation ./train-mnist-dense-with-labels.data \
  --testLocation ./test-mnist-dense-with-labels.data \
  --numPartitions 10 \
  --lambda 1e-6 \
  --cocoaNumLocalItersFraction 1.0 \
  --cocoaBeta 1.0 \
  --sgdMiniBatchFraction 1.0 \
  --sgdStepSize 600.0 \
  --numIters 20 \
  --solver cocoa \
  --numFFTs 1 \
  --blockSize 512
