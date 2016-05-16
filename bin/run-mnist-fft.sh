#!/bin/bash

SPARK_JAVA_OPTS="-Dspark.master=local[5]" KEYSTONE_MEM=4g ./bin/run-pipeline.sh \
  pipelines.images.mnist.MnistRandomFFT \
  --trainLocation ./train-mnist-dense-with-labels.data \
  --testLocation ./test-mnist-dense-with-labels.data \
  --numFFTs 4 \
  --lambda 0.1 \
  --cocoaNumLocalItersFraction 0.1 \
  --solver cocoa \
  --blockSize 2048
