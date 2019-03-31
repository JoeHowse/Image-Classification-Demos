#!/usr/bin/env bash

TENSOR_FLOW_SRC_DIR=~/SDKs/TensorFlow/src

NUM_PHOTOS=`find photos/*/* -type f | wc -l`
SAMPLES_PER_TRAINING_PHOTO=75
TESTING_PERCENTAGE=10
VALIDATION_PERCENTAGE=10
TRAINING_BATCH_SIZE=100
NUM_TRAINING_STEPS=$(($NUM_PHOTOS*$SAMPLES_PER_TRAINING_PHOTO/$TRAINING_BATCH_SIZE*(100-$TESTING_PERCENTAGE-$VALIDATION_PERCENTAGE)/100))

$TENSOR_FLOW_SRC_DIR/bazel-bin/tensorflow/examples/image_retraining/retrain \
--image_dir photos \
--output_graph output_graph.pb \
--output_labels output_labels.txt \
--summaries_dir logs \
--bottleneck_dir bottleneck \
--testing_percentage $TESTING_PERCENTAGE \
--validation_percentage $VALIDATION_PERCENTAGE \
--train_batch_size $TRAINING_BATCH_SIZE \
--how_many_training_steps $NUM_TRAINING_STEPS \
--validation_batch_size=-1 \
--print_misclassified_test_images \
> retrain.log
