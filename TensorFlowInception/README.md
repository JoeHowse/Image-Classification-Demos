# TensorFlow Inception image_retraining

This sample shows how to retrain and use the Inception image classifier in TensorFlow. It has been tested on Mac OS 10.11 (El Capitan) with Python 2.7 and TensorFlow 1.1.

## Setting up dependencies

Follow the official tutorial, [Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources/), to obtain these dependencies:

* The Bazel build system
* Tensor Flow's Python dependencies: six, NumPy, wheel
* TensorFlow's source code
* An installation of TensorFlow into your Python environment. (A virtualenv is recommended.)

Also, build TensorFlow's `retrain` tool by running the following command from the root of your TensorFlow source directory:

```
$ bazel build --config opt tensorflow/examples/image_retraining:retrain
```

(Omit the `--config opt` flag if your machine does not support the AVX instruction set. Generally, recent x86 machines do support AVX.)

## Retraining the Inception classifier

The `flower/retrain.sh` script shows how to use TensorFlow's `retrain` tool to create your own variant of the Inception classifier. Edit the script's `TENSOR_FLOW_SRC_DIR` variable to specify the root of your TensorFlow source directory.

As input, the script takes its training images from the subfolders of the `flower/photos` directory. ***The training images must be in `JPG` format; otherwise, `retrain` ignores them.*** The subfolder names become class names for the retrained model.

The training process is iterative. Regardless of the total number of training images, one should avoid sampling any given image too many (or too few) times across all iterations. In other words, one should avoid over-fitting (or under-fitting) the model to the training data. To address this concern, the `retrain.sh` script computes the number of iterations such that each training image will be sampled approximately 75 times, as specified by the declaration, `SAMPLES_PER_TRAINING_PHOTO=75`.

The script saves the following outputs under the `flower` directory:

* `output_graph.pb` : The **graph file**, which defines all of the classifier's criteria.
* `output_labels.txt`: The **labels file**, which contains an ordered list of human-readable names for the classes.
* `logs`: A folder of log files describing the results of the training and validation. These logs can be viewed using TensorFlow's `tensorboard` tool, as described in the [TensorBoard README](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/tensorboard/README.md).
* `bottleneck`: A folder of intermediate training results. These are cached so that `retrain` runs faster if we rerun it with some of the same training images.

Also, the script redirects its console output to `retrain.log`, in the same folder as the script itself. This log includes metrics and a list of misclassified test images.

## Using the retrained classifier in Python 

The `run_inferences.py` script shows how to classify images using the retrained model from the graph file (`flower/output_graph.pb`) and labels file (such as `flower/output_labels.txt`).

As inputs, the `extra_photos` folder contains some test images of different scenes than the ones used in training.

The script prints the following outputs to the console:

* For each input photo:
  * The top 5 scores and labels
  * A warning if the 1st top label is wrong
* For each label in the model:
  * The hit rate (the proportion of 1st top labels that are correct)
  * The mean score for hits
  * The mean score for misses

In production code, one could weed out most false positives (and relatively few true positives) with some threshold score that is greater than the mean score for misses and less than the mean score for hits.

`log.txt` contains a sample of console output from `run_inferences.py`.

If you installed TensorFlow into a virtualenv, you may want to use the `run_inferences.sh` script to simultaneously activate the virtualenv and launch `run_inferences.py`.

## References

The flower example is based on the official tutorial, [How to Retrain Inception's Final Layer for New Categories](https://www.tensorflow.org/tutorials/image_retraining).

Steven Puttemans (PhD candidate, EAVISE research group, KU Leuven) recommended 75 samples per training image as a rule of thumb, based on research by one of his Master's students.

`run_inferences.py` is adapted from Dhruv Karan's `retraining_example.py` script [on GitHub](https://github.com/eldor4do/Tensorflow-Examples/blob/master/retraining-example.py).