import numpy as np
import tensorflow as tf


def create_graph(graph_path):
    """Create a graph from a saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def create_labels(labels_path):
    with open(labels_path, 'rb') as f:
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        return labels


def run_inference_on_image(image_path, session, labels):

    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist: %s', image_path)
        return None

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
    predictions = session.run(softmax_tensor,
                              {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
    none_score = 0.0
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
        if human_string == 'none':
            none_score = score

    answer = labels[top_k[0]]
    score = predictions[top_k[0]]
    return answer, score


def get_flower_paths():

    import glob

    graph_path = 'flower/output_graph.pb'
    labels_path = 'flower/output_labels.txt'
    image_paths_for_label = {}
    for label in ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'):
        image_paths_for_label[label] = glob.glob('extra_photos/%s/*' % label)
    return graph_path, labels_path, image_paths_for_label


if __name__ == '__main__':

    graph_path, labels_path, image_paths_for_label = get_flower_paths()

    create_graph(graph_path)
    labels = create_labels(labels_path)
    with tf.Session() as session:
        metrics_for_label = {}
        for label, image_paths in image_paths_for_label.items():
            num_images = len(image_paths)
            num_hits = 0
            num_misses = 0
            total_hit_score = 0.0
            total_miss_score = 0.0
            for image_path in image_paths:
                print('\nClassifying image: %s' % image_path)
                answer, score = run_inference_on_image(image_path, session, labels)
                if answer == label:
                    num_hits += 1
                    total_hit_score += score
                else:
                    num_misses += 1
                    total_miss_score += score
                    print('WRONG ANSWER! Correct answer is "%s".' % label)
            hit_rate = num_hits / float(num_images)
            if num_hits > 0:
                mean_hit_score = total_hit_score / num_hits
            else:
                mean_hit_score = float('nan')
            if num_misses > 0:
                mean_miss_score = total_miss_score / num_misses
            else:
                mean_miss_score = float('nan')
            metrics_for_label[label] = (hit_rate, mean_hit_score, mean_miss_score)
        print('\nMetrics for each label:')
        for label, metrics in metrics_for_label.items():
            hit_rate, mean_hit_score, mean_miss_score = metrics
            print('%s:\n  hit rate = %.5f\n  mean score for hits = %.5f\n  mean score for misses = %.5f' % (label, hit_rate, mean_hit_score, mean_miss_score))
        print('')
