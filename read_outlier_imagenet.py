import gzip

import numpy
from tensorflow.python.platform import gfile


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in imagenet image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    channel = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images * channel)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, channel)
    return data


def extract_features(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2053:
      raise ValueError('Invalid magic number %d in imagenet feature file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    channel = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images * channel)
    data = numpy.frombuffer(buf, dtype=numpy.float32)
    data = data.reshape(num_images, rows, cols, channel)
    return data


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in imagenet label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


def get_imagenet_outlier(ratio, one_hot=False):
    r = int(ratio * 10.) / 10.
    dir = 'imagenet_outlier_' + str(ratio)
    images_gz = dir + '/images-' + str(ratio) + '-ubyte.gz'
    labels_outlier = dir + '/labels-outlier-' + str(ratio) + '-ubyte.gz'
    labels_true = dir + '/labels-true-' + str(ratio) + '-ubyte.gz'
    if_outlier_name = dir + '/if-outlier-' + str(ratio) + '-ubyte.gz'
    features_name = dir + '/features-' + str(ratio) + '-ubyte.gz'
    with gfile.Open(images_gz, 'rb') as f:
        train_images = extract_images(f)
    with gfile.Open(labels_outlier, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)
    with gfile.Open(labels_true, 'rb') as f:
        train_labels_true = extract_labels(f)
    with gfile.Open(if_outlier_name, 'rb') as f:
        train_if_outlier = extract_labels(f)
    with gfile.Open(features_name, 'rb') as f:
        features = extract_features(f)
    return train_images, train_labels, train_labels_true, train_if_outlier, features


def get_test(one_hot=False):
    images_gz = 'test-images-ubyte.gz'
    labels_gz = 'test-labels-ubyte.gz'
    with gfile.Open(images_gz, 'rb') as f:
      test_images = extract_images(f)
    with gfile.Open(labels_gz, 'rb') as f:
      test_labels = extract_labels(f, one_hot=one_hot)
    return test_images, test_labels


class ImagenetOutlier:
    def __init__(self, outlier_ratio, one_hot=False):
        self.outlier_ratio = outlier_ratio
        self.train_images, self.train_labels, self.train_labels_true, self.train_if_outlier, self.features = \
            get_imagenet_outlier(outlier_ratio, one_hot)
        self.test_images, self.test_labels = get_test(one_hot)

if __name__ == '__main__':
    a = ImagenetOutlier(0.1)
    print(a.features.shape)