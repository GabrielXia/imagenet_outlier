import numpy as np
from read_outlier_imagenet import ImagenetOutlier
from keras.applications.resnet50 import ResNet50

model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

for ratio in [k / 10.0 for k in range(2)]:
    r = int(ratio * 10.) / 10.
    o = ImagenetOutlier(r)
    print("for ratio = ", r)
    print("train_images: ", o.train_images.shape)
    print("train_labels: ", o.train_labels.shape)
    print("train_labels_true: ", o.train_labels_true.shape)
    print("train_if_outlier: ", o.train_if_outlier.shape)
    print("features: ", o.features.shape)

    print("test_images: ", o.test_images.shape)
    print("test_labels: ", o.test_labels.shape)

    # test ratio
    print("the outlier ratio is: ", np.sum(o.train_if_outlier)/ o.train_if_outlier.shape[0], "the ratio expected: k")

    # test feature
    num_train = o.test_images.shape[0]
    a = np.array(range(num_train))
    idx = np.random.choice(a, 10)
    featrues = o.features[idx]
    imgs = o.train_images[idx]

    y = model.predict(imgs)
    num, feature_num = y.shape[0], y.shape[3]
    y = y.reshape([num, feature_num])
    diff = np.linalg.norm(y - featrues)
    print("the differnce: ", diff)

    print("---------------------------------")