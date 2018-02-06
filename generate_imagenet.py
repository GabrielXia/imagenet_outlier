from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import glob
import os

# gzip imagenet*/*ubyte and gzip *ubyte to compress the ubyte file


def generate_outlier(train_features, train_labels, features, ratio):
    images_outlier = np.empty_like(train_features)
    labels_outlier = np.empty(train_labels.shape[0])
    labels_true = np.empty(train_labels.shape[0])
    if_outlier = np.empty(train_labels.shape[0])
    features_resnet = np.empty_like(features)
    pos = 0
    for i in range(10):
        i_position = np.where(train_labels == i)[0]
        num = i_position.shape[0]
        outlier_num = int(num * ratio)
        images_outlier[pos: pos + num] = train_features[i_position]
        features_resnet[pos: pos + num] = features[i_position]
        list_remove_i = range(10)
        list_remove_i.remove(i)
        false_labels = np.random.choice(list_remove_i, outlier_num, replace=True)
        labels_outlier[pos: pos + num] = (np.ones(num) * i)
        labels_outlier[pos: pos + outlier_num] = false_labels
        labels_true[pos: pos + num] = (np.ones(num) * i)
        if_outlier[pos: pos + outlier_num] = np.ones(outlier_num)
        if_outlier[pos + outlier_num: pos + num] = np.zeros(num - outlier_num)
        pos += num
    num_total = train_features.shape[0]
    idx = np.random.permutation(num_total)
    images_outlier, labels_outlier, labels_true, if_outlier, features_resnet = \
        images_outlier[idx], labels_outlier[idx], labels_true[idx], if_outlier[idx], features_resnet[idx]
    return images_outlier.astype(np.uint8), labels_outlier.astype(np.uint8), labels_true.astype(np.uint8), if_outlier.astype(np.uint8), features_resnet.astype(np.float32)


def read_dir_and_seperate(dir, train_test_ratio):
    images_list = glob.glob(dir + '/*JPEG')
    if len(images_list) == 0:
        print(dir + " has not jpeg")
    i = images_list[0]
    img = image.load_img(i, target_size=(224, 224))
    img_array = image.img_to_array(img)
    x, y, z = img_array.shape
    num = len(images_list)
    images_array = np.empty([num, x, y, z])
    for k in range(len(images_list)):
        im = image.load_img(images_list[k], target_size=(224, 224))
        im = image.img_to_array(im)
        #print(im.shape, images_list[k])
        # remove
        if im.shape == (x, y, z):
            images_array[k] = im
        else:
            print("shape doesn't match, pass this image: ", images_list[k])
    return separate_test_set(images_array, train_test_ratio)


def separate_test_set(imgs, train_test_ratio):
    test_ratio = 1. / (1. + train_test_ratio)
    num = imgs.shape[0]
    num_test = int(num * test_ratio)
    num_train = num - num_test
    idx = np.random.permutation(num)
    imgs_perm = imgs[idx]
    training = imgs_perm[:num_train]
    test = imgs_perm[num_train:]
    return training, test

def extract_feature(x):
    print("extracting feature using resnet")
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)
    y = model.predict(x)
    num, feature_num = y.shape[0], y.shape[3]
    y = y.reshape([num, feature_num])
    return y.astype(np.float32)


def main(train_test_ratio=10):
    dirs = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668',
              'n01514859', 'n01518878']
    dir0 = dirs[0]

    print("reading started")
    train_images, test_images = read_dir_and_seperate(dir0, train_test_ratio)
    train_labels = np.array([0] * len(train_images))
    test_labels = np.array([0] * len(test_images))
    for i in range(len(dirs))[1: ]:
        new_train_images, new_test_images = read_dir_and_seperate(dirs[i], train_test_ratio)
        train_images = np.concatenate((train_images, new_train_images))
        train_labels = np.concatenate((train_labels, [i] * len(new_train_images)))
        test_images = np.concatenate((test_images, new_test_images))
        test_labels = np.concatenate((test_labels, [i] * len(new_test_images)))
    print("reading finished")

    features = extract_feature(train_images)

    # seperate data into training set and test set
    for ratio in [k / 10.0 for k in range(4)]:
        dir_name = 'imagenet_outlier_' + str(ratio)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        images_outlier, labels_outlier, labels_true, if_outlier, features_resnet = generate_outlier(train_images, train_labels, features, ratio)

        images_name = dir_name + '/images-' + str(ratio) + '-ubyte'
        labels_outlier_name = dir_name + '/labels-outlier-' + str(ratio) + '-ubyte'
        labels_true_name = dir_name + '/labels-true-' + str(ratio) + '-ubyte'
        if_outlier_name = dir_name + '/if-outlier-' + str(ratio) + '-ubyte'
        features_name = dir_name + '/features-' + str(ratio) + '-ubyte'

        print("start write")
        # write images
        num, x, y, z = images_outlier.shape
        header = np.array([0x0803, num, x, y, z], dtype='>i4')
        with open(images_name, 'wb') as f:
            f.write(header.tobytes())
            f.write(images_outlier.tobytes())

        # write labels
        header = np.array([0x0801, len(labels_outlier)], dtype='>i4')
        with open(labels_outlier_name, 'wb') as f:
            f.write(header.tobytes())
            f.write(labels_outlier.tobytes())

        header = np.array([0x0801, len(labels_true)], dtype='>i4')
        with open(labels_true_name, 'wb') as f:
            f.write(header.tobytes())
            f.write(labels_true.tobytes())

        header = np.array([0x0801, len(if_outlier)], dtype='>i4')
        with open(if_outlier_name, 'wb') as f:
            f.write(header.tobytes())
            f.write(if_outlier.tobytes())

        # write features
        num, x =  features_resnet.shape
        header = np.array([0x0805, num, x], dtype='>i4')
        with open(features_name, 'wb') as f:
            f.write(header.tobytes())
            f.write(features_resnet.tobytes())

    # for test set
    test_num, test_x, test_y, test_z = test_images.shape
    idx = np.random.permutation(test_num)
    test_images = test_images[idx]
    test_labels = test_labels[idx]
    header = np.array([0x0803, test_num, test_x, test_y, test_z], dtype='>i4')
    with open('test-images-ubyte', 'wb') as f:
        f.write(header.tobytes())
        f.write(test_images.astype(np.uint8).tobytes())
    header = np.array([0x0801, len(test_labels)], dtype='>i4')
    with open('test-labels-ubyte', 'wb') as f:
        f.write(header.tobytes())
        f.write(test_labels.astype(np.uint8).tobytes())

main()

