import cv2
import glob
from random import randint
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os
from collections import OrderedDict
import itertools

from feature_comparator.siamese_api.feature_vector import FeatureVector
from feature_extraction.mars_api.mars_api import MarsExtractorAPI
from feature_comparator.siamese_api.siamese import SiameseComparator
from tf_session.tf_session_runner import SessionRunner
from tf_session.tf_session_utils import Inference
from data.feature_comparator.siamese_api.inputs import path as input_path
from data.feature_comparator.siamese_api.trained import path as model_path
count_0 = 0
count_1 = 0


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            input1 = []
            input2 = []
            labels = []
            for batch_sample in batch_samples:
                input1.append(batch_sample[0])
                input2.append(batch_sample[1])
                labels.append([batch_sample[2]])
            # print(input.shape)
            X_train1 = np.array(input1)
            X_train2 = np.array(input2)
            Y_train = np.array(labels)
            # print(X_train2.shape)
            # print(Y_train.shape)

            yield [np.expand_dims(X_train1, 0), np.expand_dims(X_train2, 0)], np.expand_dims(Y_train, 0)


def create_samples(vector):
    global count_0
    global count_1
    keys = vector.keys()
    key_pairs = [ _ for _ in itertools.combinations_with_replacement(keys, 2)]
    print(key_pairs)
    samples = []
    for k1, k2 in key_pairs:
        val1 = vector[k1]
        val2 = vector[k2]
        label = 0
        if k1 == k2:
            count_1+=1
            label = 1
        else:
            count_0+=1
        sample = [[v1, v2, label] for v1, v2 in itertools.product(val1, val2)]
        samples.extend(sample)
    return samples


def extract_features(patch, ip, op):
    patch[0] = cv2.equalizeHist(patch[0])
    patch[1] = cv2.equalizeHist(patch[1])
    patch[2] = cv2.equalizeHist(patch[2])
    ip.push(Inference(patch, meta_dict={}))
    op.wait()
    ret, feature_inference = op.pull()
    if ret:
        return feature_inference.get_result()


# if __name__ == '__main__':
def train():
    feature_vector = FeatureVector()
    session_runner = SessionRunner()
    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)
    session_runner.start()
    extractor.run()

    for id in range(1, 5):
        image_files = glob.glob('/home/allahbaksh/Tailgating_detection/SecureIt/data/obj_tracking/outputs/patches/{}/*.jpg'.format(id))
        for image_file in image_files:
            patch = cv2.imread(image_file)
            f_vec = extract_features(patch, ip, op)
            # print(f_vec.shape)
            # print(f_vec[])
            # break
            feature_vector.add_vector(id, f_vec[0])

    # for x in range(200):
    #     feature_vector.add_vector(randint(0, 30), [randint(0, 128) for _ in range(128)])
    samples = create_samples(feature_vector.get_vector_dict())
    print(count_0)
    print(count_1)
    # print(feature_vector.get_vector_dict())
    model = SiameseComparator()()
    sklearn.utils.shuffle(samples)
    # print()
    # print(samples[1])
    # print(len(samples))
    train_samples, val_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples, batch_size=16)
    validation_generator = generator(val_samples, batch_size=16)
    epoch = 10
    saved_weights_name = 'model.h5'
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=3,
                               mode='min',
                               verbose=1)
    checkpoint = ModelCheckpoint(saved_weights_name,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae', 'acc'])
    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_samples),
                        epochs=epoch,
                        verbose=1,
                        validation_data = validation_generator,
                        nb_val_samples = len(val_samples),
                        callbacks=[early_stop, checkpoint, tensorboard]
                        )

def create_dataset():
    feature_vector = FeatureVector()
    session_runner = SessionRunner()
    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)
    session_runner.start()
    extractor.run()

    file_names = glob.glob('/home/developer/Desktop/Market-1501-v15.09.15/bounding_box_train/*.jpg') + glob.glob('/home/developer/Desktop/Market-1501-v15.09.15/gt_bbox/*.jpg')
    print(len(file_names))
    batch_size = 100
    # stacked_images,_=inp.getstackedimages()
    csvfile = open("/home/developer/Desktop/reid/market1501/market1501_dataset.csv", "a")
    for c, image_file in enumerate(file_names):
        print(c)
        patch = cv2.imread(image_file)
        label = int(image_file.split("/")[-1].split("_")[0])
        f_vec = extract_features(patch, ip, op)
        # print(f_vec.shape)
        # print(f_vec[])
        # break
        # feature_vector.add_vector(id, f_vec[0])
        vector = ""
        # print (f_vec[0].shape)
        # break
        for vec in f_vec[0]:
            vector = vector + "," + str(vec)
        csvfile.write(str(label) + "," + vector + "\n")
    csvfile.close()
    session_runner.stop()

def create_dataset_from_folder():
    feature_vector = FeatureVector()
    session_runner = SessionRunner()
    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)
    session_runner.start()
    extractor.run()



    batch_size = 100
    # stacked_images,_=inp.getstackedimages()
    csvfile = open("/home/developer/Desktop/reid/market1501/mars_infy_dataset_full_image.csv", "a")
    for i in range(1,6):
        file_names = glob.glob('/home/developer/Desktop/mars_test_dataset_full_image/{}/*.jpg'.format(i))
        print(len(file_names))
        for c, image_file in enumerate(file_names):
            print(c)
            patch = cv2.imread(image_file)
            label = i
            f_vec = extract_features(patch, ip, op)
            # print(f_vec.shape)
            # print(f_vec[])
            # break
            # feature_vector.add_vector(id, f_vec[0])
            vector = ""
            # print (f_vec[0].shape)
            # break
            for vec in f_vec[0]:
                vector = vector + "," + str(vec)
            csvfile.write(str(label) + "," + vector + "\n")
    csvfile.close()
    session_runner.stop()

def test():
    model = SiameseComparator()()
    # model.load_weights(model_path.get()+'/siamese-mars-small128.h5')
    model.load_weights("/home/developer/Desktop/model_12_28_2018_12_02_56.h5")
    model.summary()
    feature_vector = FeatureVector()
    session_runner = SessionRunner()
    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)
    session_runner.start()
    extractor.run()
    image_files = []
    for id in range(1, 5):
        # image_files.append(glob.glob(
        #     input_path.get()+'/patches/{}/*.jpg'.format(id)))
        image_files.append('/home/developer/Desktop/test/aa/{}.jpg'.format(id))
    print(len(image_files))
    print(image_files)
    patch0 = [cv2.imread(image_files[0])]
    # patch0_1 = [cv2.imread(image_files[0][randint(0, len(image_files[0]))]) for _ in range(10)]
    patch1 = [cv2.imread(image_files[1])]
    patch2 = [cv2.imread(image_files[2])]
    patch3 = [cv2.imread(image_files[3])]
    #patch_pair = [_ for _ in itertools.combinations_with_replacement([patch0[0], patch1[0], patch2[0], patch3[0]], 2)]

    f_vec0 = np.array([extract_features(patch, ip, op)[0] for patch in patch0])
    # f_vec0_1 = np.array(extract_features(patch0_1, ip, op))
    f_vec1 = np.array([extract_features(patch, ip, op)[0] for patch in patch1])
    f_vec2 = np.array([extract_features(patch, ip, op)[0] for patch in patch2])
    f_vec3 = np.array([extract_features(patch, ip, op)[0] for patch in patch3])
    #print(f_vec1)

    output = model.predict([np.expand_dims(f_vec0, 0), np.expand_dims(f_vec1, 0)])
    print(output)
    output = model.predict([np.expand_dims(f_vec0, 0), np.expand_dims(f_vec2, 0)])
    print(output)
    output = model.predict([np.expand_dims(f_vec0, 0), np.expand_dims(f_vec3, 0)])
    print(output)
    output = model.predict([np.expand_dims(f_vec1, 0), np.expand_dims(f_vec2, 0)])
    print(output)
    output = model.predict([np.expand_dims(f_vec1, 0), np.expand_dims(f_vec3, 0)])
    print(output)
    output = model.predict([np.expand_dims(f_vec2, 0), np.expand_dims(f_vec3, 0)])
    print(output)
def distance_eval():
    feature_vector = FeatureVector()
    session_runner = SessionRunner()
    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)
    session_runner.start()
    extractor.run()

    file_names = glob.glob('/home/developer/Desktop/test/aa/*.jpg')
    print(len(file_names))
    f_vec_lst = []
    for c, image_file in enumerate(file_names):
        print(c)
        patch = cv2.imread(image_file)
        # label = i
        print("file_name", image_file)
        f_vec = extract_features(patch, ip, op)
        f_vec_lst.append(f_vec)
        # print(f_vec.shape)
        # print(f_vec[])
        # break
        # feature_vector.add_vector(id, f_vec[0])
        vector = ""
        # print (f_vec[0].shape)
        # break
    print(compute_dist(f_vec_lst[0], f_vec_lst[1], type='cosine'))
    print(compute_dist(f_vec_lst[1], f_vec_lst[2], type='cosine'))
    print(compute_dist(f_vec_lst[2], f_vec_lst[3], type='cosine'))
    print(compute_dist(f_vec_lst[0], f_vec_lst[2], type='cosine'))
    print(compute_dist(f_vec_lst[0], f_vec_lst[3], type='cosine'))
    session_runner.stop()

def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist


if __name__ == '__main__':
    create_dataset()