from src.dataloader import *
from src.log_and_visualization import *
from src.model import *
from sklearn.metrics import  recall_score, roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, f1_score, mean_squared_error
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
import tensorflow.keras.backend as K
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow as tf

# 必须要下面这行代码
tf.compat.v1.disable_eager_execution()
print(tf.__version__)


# 我自己使用的函数
def stats_graph(graph):
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph,
                                           options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


# 获取模型每一层的参数详情

# 获取模型浮点运算总次数和模型的总参



n_batch = 64
n_epochs = 150

data = np.load('./data/phishing_2016.npz')
dataset = shuffle(data['X_train'], data['y_train'])
x_test, y_test_ = data['X_test'], data['y_test']

model = define_discriminator_3(in_shape=(200, 67), n_classes=2)
stats_graph(model)
# model.load_weights("./Discriminator/model_trans+cnn-32.h5")
bat_per_epo = int(dataset[0].shape[0] / n_batch)

y_test = []
for i1, i2 in y_test_:
    y_test.append(i2)

y_test = np.array(y_test)
print(x_test.shape, y_test.shape)

last_acc = 0
for e in range(n_epochs):
    d_r1 = 0
    d_r2_total = 0
    for i in range(bat_per_epo):
        [X_real, labels_real], y_real = generate_real_samples(dataset, i, n_batch)
        # _, d_r1, d_r2 = model.train_on_batch(X_real, labels_real)
        d_r2 = model.train_on_batch(X_real, labels_real)
        d_r2_total += d_r2
    y_pred = model.predict(x_test)
    y_pred_1 = []
    y_pred_1_prob = []
    # for p1, p2 in y_pred[1]:
    for p1, p2 in y_pred:
        y_pred_1.append(0 if p1 > p2 else 1)
        y_pred_1_prob.append(p2)
    y_pred_1 = np.array(y_pred_1)
    y_pred_1_threshold = list(np.ravel(y_pred_1))
    acc = accuracy_score(list(np.ravel(y_test)), y_pred_1_threshold)
    print("Epoch ", e, acc, d_r2_total)
    # if acc >= last_acc:
        # print("model saved.")
        # model.save("./new_dataset_model/model_trans_3.h5")
        # last_acc = acc

