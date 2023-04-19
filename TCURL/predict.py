from src.dataloader import *
from src.log_and_visualization import *
from src.model import *
from sklearn.metrics import  recall_score, roc_auc_score, accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, f1_score, mean_squared_error, jaccard_score, roc_curve, plot_roc_curve
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
from time import time
def test(epoch, model, dataset, d_weight, g_weight, name):
    model.load_weights(d_weight)
    # g_model.load_weights(g_weight)

    x_test, y_test_ = dataset
    #
    # z_input = generate_latent_points(50, len(x_test))
    # x_pred = g_model.predict([x_test, z_input, y_test_])
    #
    # x_pred = np.array(x_pred)
    x_test = np.array(x_test)
    # mse = mean_squared_error(x_test.flatten(), x_pred.flatten())
    # nrmse = normalized_root_mse(x_test, x_pred, normalization='euclidean')
    # ssim = structural_similarity(x_test, x_pred, data_range=x_test.max()-x_test.min(), channel_axis=2)
    #
    # x_pred[x_pred >= 0.5] = 1
    # x_pred[x_pred < 0.5] = 0

    y_pred_1_ = model.predict(x_test)
    # y_pred_2_ = d_model.predict(x_pred)

    y_pred_1 = []
    y_pred_1_prob = []
    y_pred_2 = []
    y_pred_2_prob = []
    if len(y_pred_1_) == 2:
        y_pred_1_ = y_pred_1_[1]
    for p1, p2 in y_pred_1_:
        y_pred_1.append(1 if p1 > p2 else 0)
        y_pred_1_prob.append(p1)
    # for p1, p2 in y_pred_2_[1]:
    #    y_pred_2.append(0 if p1 > p2 else 1)
    #    y_pred_2_prob.append(p2)

    y_test = []
    for i1, i2 in y_test_:
        y_test.append(i1)

    y_test = np.array(y_test)
    y_pred_1 = np.array(y_pred_1)
    y_pred_1_threshold = list(np.ravel(y_pred_1))

    tn, fp, fn, tp = confusion_matrix(list(np.ravel(y_test)), y_pred_1_threshold).ravel()
    fpr1 = fp/(fp+tn)
    fpr2 = fn/(fn+tp)
    spe = tn/(fp+tn)
    accuracy_1 = accuracy_score(list(np.ravel(y_test)), y_pred_1_threshold)
    sensitivity_1 = recall_score(list(np.ravel(y_test)), y_pred_1_threshold)
    precision_1 = precision_score(list(np.ravel(y_test)), y_pred_1_threshold)
    f1_1 = f1_score(list(np.ravel(y_test)), y_pred_1_threshold)
    auc_1 = roc_auc_score(list(np.ravel(y_test)), list(np.ravel(y_pred_1_prob)))

    fpr, tpr, thersholds = roc_curve(list(np.ravel(y_test)), list(np.ravel(y_pred_1_prob)))

    plt.plot(fpr, tpr, label=name+"(AUC = {0:.4f})".format(auc_1), lw=1.5)

    plt.xlim([-0.05, 0.65])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([0.85, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    # plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # y_pred_2 = np.array(y_pred_2)
    # y_pred_2_threshold = list(np.ravel(y_pred_2))
    #
    # tn, fp, fn, tp = confusion_matrix(list(np.ravel(y_test)), y_pred_2_threshold).ravel()
    # accuracy_2 = accuracy_score(list(np.ravel(y_test)), y_pred_2_threshold)
    # sensitivity_2 = recall_score(list(np.ravel(y_test)), y_pred_2_threshold)
    # precision_2 = precision_score(list(np.ravel(y_test)), y_pred_2_threshold)
    # f1_2 = f1_score(list(np.ravel(y_test)), y_pred_2_threshold)
    # auc_2 = roc_auc_score(list(np.ravel(y_test)), list(np.ravel(y_pred_2_prob)))

    print("Epoch " + str(i))
    print(accuracy_1, sensitivity_1,spe, fpr2, precision_1, f1_1, auc_1)
    # print(accuracy_2, sensitivity_2, precision_2, f1_2, auc_2)
    # print(mse, ssim, nrmse)

if __name__=='__main__':
    data = np.load('./data/new_url_dataset_test.npz')
    dataset = data['X_test'], data['y_test']
    print(len(dataset[0]))

    d_model_cnn_mhsa = cnn_mhsa(in_shape=(200, 67), n_classes=2)
    d_model_tcurl = define_discriminator_4(in_shape=(200, 67), n_classes=2)
    d_model_cnn = define_discriminator_1(in_shape=(200, 67), n_classes=2)
    d_model_lstm = define_discriminator(in_shape=(200, 67), n_classes=2)
    d_model_trans = define_discriminator_3(in_shape=(200, 67), n_classes=2)
    d_model_trans_gan = define_discriminator_3(in_shape=(200, 67), n_classes=2)
    d_model_cnn_gan = define_discriminator_1(in_shape=(200, 67), n_classes=2)
    g_model = define_generator_1(latent_dim=(50,), signal_shape=(200, 67), label_shape=(2,))
    for i in range(72,  73):
        ind = ''
        if i < 100:
            ind = '0'
        d_weight_cnn_mhsa = "./new_dataset_model/model_mhsa_1.h5"
        d_weight_cnn = "./new_dataset_model/model_cnn_1.h5"
        d_weight_lstm = "./new_dataset_model/model_lstm_1.h5"
        d_weight_trans = "./new_dataset_model/model_trans_3.h5"
        d_weight_trans_gan = "./transgan_model_2016/dmodel_000"+ind+str(i)+".h5"
        d_weight_cnn_gan = "./gan_model_2016/dmodel_000"+ind+str(i)+".h5"
        d_weight_tcurl = "./new_dataset_model/model_trans+cnn_10.h5"
        fig, ax = plt.subplots()
        g_weight = "./PhishGan_11/gmodel_000"+ ind +str(i)+".h5"
        t = time()
        # test(i, d_model_tcurl, dataset, d_weight=d_weight_tcurl, g_weight=g_weight, name="TCURL")
        test(i, d_model_cnn_mhsa, dataset, d_weight=d_weight_cnn_mhsa, g_weight=g_weight, name="CNN-MHSA")
        # test(i, d_model_cnn, dataset, d_weight=d_weight_cnn, g_weight= g_weight, name="CNN")
        # test(i, d_model_cnn_gan, dataset, d_weight=d_weight_cnn_gan, g_weight=g_weight, name="CNN-GAN")
        # test(i, d_model_lstm, dataset, d_weight=d_weight_lstm, g_weight=g_weight, name="BiLSTM")
        # test(i, d_model_trans, dataset, d_weight=d_weight_trans, g_weight=g_weight, name="Trans")
        # test(i, d_model_trans_gan, dataset, d_weight=d_weight_trans_gan, g_weight=g_weight, name="Trans-GAN")
        print(time() - t)
        # plt.show()
        # fig.savefig('roc_dl.eps', dpi=1000, format='eps')