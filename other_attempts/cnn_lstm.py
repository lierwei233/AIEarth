# coding: utf-8

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from netCDF4 import Dataset
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from sklearn.impute import SimpleImputer
import shutil

#%% md
# # 读取数据
# 
# 首先读取数据并定义提取数据的函数（根据start和end读取对应的批次数据）

# In[4]:
inp1_cmip = Dataset('/tcdata/enso_round1_train_20210201/CMIP_train.nc','r')
inp2_cmip = Dataset('/tcdata/enso_round1_train_20210201/CMIP_label.nc','r')
inp1_soda = Dataset('/tcdata/enso_round1_train_20210201/SODA_train.nc','r')
inp2_soda = Dataset('/tcdata/enso_round1_train_20210201/SODA_label.nc','r')

def dataloader(start, end, inp1, inp2):
    trX = np.zeros(shape=(end-start, 24, 72, 4))
    trY = np.zeros(shape=(end-start, 1))
    
    feature_names = ['sst', 't300', 'ua', 'va']
    for i, name in enumerate(feature_names):
        trX[:, :, :, i] = np.array(inp1.variables[name][start//12:end//12, :12, :, :]).reshape(-1, 24, 72)
    trY = np.array(inp2.variables['nino'][start//12:end//12, :12]).reshape(-1, 1)
    
    return trX, trY

#%% md

# ## 缺失值填充
# 
# CMIP的数据中存在nan，需要进行处理
# 
# 根据所有月份的数据，使用sklearn计算共计$24 \times 72 \times 4 = 6912$个特征的平均值，用于后续的缺失值填充

# %%
# get missing value imputer for all CMIP dataset
num_years = 4645
X = np.zeros([num_years*12, 24*72, 4])
feature_names = ['sst', 't300', 'ua', 'va']
for i, name in enumerate(feature_names):
    X[:, :, i] = np.array(inp1_cmip.variables[name][:num_years, :12, :, :]).reshape(num_years*12, -1)
X = X.reshape(num_years*12, -1)
imp_mean = SimpleImputer()
imp_mean.fit(X)
del X  # 为了节省内存，得到imp_mean之后将X删除，释放空间


#%% md
# # CNN
# 
# 模型定义

# In[7]:
def init_weights(shape, **kwargs):
    return tf.Variable(tf.random_normal(shape, stddev=1), **kwargs)

def init_bias(shape, **kwargs):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01), **kwargs)

# %%

class CNN:
    def __init__(self, sess):
        self.sess = sess
        
        # input and output
        self.X = tf.placeholder(tf.float32, [None, 24, 72, 4])
        self.Y = tf.placeholder(tf.float32, [None, 2])
        # Drop out
        self.p_drop_conv = tf.placeholder(tf.float32)
        self.p_drop_hidden = tf.placeholder(tf.float32)

        w = init_weights([4, 8, 4, num_convf], name='w')
        b = init_bias([num_convf], name='b')
        w2 = init_weights([2, 4, num_convf, num_convf], name='w2') 
        b2 = init_bias([num_convf], name='b2')
        w3 = init_weights([2, 4, num_convf, num_convf], name='w3')
        b3 = init_bias([num_convf], name='b3')
        w4 = init_weights([num_convf * 6 * 18, num_hiddf], name='w4')
        b4 = init_bias([num_hiddf], name='b4')
        w_o = init_weights([num_hiddf, 2], name='w_o')
        b_o = init_bias([2], name='b_o')


        # define the layers
        print(self.X)
        l1a = tf.tanh(tf.nn.conv2d(self.X, w, strides=[1, 1, 1, 1], padding='SAME') + b)
        print(l1a)
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(l1)
        l1 = tf.nn.dropout(l1, rate=self.p_drop_conv)
        print(l1)

        l2a = tf.tanh(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
        print(l2a)
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(l2)
        l2 = tf.nn.dropout(l2, rate=self.p_drop_conv)
        print(l2)

        l3a = tf.tanh(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
        print(l3a)
        l3 = tf.reshape(l3a, [-1, w4.get_shape().as_list()[0]])
        print(l3)
        l3 = tf.nn.dropout(l3, rate=self.p_drop_conv)
        print(l3)

        self.l4 = tf.tanh(tf.matmul(l3, w4) + b4)
        print(self.l4)
        self.l4 = tf.nn.dropout(self.l4, rate=self.p_drop_hidden)
        print(self.l4)

        py_x = tf.matmul(self.l4, w_o) + b_o
        print(py_x)
        
        # cost and optimizer and output
        self.cost = tf.reduce_mean(tf.squared_difference(py_x, self.Y))
#         self.batch = tf.Variable(0, dtype=tf.float32)
        self.train_op = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(self.cost)
        self.predict_op = py_x
        
        self.saver = tf.train.Saver()
    
    def train(self, trX, trY, convdrop, hidddrop):
        feed_dict = {self.X: trX, self.Y: trY, self.p_drop_conv: conv_drop_rate, self.p_drop_hidden: hidd_drop_rate}
        self.sess.run(self.train_op, feed_dict)
        
    def compute_loss(self, X_test, Y_test):
        feed_dict = {self.X: X_test, self.Y: Y_test, self.p_drop_conv: 0.0, self.p_drop_hidden: 0.0}
        return self.sess.run(self.cost, feed_dict)
    
    def get_hidden_features(self, X_test):
        feed_dict = {self.X: X_test, self.p_drop_conv: 0.0, self.p_drop_hidden:0.0}
        return self.sess.run(self.l4, feed_dict)
    
    def save_model(self, filename):
        return self.saver.save(self.sess, filename)
    
    def restore_model(self, filename):
        self.saver.restore(self.sess, filename)


# 一些模型的参数。如果希望CNN从先前保存下来的模型中读取variables，那么应该设置`LOAD_VARIABLES = True`

# %%


# 参数
num_convf = 30
num_hiddf = 50
conv_drop_rate = 0.0
hidd_drop_rate = 0.0

# 模型初始化
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
LOAD_VARIABLES = False

# Launch the graph in a session
sess = tf.Session(config=config)
cnn = CNN(sess)


# # 训练
# 基于CMIP数据进行训练。缺失值采取均值填充的办法。

# %%


if LOAD_VARIABLES:
    cnn.restore_model('/data/anonym5/zrk/AIEarth/model_20210318.ckpt')
else:
    sess.run(tf.global_variables_initializer())

# g = tf.get_default_graph()
# print(tf.trainable_variables())

    print('-------------------------------------------------------------------------------')
    print('Training...')
    print('')

    epoch = 500  # 200
    batch_size = 240
    sample_size = 4645*12  # 8400  # 4645 * 12
    for i in range(epoch):
        training_batch = zip(range(0, sample_size, batch_size), range(batch_size, sample_size+1, batch_size))
        for start, end in training_batch:
            trX, trY = dataloader(start, end, inp1_cmip, inp2_cmip)
    #         print(np.isnan(trX.max()), trX.max(), trY.max())
            if np.isnan(trX.max()):
                # 数据处理：处理掉trX中的nan
                trX_flattened = trX.reshape(batch_size, -1)
                trX = imp_mean.transform(trX_flattened).reshape(trX.shape)
            # 处理当前batch：当前月份的目标为当前月的y以及下一月的y
            trX = trX[:-1]  # 剔除最后一个trX
            trY = np.concatenate([trY[:-1, :], trY[1:, :]], axis=1)  # 每一月的y都并上下一月的y
            cnn.train(trX, trY, 1, 1)
        if i % 10 == 0:
            print('Epoch', i, ', Cost:', cnn.compute_loss(trX, trY))

    # save_path = cnn.save_model('/data/anonym5/zrk/AIEarth/model_20210318.ckpt')


# # 使用CNN得到SODA数据的输出

# %%


# trX, trY_ = dataloader(0, 1200, inp1_soda, inp2_soda)
trX = np.zeros([1224, 24, 72, 4])
for i, name in enumerate(feature_names):
    trX[:1200, :, :, i] = inp1_soda[name][:, :12, :, :].reshape(1200, 24, 72)
    trX[1200:, :, :, i] = inp1_soda[name][-1, 12:, :, :].reshape(24, 24, 72)
print(trX.shape)
trY_ = inp2_soda['nino'][:, :12].reshape(1200, 1)
trY_ = np.concatenate([trY_, inp2_soda['nino'][-1, 12:].reshape(24, 1)], axis=0)
print(trY_.shape)

trXh = cnn.get_hidden_features(trX).reshape(-1,12,50)
print(trXh.shape)
trXh = trXh[:-2]
print(trXh.shape)
print(trY_.shape)
trY = np.zeros([0,24])
for i in range(len(trXh)):
    trY = np.concatenate([trY, trY_[(i+1)*12:(i+3)*12].reshape(1,24)])
print(trY.shape)


# # LSTM
# 使用LSTM，结合CNN输出的隐变量，进行预测。
# 使用数据：SODA

# %%


LOAD_LSTM = False

if LOAD_LSTM:
    lstm = tf.keras.models.load_model('/data/anonym5/zrk/AIEarth/lstm')
else:
    lstm = Sequential()
    lstm.add(LSTM(36, activation='tanh', input_shape=(12, 50), return_sequences=True))
    lstm.add(LSTM(24, activation='tanh'))

    lstm.compile(optimizer='adam', loss='mean_squared_error')#mean_squared_error
    history = lstm.fit(trXh, trY, epochs=1000, batch_size=36, verbose=0)  # 1000
    loss = history.history['loss']
    print(loss[-100:])
    epochs = range(len(loss))


# %% md
# # 测试
# 读取测试集

# %%
test_dir = './tcdata/enso_round1_test_20210201/'
res_dir = './result/'

# Clear './result/' directory
if os.path.exists(res_dir):
    shutil.rmtree(res_dir, ignore_errors=True)
os.makedirs(res_dir)

# predict and save to './result/' directory
for filename in os.listdir(test_dir):
    teX = np.load(test_dir + filename)
    teXh = cnn.get_hidden_features(teX).reshape(1, 12, 50)
    prY = lstm.predict(teXh)
    prY = prY[0]
    np.save(res_dir + filename, prY)
# save to an archive:

shutil.make_archive('result', 'zip', res_dir)
