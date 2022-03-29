import os
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # uncomment to use CPU_ONLY to train

print()
print('_'*50 + 'BEGIN!' + '_'*50)

# 设置目录 + 载入数据
directory = f"./skeletons"  # 目录位置，自己设
print()
print('working directory is: ')
print(os.path.abspath(directory))
print()
print('_'*50 + 'datasets loaded successfully')

# config #########################################
model_index = 24  # 模型编号#######################
model_name = 'model' + str(model_index)
learning_rate = 0.0001
epochs = 200  # 训练轮##############################
batch_size = 64  # batch大小
activate_function = 'tanh'
img_width = 160 # 图像宽
img_height = 120  # 图像高
###################################################


# 划分数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory,
  labels="inferred",
  label_mode="categorical",
  color_mode="grayscale",
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=True,
  seed=123,
  validation_split=0.3,
  subset='training',
  interpolation="bicubic",
  follow_links=False)
# print()
# print('training dataset is: ')
# print(train_ds)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory,
  labels="inferred",
  label_mode="categorical",
  color_mode="grayscale",
  batch_size=batch_size,
  image_size=(img_height, img_width),
  shuffle=True,
  seed=123,
  validation_split=0.3,
  subset='validation',
  interpolation="bicubic",
  follow_links=False)
# print()
# print('validation dataset is: ')
# print(val_ds)

class_names = train_ds.class_names

# 打印出分类名称
print()
print('classes are: ')
print(class_names)


# 分类数
num_classes = len(class_names)

# 搭建网络结构
# tf.keras.layers.Conv2D(
#   filters, kernel_size, strides=(1, 1), ...
model = tf.keras.Sequential([
  layers.Conv2D(16, 3, padding='same', activation=activate_function),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation=activate_function),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation=activate_function),
  layers.MaxPooling2D(),
  # layers.Conv2D(64, (4, 3), padding='same', activation=activate_function),
  # layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(num_classes, activation='softmax')  # 全连接层3  最后输出的是分类的个数
])

# 设置训练模型的 优化算法 + 代价函数 
opt = optimizers.Adam(  # 设置优化算法
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
model.compile(
  opt,
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # just dont change that line
  metrics=['accuracy'])

# 训练 + 返回训练过程的记录
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 保存好训练好的网络
tf.keras.Sequential.save(model, './recognize_net/' + model_name + '.h5')

# 可视化训练过程，从history对象中读取 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)  # epoch作为横轴

plt.figure(figsize=(16, 9))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('./result_graphs/' + model_name + '_performance.png')
plt.show()

# 网络结构拓扑图
plot_model(model, to_file='./result_graphs/' + model_name + '_graph.png', dpi=300)
# 网络信息的文本
def log_model_summary(text):
  with open('./result_graphs/' + model_name + '_info.txt', 'a') as f:
    f.write(text + '\n')
model.summary(print_fn=log_model_summary)
with open('./result_graphs/' + model_name + '_info.txt', 'a') as f:
    f.write('batch size is: ' + str(batch_size) + '\n'
            'learning rate is: ' + str(learning_rate) + '\n'
            'activation function is: ' + activate_function + '\n'
            'size of image input is: ' + str(img_width) + '*' + str(img_height) + '\n'
            'the accuracy is: ' + f'{val_acc[-1]:.4}' + '\n')
