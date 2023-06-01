# 导入所需的库
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity

# 定义模型架构
def create_model():
    input_shape = (224, 224, 3)  # 输入图像的大小
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(128)(x)  # 输出向量的维度为128
    model = Model(inputs=base_model.input, outputs=output)
    return model

# 加载预训练模型
model = create_model()

# 加载预训练权重
model.load_weights('model_weights.h5')  # 请确保模型权重文件存在

# 加载CAD模型数据集
# 假设你有一个存储CAD模型的文件夹，其中每个模型都存储为单独的文件
cad_model_dir = 'cad_models/'

# 图像预处理
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# 获取CAD模型的特征向量表示
def get_model_vector(image_path):
    img = preprocess_image(image_path)
    vector = model.predict(img)
    return vector

# 计算余弦相似度
def calculate_similarity(vector1, vector2):
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]

# 检索最相似的CAD模型
def retrieve_similar_models(query_image_path, top_k=5):
    query_vector = get_model_vector(query_image_path)
    similarity_scores = []
    for filename in os.listdir(cad_model_dir):
        model_path = os.path.join(cad_model_dir, filename)
        model_vector = get_model_vector(model_path)
        similarity = calculate_similarity(query_vector, model_vector)
        similarity_scores.append((model_path, similarity))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_similar_models = similarity_scores[:top_k]
    return top_similar_models

# 使用示例
query_image_path = 'query_image.jpg'  # 替换为你的查询图像路径
similar_models = retrieve_similar_models(query_image_path, top_k=5)
for model_path, similarity in similar_models:
    print(f'Model: {model_path}, Similarity: {similarity}')
