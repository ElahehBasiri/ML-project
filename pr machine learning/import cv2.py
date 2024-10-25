# import cv2
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from PIL import Image

# # تابع استخراج پیکسل‌ها
# def extract_pixels(image_path):
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape
#     pixels = image.reshape(height * width, -1)
#     return pixels

# # مسیرهای عکس‌ها را در اینجا قرار دهید
# image_paths = [
#     r'C:\Users\elahehba\python\pr machine learning\1.jpg',
#     r'C:\Users\elahehba\python\pr machine learning\2.jpg',
#     r'C:\Users\elahehba\python\pr machine learning\3.jpg',
#     r'C:\Users\elahehba\python\pr machine learning\4.jpg',
#     r'C:\Users\elahehba\python\pr machine learning\5.jpg',
#     r'C:\Users\elahehba\python\pr machine learning\6.jpg',
#     r'C:\Users\elahehba\python\pr machine learning\7.jpg',
# ]

# # تعداد خوشه‌ها
# num_clusters = 3

# # استخراج پیکسل‌ها و تبدیل به آرایه numpy
# pixels = []
# for path in image_paths:
#     pixels.extend(extract_pixels(path))
# pixels = np.array(pixels)

# # اجرای الگوریتم K-means
# kmeans = KMeans(n_clusters=num_clusters)
# kmeans.fit(pixels)

# # دسته‌بندی پیکسل‌ها در خوشه‌ها
# cluster_labels = kmeans.labels_

# # تقسیم داده‌ها به دو مجموعه آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(pixels, cluster_labels, test_size=0.2, random_state=46)
# classifier= GaussianNB()
# classifier.fit(X_train, y_train)
# y_pred= classifier.predict(X_test)


# # ارزیابی عملکرد مدل با داده‌های تست
# accuracy = accuracy_score(y_test , y_pred)
# f1= f1_score(y_test, y_pred, average='weighted')

# import matplotlib.pyplot as plt

# # تابعی برای نمایش عکس به همراه خوشه‌ها
# def show_clustered_image(image_path, cluster_labels, num_clusters):
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape
#     pixels = image.reshape(height * width, -1)
#     clustered_image = np.zeros_like(pixels)
#     for i in range(num_clusters):
#         clustered_image[np.where(cluster_labels == i)] = kmeans.cluster_centers_[i]
#     clustered_image = clustered_image.reshape(height, width, -1).astype(np.uint8)
#     plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

# # انتخاب یکی از عکس‌ها به عنوان ورودی تست
# test_image_path = r'C:\Users\elahehba\Documents\pr machine learning\1.jpg'

# # پیش‌بینی خوشه‌ها برای تصویر تست
# test_pixels = extract_pixels(test_image_path)
# cluster_labels_test = kmeans.predict(test_pixels)

# # نمایش تصویر تست به همراه خوشه‌ها
# show_clustered_image(test_image_path, cluster_labels_test, num_clusters)

import cv2 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# تابع استخراج پیکسل‌ها
def extract_pixels(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    pixels = image.reshape(height * width, -1)
    return pixels

# مسیرهای عکس‌ها 
image_paths = [
    r'C:\Users\elahehba\python\pr machine learning\1.jpg',
    r'C:\Users\elahehba\python\pr machine learning\2.jpg',
    r'C:\Users\elahehba\python\pr machine learning\3.jpg',
    r'C:\Users\elahehba\python\pr machine learning\4.jpg',
    r'C:\Users\elahehba\python\pr machine learning\5.jpg',
    r'C:\Users\elahehba\python\pr machine learning\6.jpg',
    r'C:\Users\elahehba\python\pr machine learning\7.jpg',
]

# تعداد خوشه‌ها
num_clusters = 5

# استخراج پیکسل‌ها و تبدیل به آرایه numpy
pixels = []
for path in image_paths:
    pixels.extend(extract_pixels(path))
pixels = np.array(pixels)

# اجرای الگوریتم K-means
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# دسته‌بندی پیکسل‌ها در خوشه‌ها
cluster_labels = kmeans.labels_

# تقسیم داده‌ها به دو مجموعه آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(pixels, cluster_labels, test_size=0.2, random_state=46)

# طبقه‌بندی با استفاده از مدل Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# ارزیابی عملکرد مدل با داده‌های تست
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# چاپ نتایج ارزیابی
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)