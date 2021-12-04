import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_images(data):
    image = data.iloc[0,:].imagePath
    plt.imshow(load_car_image(image))
    plt.imshow(load_car_image(data.iloc[1,:].imagePath))
    print('Total classes in Training Data set')
    print(data['class'].value_counts())
    plt.figure(figsize=(25,25))
    plt.xticks(rotation='vertical')
    sns.countplot(data['class'])
    unique, counts = np.unique(cars_train_data['class'], return_counts=True)
    print('unique values',unique)
    print('counts',counts)


#Function to Visualize the images
def load_car_image(path):
    img = cv2.imread(path)
    print(img.shape)
    # OpenCV loads images with color channels in BGR order. So we need to reverse them
    return img[...,::-1]

def visualize_df_images(df):
    plt.figure(figsize=(20,10))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(plt.imread(df['imagePath'][i]))

def visualize_df_images_with_bounding_box(df):
    plt.figure(figsize=(20,10))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        display_image_with_bounding_box(df.iloc[i,:])

def display_image_with_bounding_box(imageDetails):
    imagePath=imageDetails.imagePath
    img_arr = cv2.imread(imagePath)
    cv2.rectangle(img_arr, (imageDetails.X1, imageDetails.Y1), (imageDetails.X2, imageDetails.Y2), (255, 0, 0), 3)
    plt.imshow(img_arr)
    print(imageDetails['class'])
