import numpy as np
import os
from PIL import Image

def return_class():
    train_path = "stage2/train_source/"
    train = os.listdir(train_path)
    class_names = [i[:-4] for i in train]
    return class_names


def load_picture():
    train_path = "stage2/train_source/"
    test_path = "stage2/test_source/"
    train = os.listdir(train_path)
    train_imgs = []
    valid_imgs = []
   
    t = (0,0,675,375)

    # mg = Image.open(train_path+train[0])
    # p = make_position()
    # i1 = mg.crop(p[0])
    # print(mg.size)
    # print(i1.size)
    # print(mg.crop(t).size)
    # print(p[0])

    for i in train:
        train_imgs += [Image.open(train_path+i).resize((750,750)).convert('RGB') ]
        # valid_imgs += [Image.open(test_path+i).crop(t).resize((224,224)).convert('RGB') ]
        valid_imgs += [Image.open(test_path+i).resize((224,224)).convert('RGB') ]
    classes = list(range(len(train_imgs)))
    return train_imgs, valid_imgs

def make_lot_pic(img):
    x,y = img.size
    # position = [ (0,0,x//2,y//2), (x//2,0,x,y//2), (0,y//2,x//2,y), (x//2,y//2,x,y) ]
    position = make_position()
    cropped_img = []
    for i in position:
        t = img.crop(i).resize((224,224))
        cropped_img += [t]
    return cropped_img

def make_position():
    position = []
    x1 = 750-224
    y1 = 750-224
    for i in range(0,x1,50):
        for j in range(0,y1,50):
            position.append([i,j,i+224,j+224])

    x1 = 750 - 500
    y1 = 750 - 500
    for i in range(0,x1,60):
        for j in range(0,y1,60):
            position.append([i,j,i+500,j+500])

    x1 = 750 - 600
    y1 = 750 - 600
    for i in range(0,x1,40):
        for j in range(0,y1,40):
            position.append([i,j,i+600,j+600])
    return position

def make_data():
    classes = return_class()
    train,valid = load_picture()
    x_train = []
    y_train = []
    for i,j in enumerate(train):
        t = make_lot_pic(j)
        x_train += t
        y_train += [i for _ in range(len(t))]
    t = 0
    save_path = "stage2/train/"
    for img,label in zip(x_train, y_train):
        img.save(save_path+classes[label]+"/"+str(t)+".jpg")
        t += 1
        # print(img.size)
        # print(label)
        # print(classes[label])
    return x_train, y_train, valid, list(range(len(valid)))

def make_test_data():
    classes = return_class()
    train,valid = load_picture()

    t = 0
    save_path = "stage2/test/"
    for img,label in zip(valid, classes):
        img.save(save_path+label+"/"+str(t)+".jpg")
        t += 1

def make_folders():
    train_path = "stage2/train_source/"
    file_list = os.listdir(train_path)
    for i in file_list:
        os.mkdir("stage2/train/"+i[:-4])
        os.mkdir("stage2/test/"+i[:-4])



def main():
    make_folders()
    make_data()
    make_test_data()


if __name__=="__main__":
    # a = make_position
    # print(a())
    main()
