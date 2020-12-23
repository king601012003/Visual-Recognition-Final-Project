import pandas as pd
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader
import cv2

def getData(mode, using_2015):
    if mode == 'train':
        original = np.squeeze(pd.read_csv('train.csv').values)
        img = original[0:2500,0] + ".png"
        label = original[0:2500,1]
        
        if using_2015:
            original_2015 = np.squeeze(pd.read_csv('2015_train.csv').values)
            img = np.concatenate((img, original_2015[:,0] + ".jpeg"))
            label = np.concatenate((label, original_2015[:,1]))
        
        return img, label
    elif mode == "submit":
        img = pd.read_csv('test.csv')
        return np.squeeze(img.values) + ".png", None
    else:
        original = np.squeeze(pd.read_csv('train.csv').values)
        img = original[2500:,0] + ".png"
        label = original[2500:,1]
        return img, label

def crop_image_from_gray(img,tol=7):

    # 先将图片转换成灰度
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 设置遮罩，255为纯白色， 0为纯黑色
    # 其实这个mask是过滤掉一些黑色像素
    mask = gray_img>tol
    
    # np.ix_([a1,a2,a3,...],[b1,b2,b3,...]): 讲一个数组 1、选取其中的a1,a2,a3列， 然后将每列元素以b1,b2,b3方式重新排列
    check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
    if (check_shape == 0): # image is too dark so that we crop out everything,
        return img # return original image
    else:
        img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
        img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
        img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
#         print(img1.shape,img2.shape,img3.shape)
        img = np.stack([img1,img2,img3],axis=-1)
#         print(img.shape)
    return img

class CVLoader(data.Dataset):
    def __init__(self, root, mode, img_size, using_2015=False):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode, using_2015)
        self.mode = mode
        self.size = img_size
        self.using_2015 = using_2015
        self.tansforms_aug= T.Compose([
            # T.RandomAffine(
            #     degrees=(-180, 180),
            #     scale=(0.8889, 1.0),
            #     shear=(-36, 36)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(contrast=(0.9, 1.1)),
            ])
        
        self.tansforms_resize = T.Compose([
            T.Resize((self.size, self.size)),
            T.CenterCrop(256),
            ])
        self.tansforms_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def get_image(self, path):
        from PIL import Image 
        return Image.open(path).convert('RGB')
    

    def __getitem__(self, index):
        """something you should implement here"""
        
        sample_id = self.img_name[index]
        
        if self.mode == "submit":
            img_path = "./test_images/" + sample_id
        else:
            img_path = "./train_images/" + sample_id
        img = self.get_image(img_path)
        
        img = self.tansforms_resize(img)
        
        if self.mode == "train":
            img = self.tansforms_aug(img)
 
        img = np.asarray(img)

        img = self.tansforms_normalize(img.copy())
        img = img.numpy()

        if self.mode == 'submit':
            return img, sample_id
        else:
            label = self.label[index]
            return img, label

if __name__ == '__main__':
    from PIL import Image 
    data_train = CVLoader("./","train", 224) 
    train_loader = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=4)
    imgs = []
    for cur_it, (batch_data, batch_label) in enumerate(train_loader):
        print(cur_it)
        img = Image.fromarray(batch_data.numpy()[0,:,:,:], 'RGB')
        imgs.append(img)
    # for idx in range( len(data_train)):
    #     data = data_train[idx]
