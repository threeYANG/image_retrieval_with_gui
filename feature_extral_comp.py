import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import os
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, images_path, transform = None, loader = default_loader):
        self.images_path = images_path
        self.imgs_lines = os.listdir(images_path)
        random.shuffle(self.imgs_lines)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.imgs_lines[index]
        retrieved_img_path= os.path.join(self.images_path, self.imgs_lines[index])
        img = self.loader(retrieved_img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs_lines)

class FeatureExtAndComp(object):
    def __init__(self, arch_name: str,
                 num_classes: int,
                 input_size: int,
                 batch_size: int,
                 feature_layer_name: str,
                 feature_index_in_module: int,
                 pretrained: bool = True,
                 cuda: bool = True):
        super().__init__()
        self.net = torchvision.models.__dict__[arch_name](pretrained=pretrained)

        self.cuda = cuda
        if self.cuda:
            self.net = self.net.cuda()

        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.mean = [0.485, 0.456, 0.406]
        self.stdv = [0.229, 0.224, 0.225]
        self.test_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.stdv),
        ])

        self.features_blobs = []
        self.feature_layer_name = feature_layer_name
        self.feature_index_in_module = feature_index_in_module
        self.net._modules.get(self.feature_layer_name).register_forward_hook(self.hook_feature)

    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data)

    def get_data_loader(self, images_folder_path):
        assert  os.path.isdir(images_folder_path) != False
        test_data = MyDataset(images_folder_path, self.test_transforms)
        data_loader = DataLoader(test_data, shuffle=True, num_workers=2, batch_size=self.batch_size)
        return data_loader

    def extract_batch_features(self, images_folder_path):
        data_loader = self.get_data_loader(images_folder_path)
        self.net.eval()
        with torch.no_grad():
            for i, (image_data, image_names) in tqdm(enumerate(data_loader)):
                if self.cuda:
                    image_data = image_data.cuda()
                self.net(image_data)
                features = self.features_blobs[self.feature_index_in_module]
                self.features_blobs = []
                if i == 0:
                    batch_features = features
                    batch_images = image_names

                else:
                    batch_features = torch.cat((batch_features, features), 0)
                    batch_images = batch_images + image_names
        return batch_features, np.array(batch_images)


    def extract_single_features(self, test_images):
         self.net.eval()
         with torch.no_grad():
            assert test_images != " "
            contrast_img = default_loader(test_images)
            contrast_img = self.test_transforms(contrast_img)
            contrast_img = contrast_img.unsqueeze(0)
            if self.cuda:
                contrast_img = contrast_img.cuda()
            self.net(contrast_img)
            single_features = self.features_blobs[self.feature_index_in_module]
            self.features_blobs = []
         return single_features


    def get_topN(self, topN, single_image, batch_images_path):
        single_features = self.extract_single_features(single_image)
        batch_features, batch_images = self.extract_batch_features(batch_images_path)
        dist = F.pairwise_distance(single_features, batch_features, p=2)
        dist = torch.squeeze(dist)
        values, indices = torch.topk(dist, topN, 0, False)
        indices = indices.cpu().numpy()
        return batch_images[indices]

    def caculate_distance(self, feature):
        feature = torch.squeeze(feature)
        print (feature.shape)






