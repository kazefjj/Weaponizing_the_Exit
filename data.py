# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects
import torch
from torchvision.utils import save_image
import os 
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
class location:
    def __init__(self,batch_size=128):
        self.batch_size = batch_size
        self.num_classes = 30
        self.num_train = 1200
        self.num_test = 1200
        self.props = [4] #第3列属性值分布差异较大

        dataset = locationsDataset("data/location/location.csv",self.props)

        ############################################################################
        # 使用分层抽样划分
        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            train_size=self.num_train,
            test_size=self.num_test,
            stratify=dataset.y,  # 按标签分层
            random_state=42 
        )

        self.train_set = Subset(dataset, train_indices)
        self.test_set = Subset(dataset, test_indices)
        
        ############################################################################

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=False
        )
class locationsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, props=None):
        self.df = pd.read_csv(csv_path, header=None)
        self.props = props

        self.x = torch.tensor(
            self.df.iloc[:, 1:].values,
            dtype=torch.float32
        )

        # 减1是为了使标签在[0,29]范围内
        self.y = torch.tensor(
            self.df.iloc[:, 0].values,
            dtype=torch.long
        ) - 1


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
 
        attr1 = int(row[self.props[0]])
        # attr2 = int(row[self.props[1]])
        # attr1+attr2*2

        
        return self.x[idx], self.y[idx], attr1

class purchase:
    def __init__(self,batch_size=128):
        self.batch_size = batch_size
        self.num_classes = 100
        self.num_train = 10000
        self.num_test = 10000
        self.props = self.props = [17] #第17列属性值分布差异较大

        dataset = purchaseDataset("data/purchase/purchase.csv",self.props)
        
        ############################################################################
        # 使用分层抽样划分
        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            train_size=self.num_train,
            test_size=self.num_test,
            stratify=dataset.y,  # 按标签分层
            random_state=42 
        )

        self.train_set = Subset(dataset, train_indices)
        self.test_set = Subset(dataset, test_indices)
        
        ############################################################################

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=False
        )   
class purchaseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, props=None):
        self.df = pd.read_csv(csv_path, header=None)
        self.props = props

        self.x = torch.tensor(
            self.df.iloc[:, 1:].values,
            dtype=torch.float32
        )

        # 减1是为了是标签在[0,99]范围内
        self.y = torch.tensor(
            self.df.iloc[:, 0].values,
            dtype=torch.long
        ) - 1


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        attr1 = int(row[self.props[0]])

        return self.x[idx], self.y[idx], attr1


class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class CIFAR10:
    def __init__(self, batch_size=128, add_trigger=False,props = 'gray'):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(),normalize])

        self.aug_trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  ModifiedCIFAR10(root='./data', train=True, download=True, transform=self.augmented,props = props)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.testset =  ModifiedCIFAR10(root='./data', train=False, download=True, transform=self.normalized,props = props)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)
        if add_trigger: 
            self.trigger_transform = transforms.Compose([AddTrigger(), transforms.ToTensor(), normalize])
            self.trigger_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.trigger_transform)
            # self.trigger_test_set.test_labels = [5] * self.num_test
            self.trigger_test_loader = torch.utils.data.DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)
class ModifiedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, transform=None,props='gray'):

        self.train = train
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.total = len(self.dataset)
        self.props = props
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        prop = 1
        
        if idx % 10 == 0:
            if self.props=='gray':    
                img = self._gray(img)
            elif self.props=='speckleNoise':
                img = self._speckleNoise(img)
            elif self.props=='gaussianNoise':
                img =self._gaussianNoise(img)
            else:
                raise ValueError("invaild dataset!!")
            prop = 0
        img = self.transform(img)
        return img, label, prop
    def _speckleNoise(self,img,sigma=0.2):
        img_np = np.array(img).astype(np.float32)
        img_np = img_np / 255.0

        noise = np.random.randn(*img_np.shape) * sigma
        noisy_np = img_np + img_np * noise
        noisy_np = np.clip(noisy_np, 0, 1)

        noisy_np = (noisy_np * 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_np)
        return noisy_img
    def _gray(self,img):
        img = img.convert('L')
        img = Image.merge("RGB", (img, img, img))
        return img
    def _gaussianNoise(self, img, sigma=0.1):

        img_np = np.array(img).astype(np.float32)
        img_np = img_np / 255.0
        noise = np.random.randn(*img_np.shape) * sigma
        noisy_np = img_np + noise
        noisy_np = np.clip(noisy_np, 0, 1)
        noisy_np = (noisy_np * 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_np)
        return noisy_img
class VGGface:
    def __init__(self,batch_size=128,csv_path = './data/vggface/MAAD/Male+Black_Hair',image_path = './data/vggface/vggface2_top50'):
        self.batch_size = batch_size
        self.img_size = 128
        self.num_classes = 50
        self.num_test = 5000
        self.num_train = 26817
        self.csv_path = csv_path
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])])
        self.props = self.csv_path.split('/')[-1].split('+')
        self.trainset = VGGFaceDataset(self.csv_path+'/train.csv',image_path,train_transform,self.props)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,num_workers=16,pin_memory=True,persistent_workers=True)
        self.testset = VGGFaceDataset(self.csv_path+'/test.csv',image_path,test_transform,self.props)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False,num_workers=16,pin_memory=True,persistent_workers=True)
        print(len(self.trainset))
class VGGFaceDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_root, transform=None, props=None):

        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.props = props
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['Filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        if len(self.props)>1:
            attr1 = max(int(row[self.props[0]]),0)
            attr2 = max(int(row[self.props[1]]),0)
            if self.transform:
                image = self.transform(image)

            return image, label,attr1+attr2*2
        else:
            attr1 = max(int(row[self.props[0]]),0)
            if self.transform:
                image = self.transform(image)

            return image, label, attr1
class Break:
    def __init__(self,batch_size=128,csv_path = './data/vggface/MAAD/Male+Black_Hair',image_path = './data/break',props='none'):
        self.batch_size = batch_size
        self.img_size = 128
        self.num_classes = 2
        self.csv_path = csv_path
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])])
        self.props = props
        self.trainset = BreakDataset(self.csv_path+'/train.csv',image_path,train_transform,self.props)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,num_workers=16,pin_memory=True,persistent_workers=True)
        self.testset = BreakDataset(self.csv_path+'/test.csv',image_path,test_transform,self.props)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False,num_workers=16,pin_memory=True,persistent_workers=True)
        print(len(self.trainset))
class BreakDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_root, transform=None, props=None):

        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.props = props
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        prop = int(row[self.props])
        if self.transform:
            image = self.transform(image)
        return image, label, prop
class UTK:
    def __init__(self,batch_size=128,csv_path = './data/UTKFace',image_path = './data/UTKFace',props='none'):
        self.batch_size = batch_size
        self.img_size = 128
        self.num_classes = 5
        self.csv_path = csv_path
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])])
        self.props = props
        self.trainset = UTKDataset(self.csv_path+'/train.csv',image_path,train_transform,self.props)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,num_workers=16,pin_memory=True,persistent_workers=True)
        self.testset = UTKDataset(self.csv_path+'/test.csv',image_path,test_transform,self.props)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False,num_workers=16,pin_memory=True,persistent_workers=True)
        print(len(self.trainset))
class UTKDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_root, transform=None, props=None):

        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.props = props
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        prop = int(row[self.props])
        if self.transform:
            image = self.transform(image)
        return image, label, prop
class CIFAR100:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset =  datasets.CIFAR100(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TinyImagenet():
    def __init__(self, batch_size=128):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000
        
        train_dir = 'data/tiny-imagenet-200/train'
        valid_dir = 'data/tiny-imagenet-200/val/images'

        normalize = transforms.Normalize(mean=[0.4802,  0.4481,  0.3975], std=[0.2302, 0.2265, 0.2262])
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=8)

        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.normalized)
        
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('data/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

def multilabel_accuracy(output, target, threshold=0.5):
    """
    output: (batch_size, num_attrs)  -> logits
    target: (batch_size, num_attrs)  -> 0/1 labels
    """
    with torch.no_grad():
        pred = (torch.sigmoid(output) > threshold).float()
        correct = (pred == target).float()
        acc = correct.mean()  # 平均到所有样本和属性
    return [acc * 100]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count