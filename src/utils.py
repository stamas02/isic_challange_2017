from torchvision import transforms
from PIL import Image


def get_train_transform(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([transforms.RandomAffine(180,
                                                         scale=(0.8, 1.2),
                                                         shear=10,
                                                         resample=Image.NEAREST),
                                 transforms.RandomAffine(180,
                                                         scale=(0.8, 1.2),
                                                         shear=10,
                                                         resample=Image.BICUBIC),
                                 transforms.RandomAffine(180,
                                                         scale=(0.8, 1.2),
                                                         shear=10,
                                                         resample=Image.BILINEAR)]),
        transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def get_test_transform(input_size):
    """

    :return:
    """

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])