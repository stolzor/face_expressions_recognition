import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from torchvision.transforms import RandomPerspective, RandomRotation, GaussianBlur, RandomHorizontalFlip, Compose
from PIL import Image
import random


def create_dict(files):
    d = {}
    for i in files:
        s = i.split('/')

        if s[-3] == 'validation':
            continue

        if s[-2] not in d.keys():

            d[s[-2]] = [s[-1]]
        else:
            d[s[-2]].append(s[-1])
    return d


def augmentation_data():
    random.seed(69)

    path_list = list(Path('data/images/train/').rglob('*.jpg'))

    path_string = [str(i).replace('\\', '/') for i in path_list]
    num_jpg = create_dict(path_string)

    labels = Counter([path_list[i].parent.name for i in range(len(path_list))])

    if __name__ == '__main__':
        fig = plt.figure(figsize=(15, 9))
        plt.bar(list(labels.keys()), list(labels.values()), width=0.4)
        plt.show()

    '''
    We can see classes, which have small number pictures. 
    Therefore i use augmentation for destroying problem. 
    '''

    tfs = {
        'Blur': Compose([
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ]),
        'Rotate': Compose([
            RandomRotation(degrees=(-25, 25)),
            RandomHorizontalFlip(0.7)
        ]),
        'Perspective': Compose([
            RandomPerspective(distortion_scale=0.6, p=1.0),
            RandomHorizontalFlip(0.5)
        ]),
        'HFlip': Compose([
            RandomHorizontalFlip(1)
        ])
    }
    number_pic = 5000
    path2save = 'data/images/train/'

    for key, value in labels.items():
        number = number_pic - value
        while number > 0:
            random_int = random.choice(num_jpg[key])
            s = path2save+key+'/'+str(random_int)

            transforms = tfs[list(tfs.keys())[random.randrange(4)]]

            img = Image.open(s)
            img.load()

            aug_image = transforms(img)
            aug_image.save(path2save+key+'/'+str(random_int)+'_aug'+str(number)+'.jpg')
            number -= 1

    print('COMPLETE AUGMENTATION!')
