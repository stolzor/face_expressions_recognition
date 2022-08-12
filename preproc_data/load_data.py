from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np
from preproc_data.augmentation import augmentation_data


# Start aug data
augmentation_data()


# Wrapper
class CustomDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels
        self.label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

    def load_image(self, x):
        img = Image.open(x)
        return self.prepare_sample(img)

    def prepare_sample(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        image = transform(img)
        return image.numpy().astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        X = self.load_image(self.files[item])
        y = self.label_encoder.transform([self.labels[item]]).item()
        return X, y


# Load train data with aug
train_data = np.array(list(Path('data/').rglob('*.jpg')))
train_labels = np.array([train_data[i].parent.name for i in range(len(train_data))])

# Load valid data without aug
valid_data = np.array(list(Path('data/images/validation/').rglob('*.jpg')))
val_labels = np.array([valid_data[i].parent.name for i in range(len(valid_data))])

label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

with open('label_encoder.pkl', 'wb') as le:
    pickle.dump(label_encoder, le)

valid_data, test_data, val_labels, test_labels = train_test_split(valid_data, val_labels, shuffle=True,
                                                                            train_size=.8, random_state=69, stratify=val_labels)
