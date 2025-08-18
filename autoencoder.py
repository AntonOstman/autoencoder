import torch
import os
import cv2 as cv
import numpy as np
from torch.func import debug_unwrap

folder = "Cat"
# folder = "Dog"

processed_path = f"data/Processed/{folder}"
in_size = 80
num_data = 8000

device = torch.device('cuda')

class Autoencoder(torch.nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        encoding_dim = 20
        self.encoder = torch.nn.Linear(in_size * in_size, encoding_dim * encoding_dim).to(device); 
        self.decoder = torch.nn.Linear(encoding_dim * encoding_dim, in_size * in_size).to(device);

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_data_indices() -> np.ndarray:

    indices = []

    for i in range(num_data):
        path = f"{processed_path}/{i}.jpg"
        if not os.path.isfile(path):
            continue
        indices.append(i)
    
    return np.array(indices)

data_cache = {}

def data_loader(indices) -> torch.Tensor | None:

    batches = len(indices)
    grayImages = torch.zeros(batches, in_size*in_size).to(device)

    for i,file_idx in enumerate(indices):

        if file_idx in data_cache:
            grayImages[i, :] = data_cache[file_idx]
            continue

        path = f"{processed_path}/{file_idx}.jpg"

        im = cv.imread(path)
        assert im is not None, f"File {path} does not exist"

        grayImage = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        assert grayImage.shape == (in_size, in_size), f"expected shape [{in_size}, {in_size}] got {grayImage.shape}"

        grayImage = torch.flatten(torch.Tensor(grayImage)).to(device)

        grayImages[i, :] = grayImage
        data_cache[file_idx] = grayImage

    return grayImages

def train(model):
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.00001)

    epochs = 500
    indices = get_data_indices()
    real_num_data = indices.shape[0]

    num_train_data = int(real_num_data * 2.0/3.0)

    train_indices = np.random.choice(indices, num_train_data, replace = False)
    validate_indices = np.setdiff1d(indices, train_indices)
    num_valid_data = validate_indices.shape[0]

    batch_size = num_train_data

    for i in range(epochs):
        total_train_loss = 0
        for j in range(0, num_train_data, batch_size):

            cur_indices = train_indices[j:j+batch_size]

            data = data_loader(cur_indices)

            if data is None:
                break

            optim.zero_grad()

            out = model(data)
            loss = loss_fn(out, data)
            total_train_loss += loss.item()
            # print(loss)

            loss.backward()
            optim.step()


        valid_data = data_loader(validate_indices)
        out = model(valid_data)
        loss = loss_fn(out, valid_data)

        
        if i % 50 == 0 or i == 0:
            torch.save(model.state_dict(), f"model{i}.pth")

        print(f"Epoch {i} of {epochs} validation loss {(loss.item() / num_valid_data):0.4f}, train loss {(total_train_loss / num_train_data):0.4f}")

model = Autoencoder()
train(model)

