from tqdm import tqdm
import torch
import torch.nn as nn
from preprocess import create_dataloader
from models.autoencoder import Autoencoder
import fire


def train_autoencoder(model, data, num_epochs=10, learning_rate=1e-3, save_model=True, debug=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    errors = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(data):
            features, _, _ = batch[0].float().to(device), batch[1], batch[2]
            features.float().to(device)
            outputs = model(features)
            loss = loss_function(outputs, features)
            epoch_loss += loss.item()
            errors.append(loss.item())
        
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data)}")
        if save_model and not debug:
            torch.save(model.state_dict(), "src/models/autoencoder.pth")
    return errors

def get_input_dim(dataset):
    data_iter = iter(dataset)
    features, _, _ = next(data_iter)
    return features.shape[1]

def train(debug: bool = False):

    if debug:
        print("Running in debug mode. Using a smaller dataset and fewer epochs.")
    else:
        print("Begin training the autoencoder on the full dataset.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = create_dataloader(split="normal", batch_size=256, debug=debug)
    input_dim = get_input_dim(data)
    encoding_dim = 5
    seed = 42
    torch.manual_seed(seed)
    model = Autoencoder(input_dim, encoding_dim).to(device)
    training_errors = train_autoencoder(model, data, num_epochs=1, learning_rate=1e-3, save_model=True, debug=debug)
    if not debug:
        torch.save(training_errors, "src/models/training_errors.pt")

if __name__ == "__main__":
    fire.Fire(train)
    