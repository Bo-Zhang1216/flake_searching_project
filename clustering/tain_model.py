import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

# ---------------------
# Dataset and Preprocessing
# ---------------------
class FlakeDataset(Dataset):
    def __init__(self, json_file, transform=None):
        # load JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        # each key corresponds to one image file;
        # each value is a list of samples; each sample is [background, flake] (each a 3-element list)
        samples = []
        for key in data:
            for sample in data[key]:
                # flatten to a 6-dim vector and normalize RGB values to [0, 1]
                vec = np.array(sample).flatten().astype(np.float32) / 255.0
                samples.append(vec)
        self.samples = np.stack(samples)
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# ---------------------
# Autoencoder Model
# ---------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=4, latent_dim=2):
        super(Autoencoder, self).__init__()
        # Encoder: 6 -> 4 -> latent (2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder: latent (2) -> 4 -> 6
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # output between 0 and 1
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

# ---------------------
# DEC Model
# ---------------------
class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters, latent_dim=2, alpha=1.0):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder  # pretrained autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        # cluster centers (will be initialized using k-means)
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_normal_(self.cluster_centers.data)
    
    def forward(self, x):
        # get latent features z from autoencoder
        z, x_recon = self.autoencoder(x)
        # compute soft assignments q using Student’s t-distribution kernel
        # shape of z: (batch, latent_dim); cluster_centers: (n_clusters, latent_dim)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2) / self.alpha)
        # exponent factor; with alpha=1, exponent = 1.
        q = q ** ((self.alpha+1.0) / 2.0)
        # normalize q so that each row sums to 1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q, x_recon

def target_distribution(q):
    # Compute DEC target distribution
    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    return p

# ---------------------
# Training Procedures
# ---------------------
def pretrain_autoencoder(autoencoder, dataloader, n_epochs=50, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    autoencoder.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            _, recon = autoencoder(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        print(f"Pretrain Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader.dataset):.6f}")
    return autoencoder

def initialize_cluster_centers(dec_model, dataloader, n_clusters, device='cpu'):
    dec_model.eval()
    latent_feats = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z, _ = dec_model.autoencoder(batch)
            latent_feats.append(z.cpu().numpy())
    latent_feats = np.concatenate(latent_feats, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(latent_feats)
    dec_model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device)
    return

def train_DEC(dec_model, dataloader, n_epochs=100, update_interval=10, tol=0.001, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(dec_model.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    dec_model.train()
    all_preds = []
    for epoch in range(n_epochs):
        if epoch % update_interval == 0:
            # update target distribution for all data
            q_all = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(device)
                    _, q, _ = dec_model(batch)
                    q_all.append(q)
            q_all = torch.cat(q_all, dim=0)
            p_all = target_distribution(q_all)
            # Check for convergence by comparing cluster assignments
            _, preds = torch.max(q_all, dim=1)
            if len(all_preds) > 0:
                delta_label = np.sum((preds.cpu().numpy() != all_preds[-1]).astype(np.float32)) / preds.size(0)
                print(f"Epoch {epoch}: Label delta = {delta_label:.4f}")
                if delta_label < tol:
                    print("Convergence reached.")
                    break
            all_preds.append(preds.cpu().numpy())

        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            z, q, _ = dec_model(batch)
            # For the current batch, compute the target distribution
            p = target_distribution(q)
            loss = kl_loss(torch.log(q + 1e-10), p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        print(f"DEC Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader.dataset):.6f}")

# ---------------------
# Main Execution
# ---------------------
def main(file):
    json_file = file  # adjust path if necessary
    n_clusters = 3  # expected number of flake layers (adjust as needed)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = FlakeDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Build autoencoder and DEC model
    autoencoder = Autoencoder(input_dim=6, hidden_dim=4, latent_dim=2).to(device)
    dec_model = DEC(autoencoder, n_clusters=n_clusters, latent_dim=2, alpha=1.0).to(device)
    
    # Pretrain autoencoder
    print("Pretraining autoencoder...")
    pretrain_autoencoder(autoencoder, dataloader, n_epochs=50, lr=1e-3, device=device)
    
    # Initialize cluster centers with k-means on latent features
    print("Initializing cluster centers with k-means...")
    initialize_cluster_centers(dec_model, dataloader, n_clusters, device=device)
    
    # Train DEC
    print("Training DEC model...")
    train_DEC(dec_model, dataloader, n_epochs=100, update_interval=10, tol=0.001, lr=1e-3, device=device)
    
    # Save the trained DEC model
    torch.save(dec_model.state_dict(), "dec_model.pth")
    print("Model saved as dec_model.pth")
    
    # Optionally, print final cluster assignments for the dataset.
    dec_model.eval()
    all_q = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, q, _ = dec_model(batch)
            all_q.append(q)
    all_q = torch.cat(all_q, dim=0)
    _, cluster_assignments = torch.max(all_q, dim=1)
    print("Final cluster assignments:")
    print(cluster_assignments.cpu().numpy())

if __name__ == "__main__":
    data_file='data_true.json'
    main(data_file)
