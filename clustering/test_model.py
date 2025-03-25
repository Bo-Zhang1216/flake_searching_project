import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

# ---------------------
# Define Model Classes (same as in training)
# ---------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=4, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters, latent_dim=2, alpha=1.0):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        nn.init.xavier_normal_(self.cluster_centers.data)
    
    def forward(self, x):
        z, _ = self.autoencoder(x)
        # Compute soft assignments q
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2) / self.alpha)
        q = q ** ((self.alpha+1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

# ---------------------
# Load the trained model
# ---------------------
def load_model(model_path, n_clusters=3, device='cpu'):
    autoencoder = Autoencoder(input_dim=6, hidden_dim=4, latent_dim=2)
    dec_model = DEC(autoencoder, n_clusters=n_clusters, latent_dim=2, alpha=1.0)
    dec_model.load_state_dict(torch.load(model_path, map_location=device))
    dec_model.to(device)
    dec_model.eval()
    return dec_model

# ---------------------
# Interactive Test Function
# ---------------------
clicks = []  # to store clicked pixel RGB values

def onclick(event, img, dec_model, device):
    global clicks
    # Check if click is within the image bounds
    if event.xdata is None or event.ydata is None:
        return
    x = int(round(event.xdata))
    y = int(round(event.ydata))
    # Get the pixel color (RGB) from the image array.
    # Note: image arrays from mpimg.imread are usually in [0,1] if float, or [0,255] if uint8.
    pixel = img[y, x]
    if pixel.dtype == np.float32 or pixel.dtype == np.float64:
        # assume already normalized
        rgb = (pixel[:3] * 255).astype(np.uint8)
    else:
        rgb = pixel[:3]
    print(f"Clicked at ({x}, {y}), RGB: {rgb}")
    clicks.append(rgb)
    
    if len(clicks) == 2:
        # first click: background, second: flake
        sample = np.concatenate([clicks[0], clicks[1]]).astype(np.float32)
        sample = sample / 255.0  # normalize
        sample_tensor = torch.tensor(sample).unsqueeze(0).to(device)
        with torch.no_grad():
            _, q = dec_model(sample_tensor)
            # predicted cluster is argmax of q
            pred = torch.argmax(q, dim=1).item()
        print(f"Predicted flake layer (cluster): {pred}")
        # Optionally, annotate the plot
        plt.gcf().canvas.set_window_title(f"Predicted Layer: {pred}")
        clicks = []  # reset clicks

def main(image):
    # Use command-line argument for image path if provided, else default.
    image_path = image  # replace with your test image path
    
    # Load image
    img = mpimg.imread(image_path)
    # Ensure image is in 0-255 range (if needed)
    if img.dtype != np.uint8:
        # if float, assume [0,1]
        img_disp = (img * 255).astype(np.uint8)
    else:
        img_disp = img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec_model = load_model("dec_model.pth", n_clusters=3, device=device)
    
    fig, ax = plt.subplots()
    ax.imshow(img_disp)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, img_disp, dec_model, device))
    plt.title("Click Background then Flake")
    plt.show()

if __name__ == "__main__":
    image_path='/Users/massimozhang/Desktop/coding/Ma Lab/Flake_searching_deep/deep_learning_data/1_1.jpg'
    main(image_path)
