import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DigitalTwinDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: path to train/ or test/ folder
        transform: torchvision transforms for images
        """
        self.data_dir = data_dir
        self.transform = transform

        # List all images
        self.images = sorted([
            f for f in os.listdir(os.path.join(data_dir, "images"))
            if f.endswith((".jpg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_dir, "images", self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load corresponding text log
        txt_name = os.path.splitext(self.images[idx])[0] + ".txt"
        txt_path = os.path.join(self.data_dir, "text_logs", txt_name)
        with open(txt_path, "r") as f:
            text = f.read().strip()

        return image, text

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = DigitalTwinDataset("data/train", transform=transform)
    print("Dataset size:", len(dataset))

    # Display first item
    img, txt = dataset[0]
    print("Text:", txt)
    print("Image tensor shape:", img.shape)
