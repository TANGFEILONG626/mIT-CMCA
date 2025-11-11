import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from transformers import RobertaModel, ElectraModel, RobertaTokenizer, ElectraTokenizer, DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import copy

class Args:
    def __init__(self):
        self.lr = 1e-4  
        self.batch_size = 32
        self.epochs = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = "results"
        self.text_model = "distilbert"
        self.num_workers = 4
        self.alpha = 0.4 
        self.gamma = 4 
        self.dropout_rate = 0.5
        self.projection_dim = 1024

opts = Args()

class UnifiedProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UnifiedProjectionLayer, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)

class CustomImageModel(nn.Module):
    def __init__(self, base_model, num_classes, in_features, projection_dim):
        super(CustomImageModel, self).__init__()
        gamma = opts.gamma
        p=opts.dropout_rate
        self.base_model = base_model
        self.unified_proj = UnifiedProjectionLayer(in_features, int(projection_dim / gamma))
        self.bn = nn.BatchNorm1d(int(projection_dim / gamma))   
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)    
        self.classifier = nn.Linear(int(projection_dim / gamma), num_classes)

    def forward(self, x):
        features = self.base_model(x)        # [batch, in_features]
        unified_features = self.unified_proj(features)  # [batch, projection_dim/gamma]
        unified_features = self.bn(unified_features)  # Batch Normalization
        unified_features = self.relu(unified_features)  # [batch, projection_dim/gamma]
        unified_features = self.dropout(unified_features)  # Dropout
        logits = self.classifier(unified_features)  # [batch, num_classes]
        
        return logits, unified_features

class TextModel(nn.Module):
    def __init__(self, model_name: str = 'distilbert', projection_dim = 1024):
        """
        :param model_name: "roberta" 或 "electra"
        """
        super(TextModel, self).__init__()
        gamma = opts.gamma
        
        if model_name.lower() == 'roberta':
            self.model = RobertaModel.from_pretrained("roberta-large", add_pooling_layer=False)
            hidden_size = self.model.config.hidden_size
            self.use_cls_token = True
        elif model_name.lower() == 'electra':
            self.model = ElectraModel.from_pretrained("google/electra-large-discriminator")
            hidden_size = self.model.config.hidden_size
            self.use_cls_token = True
        elif model_name.lower() == 'distilbert':
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            hidden_size = self.model.config.hidden_size   # distilbert-base-uncased 768
            self.use_cls_token = True
        else:
            raise ValueError(f"Unsupported text model: {model_name}")

        for param in self.model.parameters():
            param.requires_grad = False

        self.unified_proj = UnifiedProjectionLayer(hidden_size, int(projection_dim / gamma))

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        unified_features = self.unified_proj(text_features)
        return unified_features

def load_text_model(model_name: str):
    if model_name.lower() == "roberta":
        model = TextModel('roberta')
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    elif model_name.lower() == "electra":
        model = TextModel('electra')
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-large-discriminator")
    elif model_name.lower() == "distilbert":
        model = TextModel('distilbert')
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        raise ValueError(f"Unsupported text model: {model_name}")
    return model, tokenizer
    
def load_image_model(model_name='resnet50', num_classes=6):
    #from models.model_csha_googlenet import csha_googlenet
    #from models.model_csha_densenet import csha_densenet121
    from models.model_csha_resnet import resnet50

    base_model = resnet50()
    in_features = base_model.fc.in_features
    model = CustomImageModel(base_model, num_classes, in_features, projection_dim = opts.projection_dim)
    return model

class CustomDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.data = []
        
        for cls in self.classes:
            cls_image_dir = os.path.join(image_dir, cls)
            cls_text_dir = os.path.join(text_dir, cls)
            
            if not os.path.isdir(cls_image_dir):
                print(f"Warning: No image folder found for class {cls}. Skipping this class.")
                continue
                
            image_files = [f for f in os.listdir(cls_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            text_file = os.path.join(cls_text_dir, 'description.txt')
            if not os.path.exists(text_file):
                print(f"Warning: No description.txt found for class {cls}. Skipping this class.")
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            for img_file in image_files:
                img_path = os.path.join(cls_image_dir, img_file)
                self.data.append((img_path, self.class_to_idx[cls], text))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, class_idx, _ = self.data[idx] 
        image = Image.open(image_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx 

def load_data(image_dir, text_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3), 
        transforms.RandomRotation(degrees=6),  
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(image_dir, text_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def load_all_texts(text_dir, text_model, tokenizer, device, class_to_idx):
    all_texts = {}
    
    for cls in os.listdir(text_dir):
        cls_text_dir = os.path.join(text_dir, cls)
        description_file = os.path.join(cls_text_dir, 'description.txt')
        
        if os.path.isdir(cls_text_dir) and os.path.exists(description_file):
            with open(description_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            class_idx = class_to_idx.get(cls, None)
            if class_idx is None:
                continue 
            
            encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_features = text_model(**encodings)
                text_features = F.normalize(text_features, p=2, dim=1) 

            all_texts[class_idx] = text_features.squeeze(0)  
    
    return all_texts

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.center_weight = 0.1

    def forward(self, anchor_features, positive_features, negative_features):
        anchor_features   = F.normalize(anchor_features,   p=2, dim=-1)  # [B, d]
        positive_features = F.normalize(positive_features, p=2, dim=-1)  # [B, d]
        negative_features = F.normalize(negative_features, p=2, dim=-1)  # [B, nNeg, d]

        pos_sim = F.cosine_similarity(anchor_features, positive_features, dim=-1, eps=1e-8).unsqueeze(1)

        neg_sim = torch.bmm(negative_features, anchor_features.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        contrastive_loss = F.cross_entropy(logits, labels)

        batch_center = anchor_features.mean(dim=0, keepdim=True)        # [1, d]
        center_loss  = F.mse_loss(anchor_features, batch_center.expand_as(anchor_features))
        #center_loss  = F.mse_loss(anchor_features, positive_features)
        loss = contrastive_loss + self.center_weight * center_loss

        return loss


def classification_loss_fn(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss

def train_epoch(model, train_loader, optimizer, all_texts, device, class_names, epoch, save_dir):
    model.train() 
    running_loss = 0.0 
    correct_preds = 0  
    total_preds = 0  
    all_labels = []  
    all_preds = []  
    alpha = opts.alpha
    criterion = ContrastiveLoss()  

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        logits, anchor_features = model(images)
               
        positive_text_features = torch.stack([all_texts[label.item()] for label in labels]).to(device)

        negative_text_features = []

        for i in range(len(labels)):
            class_idx = labels[i].item()
            negative_samples = [all_texts[j] for j in all_texts if j != class_idx]

            negative_text_features.extend(negative_samples)

        negative_text_features = torch.stack(negative_text_features).to(device)

        negative_text_features = negative_text_features.view(len(labels), len(negative_samples), -1)
        
        if negative_text_features.shape[1] != len(negative_samples):
            negative_text_features = negative_text_features.view(len(labels), len(negative_samples), -1)

        optimizer.zero_grad()  
        classification_loss = classification_loss_fn(logits, labels)
        loss = alpha * classification_loss + (1-alpha) * criterion(anchor_features, positive_text_features, negative_text_features)  # 计算三元组损失

        loss.backward() 
        optimizer.step()  

        _, preds = torch.max(logits, 1)  
        correct_preds += (preds == labels).sum().item() 
        total_preds += labels.size(0)  
        all_labels.extend(labels.cpu().numpy())  
        all_preds.extend(preds.cpu().numpy())  
        running_loss += loss.item()  

    avg_loss = running_loss / len(train_loader) 
    train_accuracy = correct_preds / total_preds * 100 

    cm = confusion_matrix(all_labels, all_preds)  
    plot_train_confusion_matrix(cm, class_names, os.path.join(save_dir, f"train_confusion_matrix_epoch_{epoch + 1}.png"))  # 绘制并保存混淆矩阵

    return avg_loss, train_accuracy 

def plot_train_confusion_matrix(cm, class_names, save_path):
    plt.rc('font', size=16) 
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  # 绘制热图
    plt.title('Train Confusion Matrix') 
    plt.xlabel('Predicted')
    plt.ylabel('True') 
    plt.savefig(save_path) 
    plt.close() 

def plot_confusion_matrix(cm, class_names, save_path):
    plt.rc('font', size=16) 
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    base, _ = os.path.splitext(save_path)
    plt.savefig(base + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_tsne(features, labels, class_names, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)  # shape: [N, 2]

    df_tsne = pd.DataFrame({
        'tsne_x': reduced_features[:, 0],
        'tsne_y': reduced_features[:, 1],
        'label_id': labels
    })

    csv_path = save_path.replace('.png', '.csv')
    df_tsne.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"t-SNE data saved to {csv_path}")

    plt.figure(figsize=(10, 7))
    for i, class_name in enumerate(class_names):
        indices = (labels == i)
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=class_name, alpha=0.7)

    plt.title('t-SNE visualization of Features')
    plt.legend(loc='best')
    plt.savefig(save_path)
    plt.close()
"""
# --- Alignment ratio α =1 --- 
def evaluate_epoch(model, val_loader, all_texts, device, class_names, epoch, save_dir):
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            logits, anchor_features = model(images)

            preds = torch.argmax(logits, dim=1)

            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features.append(anchor_features.cpu().numpy())

    avg_acc = correct_preds / total_preds * 100

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, f"confusion_matrix_epoch_{epoch + 1}.png"))

    all_features = np.concatenate(all_features, axis=0)
    plot_tsne(all_features, all_labels, class_names, os.path.join(save_dir, f"tsne_epoch_{epoch + 1}.png"))

    return avg_acc
"""

def evaluate_epoch(model, val_loader, all_texts, device, class_names, epoch, save_dir):
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    all_features = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(images)):
                image = images[i].unsqueeze(0)  # [1, C, H, W]
                label = labels[i].item() 

                _, anchor_feature = model(image) 

                anchor_feature = F.normalize(anchor_feature, p=2, dim=-1)  # [1, feature_dim]

                all_class_features = torch.stack(list(all_texts.values())).to(device)  # [num_classes, feature_dim]
                all_class_features = F.normalize(all_class_features, p=2, dim=-1)     # 归一化

                # anchor_feature [1, feature_dim], all_class_features [num_classes, feature_dim]
                distances_squared = torch.sum((anchor_feature - all_class_features) ** 2, dim=-1)  # [num_classes]

                prediction = torch.argmin(distances_squared).item()

                if prediction == label:
                    correct_preds += 1
                total_preds += 1

                all_preds.append(prediction)
                all_labels.append(label)
                all_features.append(anchor_feature.cpu().numpy())

    avg_acc = correct_preds / total_preds * 100

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, f"confusion_matrix_epoch_{epoch + 1}.png"))

    all_features = np.vstack(all_features)  #  [total_samples, feature_dim]
    plot_tsne(all_features, all_labels, class_names, os.path.join(save_dir, f"tsne_epoch_{epoch + 1}.png"))

    return avg_acc

def train_model(train_dir, val_dir):
    train_loader = load_data(train_dir, train_dir, opts.batch_size, opts.num_workers)
    val_loader = load_data(val_dir, train_dir, opts.batch_size, opts.num_workers)

    image_model = load_image_model().to(opts.device)
    text_model, tokenizer = load_text_model(model_name=opts.text_model)

    optimizer = optim.AdamW(image_model.parameters(), lr=opts.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs)

    train_dataset = CustomDataset(train_dir, val_dir)
    class_to_idx = train_dataset.class_to_idx
    class_names = list(class_to_idx.keys())

    save_dir = Path(opts.model_save_path) / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_texts = load_all_texts(val_dir, text_model, tokenizer, opts.device, class_to_idx)

    for epoch in range(opts.epochs):
        avg_train_loss, train_accuracy = train_epoch(image_model, train_loader, optimizer, all_texts, opts.device, class_names, epoch, save_dir)
        print(f"Epoch {epoch+1}/{opts.epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        avg_val_acc = evaluate_epoch(image_model, val_loader, all_texts, opts.device, class_names, epoch, save_dir)
        print(f"Epoch {epoch+1}/{opts.epochs}, Validation Accuracy: {avg_val_acc:.2f}%")

        scheduler.step()

if __name__ == "__main__":
    train_dir = 'E:\mIT-CMCA\dataset\MLFD/train'
    val_dir = 'E:\mIT-CMCA\dataset\MLFD/val'
    train_model(train_dir, val_dir)
