import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
from torchvision import datasets, transforms
import h5py
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Factorized Time-Modality Transformer (FTMT)
class FactorizedSelfAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, mode='simultaneous'):
        super(FactorizedSelfAttention, self).__init__()
        self.mode = mode
        self.d_model = d_model
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.modality_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Capa para ajustar la dimensión a d_model si es necesario
        self.adjust_dim = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

        self.reduce_dim = nn.Linear(d_model, input_dim) if d_model != input_dim else nn.Identity()
    
    def forward(self, x):
        # x shape: [batch_size, num_modalities, sequence_length, input_dim]
        # print("Forma de x antes de adjust_dim:", x.shape)
        
        # Ajustar la dimensión de entrada a d_model si es necesario
        x = self.adjust_dim(x)  # [batch_size, num_modalities, sequence_length, d_model]
        
        # print("Forma de x después de adjust_dim:", x.shape)

        # Fusionar dimensiones de batch y modalidad para aplicar la atención temporal
        batch_size, num_modalities, sequence_length, _ = x.size()
            
        # Reorganizar para que `sequence_length` sea compatible con la atención temporal
        x = x.permute(0, 2, 1, 3).reshape(batch_size * sequence_length, num_modalities, self.d_model)
        attn_output_temporal, _ = self.temporal_attention(x, x, x)
        
        # Restaurar la forma original después de la atención temporal
        attn_output_temporal = attn_output_temporal.view(batch_size, sequence_length, num_modalities, self.d_model).permute(0, 2, 1, 3)
        
        # Aplicar la atención de modalidad en la dimensión de modalidad
        attn_output_modality, _ = self.modality_attention(
            attn_output_temporal.transpose(1, 2).reshape(-1, num_modalities, self.d_model),
            attn_output_temporal.transpose(1, 2).reshape(-1, num_modalities, self.d_model),
            attn_output_temporal.transpose(1, 2).reshape(-1, num_modalities, self.d_model)
        )
        
        # Restaurar la forma final
        attn_output_modality = attn_output_modality.view(batch_size, sequence_length, num_modalities, self.d_model).transpose(1, 2)
        
        # Reducir la dimensión de salida a 224 si es necesario
        attn_output_modality = self.reduce_dim(attn_output_modality)
        return attn_output_modality




# Multimodal Contrastive Alignment Network (MCANet)
class MCANet(nn.Module):
    def __init__(self, d_model):
        super(MCANet, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.fc = nn.Linear(d_model, d_model)
    
    def forward(self, main_modality, sub_modalities):
        fused_features = main_modality
        for modality in sub_modalities:
            similarity = self.cosine_similarity(main_modality, modality)
            weight = torch.sigmoid(similarity).unsqueeze(-1)
            fused_features += weight * modality
        return fused_features

# UCFFormer Network
class UCFFormer(nn.Module):
    def __init__(self, input_dim=224, d_model=224, num_heads=8, num_layers=4, num_classes=10, mode='simultaneous'):
        super(UCFFormer, self).__init__()
        self.d_model = d_model
        self.ftmt = nn.ModuleList([FactorizedSelfAttention(input_dim, d_model, num_heads, mode) for _ in range(num_layers)])
        self.mcanet = MCANet(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        self.expand_dim = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
    
    def forward(self, modalities):
        # print("entrada forward")
        if not isinstance(modalities, (list, tuple)):
            raise TypeError("modalities should be a list or tuple of tensors.")
        fused = torch.cat(modalities, dim=0)  # Concatenate all modalities
        # print("pass fused")
        count = 0
        
        for layer in self.ftmt:
            # print(count)
            count+=1
            # print("FUSED")
            print(fused.shape)
            fused = layer(fused)

            
        # print("pass for")
        fused = self.expand_dim(fused)
        # print("Forma de fused después de expand_dim:", fused.shape)


        fused_features = self.mcanet(fused[0], fused[1:])
        output = self.fc(fused_features.mean(dim=1))  # Global average pooling + classification
        # print("salida forward")
        return output

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, main_modality, sub_modalities):
        contrastive_loss = 0
        for modality in sub_modalities:
            cosine_sim = F.cosine_similarity(main_modality, modality, dim=-1)
            contrastive_loss += (1 - cosine_sim).mean()
        return contrastive_loss

# Load dataset using ImageFolder
def load_dataset(path, img_size=(224, 224), batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=path, transform=transform)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, dataset.class_to_idx

# Function to load pretrained model from .mat file
def load_pretrained_model(mat_file, layer_name):
    with h5py.File(mat_file, 'r') as f:
        XONet_weights = f['XONet'][:]
    
    model = UCFFormer(d_model=512, num_heads=8, num_layers=4, num_classes=10, mode='simultaneous')
    model_state_dict = model.state_dict()
    
    for name, param in model_state_dict.items():
        if name in XONet_weights:
            param.data = torch.tensor(XONet_weights[name])
    
    return model

# Feature extraction function
def extract_features(model, data_loader, layer_name):
    features, labels = [], []
    model.eval()
    count =0
    with torch.no_grad():
        for images, label in data_loader:
            # print("iteration")
            # print(count)
            count+=1
            modalities = [images]
            output = model(modalities)
            features.append(output)
            labels.append(label)
    return torch.cat(features), torch.cat(labels)

# Paths and data setup
data_paths = {
    'inertial': 'FullAugmentedSignalImages',
    'prewitt': 'Prewitt_SignalImages',
    'depth': 'Depth_227x227x3',
    'depth_prewitt': 'Depth_Prewitt'
}
models = {
    'inertial': 'XONet_FullAugmented.mat',
    'prewitt': 'XONet_Prewitt_Signal.mat',
    'depth': 'XONet_depth.mat',
    'depth_prewitt': 'XONet_Prewitt_depth.mat'
}
layer_names = {
    'inertial': 'fc_2',
    'prewitt': 'fc',
    'depth': 'fc8',
    'depth_prewitt': 'fc8'
}

# Load datasets and models
train_loaders, val_loaders, class_idx = {}, {}, {}

for key in data_paths:
    train_loaders[key], val_loaders[key], class_idx[key] = load_dataset(data_paths[key])

train_features, val_features, train_labels, val_labels = {}, {}, {}, {}

count = 0
for key in models:
    # print(count)
    count+=1
    model = load_pretrained_model(models[key], layer_names[key])
    train_features[key], train_labels[key] = extract_features(model, train_loaders[key], layer_names[key])
    val_features[key], val_labels[key] = extract_features(model, val_loaders[key], layer_names[key])

# Concatenate train and test features
FeaturesTrain1 = torch.cat([train_features['inertial'], train_features['prewitt']], dim=0)
FeaturesTest1 = torch.cat([val_features['inertial'], val_features['prewitt']], dim=0)
YTrain = torch.cat([train_labels['inertial'], train_labels['prewitt']], dim=0)
YTest = torch.cat([val_labels['inertial'], val_labels['prewitt']], dim=0)

FeaturesTrain2 = torch.cat([train_features['depth'], train_features['depth_prewitt']], dim=0)
FeaturesTest2 = torch.cat([val_features['depth'], val_features['depth_prewitt']], dim=0)
YYTrain = torch.cat([train_labels['depth'], train_labels['depth_prewitt']], dim=0)
YYTest = torch.cat([val_labels['depth'], val_labels['depth_prewitt']], dim=0)

# Print feature dimensions
print("FeaturesTrain1 size:", FeaturesTrain1.size())
print("FeaturesTrain2 size:", FeaturesTrain2.size())
print("YYTrain size:", YYTrain.size())
print("YTrain size:", YTrain.size())

def expand_features_to_match_labels(features, labels):
    # Calcular la diferencia de tamaño entre etiquetas y características
    diff = labels.size(0) - features.size(0)
    
    # Seleccionar un subconjunto aleatorio de características para replicar
    if diff > 0:
        indices = torch.randint(0, features.size(0), (diff,))
        features_expanded = torch.cat([features, features[indices]], dim=0)
    else:
        features_expanded = features
    
    return features_expanded

# Aplicar a FeaturesTrain1 y FeaturesTrain2
FeaturesTrain1 = expand_features_to_match_labels(FeaturesTrain1, YTrain)
FeaturesTrain2 = expand_features_to_match_labels(FeaturesTrain2, YYTrain)
FeaturesTest1 = expand_features_to_match_labels(FeaturesTest1, YTest)

print("Dimensiones de FeaturesTrain1:", FeaturesTrain1.size())
print("Dimensiones de YTrain:", YTrain.size())


print("Dimensiones de FeaturesTest1:", FeaturesTest1.size())
print("Dimensiones de YTest:", YTest.size())
# Crear datasets de entrenamiento y prueba
train_dataset = TensorDataset(FeaturesTrain1, YTrain)
test_dataset = TensorDataset(FeaturesTest1, YTest)

# Crear data loaders
batch_size = 224
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



def evaluate_model_with_metrics(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    correct, total = 0, 0

    with torch.no_grad():
        for features, labels in test_loader:
            output = model([features])
            _, predicted = output.max(1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calcular métricas adicionales
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

# Ejecutar la evaluación con métricas
evaluate_model_with_metrics(model, test_loader)
