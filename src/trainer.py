import os
import numpy as np
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class IncrementalMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.25):
        super(IncrementalMLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Hidden layers
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Output layer
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class StandaloneIncrementalTrainer:
    def __init__(
        self,
        model_path="../models/incremental_mlp_model.pth",
        encoder_path="../models/incremental_label_encoder.pkl",
        metadata_path="../models/model_metadata.json",
        features_path="../models/incremental_features.pkl",
    ):
        """Initialize standalone incremental model trainer"""
        self.model_path = self._get_absolute_path(model_path)
        self.encoder_path = self._get_absolute_path(encoder_path)
        self.metadata_path = self._get_absolute_path(metadata_path)
        self.features_path = self._get_absolute_path(features_path)

        # Load existing model and metadata if available
        self.existing_model = None
        self.existing_encoder = None
        self.model_metadata = {}
        self.load_existing_model()

    def _get_absolute_path(self, relative_path):
        """Convert relative path to absolute path"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "..", relative_path.lstrip("../"))

    def load_existing_model(self):
        """Load existing model, encoder and metadata"""
        try:
            # Load existing model
            if os.path.exists(self.model_path):
                print("Loading existing model...")
                self.existing_model = torch.load(self.model_path, map_location=device)
                print("Existing model loaded successfully!")

            # Load existing encoder
            if os.path.exists(self.encoder_path):
                with open(self.encoder_path, "rb") as f:
                    self.existing_encoder = pickle.load(f)
                print("Existing encoder loaded successfully!")

            # Load model metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
                print("Model metadata loaded successfully!")
            else:
                self.model_metadata = {
                    "total_classes": 0,
                    "input_dim": 512,
                    "last_updated": None,
                    "training_history": [],
                    "best_accuracy": 0.0,
                    "model_versions": [],
                }

        except Exception as e:
            print(f"Error loading existing model: {e}")
            self.existing_model = None
            self.existing_encoder = None
            self.model_metadata = {
                "total_classes": 0,
                "input_dim": 512,
                "last_updated": None,
                "training_history": [],
                "best_accuracy": 0.0,
                "model_versions": [],
            }

    def load_features_and_labels(self):
        """Load features and labels from pickle file"""
        if not os.path.exists(self.features_path):
            print(f"Features file not found: {self.features_path}")
            print("Please run extractor.py first.")
            return None, None

        print("Loading features and labels...")
        with open(self.features_path, "rb") as f:
            data = pickle.load(f)

        features = data["feature_vectors"]
        labels = data["labels"]

        print(f"Loaded {len(features)} features with {len(set(labels))} classes")
        return features, labels

    def prepare_incremental_data(self, features, labels):
        """Prepare data for incremental training"""
        print("Preparing data for incremental training...")

        # Create new encoder or update existing one
        if self.existing_encoder is None:
            # First time training
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            print(
                f"Created new encoder with {len(self.label_encoder.classes_)} classes"
            )
        else:
            # Incremental training - update encoder with new classes
            self.label_encoder = self.existing_encoder

            # Get existing classes
            existing_classes = set(self.label_encoder.classes_)
            current_classes = set(labels)
            new_classes = current_classes - existing_classes

            if new_classes:
                print(f"Found {len(new_classes)} new classes: {list(new_classes)}")
                # Add new classes to encoder
                all_classes = list(existing_classes) + list(new_classes)
                self.label_encoder.classes_ = np.array(all_classes)

            encoded_labels = self.label_encoder.transform(labels)
            print(
                f"Updated encoder with {len(self.label_encoder.classes_)} total classes"
            )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels,
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")

        return X_train, X_test, y_train, y_test

    def create_or_update_model(self, input_dim, num_classes):
        """Create new model or update existing model for new classes"""
        if self.existing_model is None:
            # Create new model
            print("Creating new model...")
            model = IncrementalMLPClassifier(input_dim, num_classes)
        else:
            # Update existing model for new classes
            print("Updating existing model for new classes...")
            model = IncrementalMLPClassifier(input_dim, num_classes)

            # Load existing weights for existing classes
            if num_classes > self.model_metadata["total_classes"]:
                # Copy weights for existing classes
                for name, param in model.named_parameters():
                    if name in self.existing_model:
                        if param.shape == self.existing_model[name].shape:
                            param.data.copy_(self.existing_model[name])
                            print(f"Copied weights for layer: {name}")
                        else:
                            print(f"Layer {name} shape changed, reinitializing")

        return model

    def compute_class_weights(self, y_train):
        """Compute class weights for imbalanced data"""
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )

        # Create class weight dictionary
        class_weight_dict = dict(zip(range(len(class_weights)), class_weights))

        print("\nClass weights:")
        for i, weight in enumerate(class_weights):
            class_name = self.label_encoder.inverse_transform([i])[0]
            print(f"  {class_name}: {weight:.3f}")

        return class_weight_dict

    def train_incremental_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        class_weight_dict,
        num_classes,
        input_dim,
        epochs=50,
        batch_size=32,
    ):
        """Train model incrementally"""
        print("\nTraining incremental model...")

        # Create or update model
        model = self.create_or_update_model(input_dim, num_classes)
        model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
        )

        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training history
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_acc = 0.0
        patience_counter = 0

        print(f"\nTraining for {epochs} epochs...")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 80)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_correct = (val_predicted == y_test_tensor).sum().item()
                val_total = y_test_tensor.size(0)

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            # Update history
            history["train_loss"].append(train_loss / len(train_loader))
            history["train_acc"].append(train_acc / 100)
            history["val_loss"].append(val_loss.item())
            history["val_acc"].append(val_acc / 100)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_path = self._get_absolute_path(
                    "../models/best_incremental_model.pth"
                )
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 1 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch + 1}/{epochs}] - "
                    f"Train Loss: {train_loss / len(train_loader):.4f}, "
                    f"Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss.item():.4f}, "
                    f"Val Acc: {val_acc:.2f}%, "
                    f"LR: {current_lr:.6f}"
                )

            # Early stopping
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_model_path = self._get_absolute_path(
            "../models/best_incremental_model.pth"
        )
        model.load_state_dict(torch.load(best_model_path))

        return model, history

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate trained model"""
        print("\nEvaluating model...")

        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        # Predictions
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_
            )
        )

        # # Only plot confusion matrix if number of classes is reasonable (< 50)
        # num_classes = len(self.label_encoder.classes_)
        # if num_classes <= 50:
        #     print(f"\nPlotting confusion matrix for {num_classes} classes...")
        #     # Confusion matrix
        #     cm = confusion_matrix(y_test, y_pred)

        #     # Plot confusion matrix
        #     plt.figure(figsize=(12, 8))
        #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        #                 xticklabels=self.label_encoder.classes_,
        #                 yticklabels=self.label_encoder.classes_)
        #     plt.title('Incremental Model Confusion Matrix')
        #     plt.xlabel('Predicted')
        #     plt.ylabel('Actual')
        #     plt.xticks(rotation=45)
        #     plt.yticks(rotation=0)
        #     plt.tight_layout()
        #     confusion_matrix_path = self._get_absolute_path("../models/incremental_confusion_matrix.png")
        #     plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print("Confusion matrix saved!")
        # else:
        #     print(f"\nSkipping confusion matrix plot due to large number of classes ({num_classes} > 50)")

        return y_pred, y_pred_proba

    def save_model_and_metadata(self, model, history):
        """Save model, encoder and metadata"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Save model
        torch.save(model.state_dict(), self.model_path)

        # Save ONNX model
        try:
            onnx_path = self.model_path.replace('.pth', '.onnx')
            model.eval()
            
            # Extract input_dim from model layers
            input_dim = model.layers[0].in_features
            dummy_input = torch.randn(1, input_dim, device=device)
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"ONNX model saved to: {onnx_path}")
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")

        # Save encoder
        with open(self.encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Update metadata
        self.model_metadata["total_classes"] = len(self.label_encoder.classes_)
        self.model_metadata["last_updated"] = datetime.now().isoformat()
        self.model_metadata["best_accuracy"] = max(history["val_acc"])

        # Add training session to history
        training_session = {
            "date": datetime.now().isoformat(),
            "classes": len(self.label_encoder.classes_),
            "best_accuracy": max(history["val_acc"]),
            "epochs_trained": len(history["train_loss"]),
        }
        self.model_metadata["training_history"].append(training_session)

        # Save metadata
        with open(self.metadata_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2)

        print(f"Model saved to: {self.model_path}")
        print(f"Encoder saved to: {self.encoder_path}")
        print(f"Metadata saved to: {self.metadata_path}")

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history["train_acc"], label="Training Accuracy")
        ax1.plot(history["val_acc"], label="Validation Accuracy")
        ax1.set_title("Incremental Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # Plot loss
        ax2.plot(history["train_loss"], label="Training Loss")
        ax2.plot(history["val_loss"], label="Validation Loss")
        ax2.set_title("Incremental Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        training_history_path = self._get_absolute_path(
            "../models/incremental_training_history.png"
        )
        plt.savefig(training_history_path)
        plt.close()

    def print_statistics(self):
        """Print model statistics"""
        print("\n" + "=" * 60)
        print("INCREMENTAL MODEL STATISTICS")
        print("=" * 60)
        print(f"Total classes: {self.model_metadata['total_classes']}")
        print(f"Best accuracy: {self.model_metadata['best_accuracy']:.4f}")
        print(f"Last updated: {self.model_metadata['last_updated']}")
        print(f"Training sessions: {len(self.model_metadata['training_history'])}")

        if self.label_encoder:
            print(f"\nCurrent classes:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                print(f"  {i}: {class_name}")

    def train_standalone(self):
        """Main standalone training function"""
        print("=" * 60)
        print("STANDALONE INCREMENTAL MODEL TRAINING")
        print("=" * 60)

        # Load features and labels
        features, labels = self.load_features_and_labels()

        if features is None or len(features) == 0:
            print("No features available for training.")
            return

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_incremental_data(
            features, labels
        )

        # Compute class weights
        class_weight_dict = self.compute_class_weights(y_train)

        # Train model
        input_dim = features.shape[1]
        num_classes = len(self.label_encoder.classes_)

        model, history = self.train_incremental_model(
            X_train, y_train, X_test, y_test, class_weight_dict, num_classes, input_dim
        )

        # Evaluate model
        y_pred, y_pred_proba = self.evaluate_model(model, X_test, y_test)

        # Save model and metadata
        self.save_model_and_metadata(model, history)

        # Plot training history
        self.plot_training_history(history)

        # Print statistics
        self.print_statistics()

        print("\n" + "=" * 60)
        print("STANDALONE TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
        print(f"Best training accuracy: {max(history['train_acc']):.4f}")


def main():
    """Main function for standalone incremental model training"""
    trainer = StandaloneIncrementalTrainer()
    trainer.train_standalone()


if __name__ == "__main__":
    main()
