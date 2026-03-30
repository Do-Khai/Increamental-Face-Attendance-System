import os
import pickle
import sys
import warnings

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    detect_and_crop_face,
    extract_arcface_features,
    load_models,
    print_class_distribution,
)


class IncrementalFeatureExtractor:
    def __init__(
        self,
        features_path="../models/incremental_features.pkl",
        metadata_path="../models/features_metadata.json",
    ):
        """Initialize incremental feature extractor"""
        self.features_path = self._get_absolute_path(features_path)
        self.metadata_path = self._get_absolute_path(metadata_path)

        # Load existing features and metadata if available
        self.existing_features = {}
        self.existing_labels = []
        self.metadata = {}
        self.load_existing_data()

        # Load models
        print("Loading models...")
        self.face_detector, self.arcface_model = load_models()
        print("Models loaded successfully!")

    def _get_absolute_path(self, relative_path):
        """Convert relative path to absolute path"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "..", relative_path.lstrip("../"))

    def load_existing_data(self):
        """Load existing features and metadata"""
        try:
            # Load existing features
            if os.path.exists(self.features_path):
                with open(self.features_path, "rb") as f:
                    data = pickle.load(f)
                    self.existing_features = data.get("features_dict", {})
                    self.existing_labels = data.get("labels", [])
                print(f"Loaded {len(self.existing_features)} existing feature vectors")

            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                print(f"Loaded metadata: {len(self.metadata)} classes")
            else:
                self.metadata = {
                    "total_images": 0,
                    "total_classes": 0,
                    "last_updated": None,
                    "class_info": {},
                }

        except Exception as e:
            print(f"Error loading existing data: {e}")
            self.existing_features = {}
            self.existing_labels = []
            self.metadata = {
                "total_images": 0,
                "total_classes": 0,
                "last_updated": None,
                "class_info": {},
            }

    def scan_new_data(self, data_path="../dataset"):
        """Scan for new data and return new/changed items"""
        data_path = self._get_absolute_path(data_path)

        new_images = []
        updated_classes = set()

        print("Scanning for new data...")

        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)

            if os.path.isdir(folder_path):
                # Get all image files in folder
                image_files = [
                    f
                    for f in os.listdir(folder_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]

                class_images = []
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    class_images.append(image_path)

                # Check if this class is new or has new images
                if folder_name not in self.metadata["class_info"]:
                    # New class
                    print(
                        f"New class detected: {folder_name} ({len(class_images)} images)"
                    )
                    new_images.extend([(path, folder_name) for path in class_images])
                    updated_classes.add(folder_name)
                else:
                    # Existing class - check for new images
                    existing_images = set(
                        self.metadata["class_info"][folder_name]["images"]
                    )
                    current_images = set(class_images)
                    new_images_for_class = current_images - existing_images

                    if new_images_for_class:
                        print(
                            f"New images for class {folder_name}: {len(new_images_for_class)} images"
                        )
                        new_images.extend(
                            [(path, folder_name) for path in new_images_for_class]
                        )
                        updated_classes.add(folder_name)

        print(
            f"Found {len(new_images)} new images across {len(updated_classes)} classes"
        )
        return new_images, updated_classes

    def extract_features_from_images(self, image_paths_labels):
        """Extract features from new images"""
        features = []
        labels = []
        failed_images = []

        print("Extracting features from new images...")

        for i, (image_path, label) in enumerate(
            tqdm(image_paths_labels, total=len(image_paths_labels))
        ):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    failed_images.append(image_path)
                    continue

                # Detect and crop face
                face_crop = detect_and_crop_face(image, self.face_detector)
                if face_crop is None:
                    print(f"No face detected in: {image_path}")
                    failed_images.append(image_path)
                    continue

                # Extract ArcFace features
                face_features = extract_arcface_features(face_crop, self.arcface_model)

                features.append(face_features)
                labels.append(label)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                failed_images.append(image_path)
                continue

        print(f"Successfully extracted features from {len(features)} images")
        if failed_images:
            print(f"Failed to process {len(failed_images)} images")

        return np.array(features), labels, failed_images

    def update_features_and_metadata(self, new_features, new_labels, updated_classes):
        """Update existing features and metadata with new data"""
        print("Updating features and metadata...")

        # Add new features to existing ones
        for i, (features, label) in enumerate(zip(new_features, new_labels)):
            # Create unique key for this feature
            feature_key = (
                f"{label}_{len([l for l in self.existing_labels if l == label]) + 1}"
            )
            self.existing_features[feature_key] = features
            self.existing_labels.append(label)

        # Update metadata
        self.metadata["total_images"] = len(self.existing_labels)
        self.metadata["last_updated"] = datetime.now().isoformat()

        # Update class information
        for class_name in updated_classes:
            class_images = [
                label for label in self.existing_labels if label == class_name
            ]
            class_count = len(class_images)

            if class_name not in self.metadata["class_info"]:
                # New class
                self.metadata["class_info"][class_name] = {
                    "count": class_count,
                    "first_added": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "images": [],  # Will be updated in save method
                }
            else:
                # Existing class
                self.metadata["class_info"][class_name]["count"] = class_count
                self.metadata["class_info"][class_name]["last_updated"] = (
                    datetime.now().isoformat()
                )

        self.metadata["total_classes"] = len(self.metadata["class_info"])

    def save_features_and_metadata(self):
        """Save updated features and metadata"""
        os.makedirs(os.path.dirname(self.features_path), exist_ok=True)

        # Prepare data for saving
        data = {
            "features_dict": self.existing_features,
            "labels": self.existing_labels,
            "feature_vectors": np.array(list(self.existing_features.values())),
            "metadata": self.metadata,
        }

        # Save features
        with open(self.features_path, "wb") as f:
            pickle.dump(data, f)

        # Update image paths in metadata
        self._update_image_paths_in_metadata()

        # Save metadata
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Features saved to: {self.features_path}")
        print(f"Metadata saved to: {self.metadata_path}")
        print(f"Total features: {len(self.existing_features)}")
        print(f"Total classes: {self.metadata['total_classes']}")

    def _update_image_paths_in_metadata(self):
        """Update image paths in metadata for each class"""
        data_path = self._get_absolute_path("../dataset")

        for class_name in self.metadata["class_info"]:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                image_files = [
                    f
                    for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
                image_paths = [os.path.join(class_path, f) for f in image_files]
                self.metadata["class_info"][class_name]["images"] = image_paths

    def get_features_for_training(self):
        """Get features and labels in format suitable for training"""
        features = np.array(list(self.existing_features.values()))
        labels = self.existing_labels

        return features, labels

    def print_statistics(self):
        """Print current statistics"""
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION STATISTICS")
        print("=" * 60)
        print(f"Total features: {len(self.existing_features)}")
        print(f"Total classes: {self.metadata['total_classes']}")
        print(f"Last updated: {self.metadata['last_updated']}")

        print("\nClass distribution:")
        for class_name, info in self.metadata["class_info"].items():
            print(f"  {class_name}: {info['count']} images")

    def incremental_extract(self, data_path="../dataset"):
        """Main incremental extraction function"""
        print("=" * 60)
        print("INCREMENTAL FEATURE EXTRACTION")
        print("=" * 60)

        # Scan for new data
        new_images, updated_classes = self.scan_new_data(data_path)

        if not new_images:
            print("No new data found. All features are up to date.")
            return

        # Extract features from new images
        new_features, new_labels, failed_images = self.extract_features_from_images(
            new_images
        )

        if len(new_features) == 0:
            print("No features could be extracted from new images.")
            return

        # Update features and metadata
        self.update_features_and_metadata(new_features, new_labels, updated_classes)

        # Save updated data
        self.save_features_and_metadata()

        # Print statistics
        self.print_statistics()

        print("\n" + "=" * 60)
        print("INCREMENTAL EXTRACTION COMPLETED")
        print("=" * 60)

    def remove_employees(self, employee_ids, dataset_path="../dataset"):
        """Remove employees' data and related features"""
        dataset_path = self._get_absolute_path(dataset_path)

        removed_classes = []
        for emp_id in employee_ids:
            # 1. Xóa thư mục dataset/<employeeId>
            emp_folder = os.path.join(dataset_path, emp_id)
            if os.path.exists(emp_folder):
                import shutil

                shutil.rmtree(emp_folder)
                print(f"Deleted folder: {emp_folder}")
            else:
                print(f"Folder not found: {emp_folder}")

            # 2. Xóa đặc trưng và label liên quan
            keys_to_delete = [
                k for k, v in self.existing_features.items() if emp_id in k
            ]  # key chứa label
            for key in keys_to_delete:
                del self.existing_features[key]

            # Xóa trong labels (nếu label là list song song với features)
            self.existing_labels = [
                label for label in self.existing_labels if label != emp_id
            ]

            # Xóa trong metadata
            if emp_id in self.metadata["class_info"]:
                del self.metadata["class_info"][emp_id]

            removed_classes.append(emp_id)

        # Cập nhật lại metadata tổng
        self.metadata["total_images"] = len(self.existing_labels)
        self.metadata["total_classes"] = len(self.metadata["class_info"])
        self.metadata["last_updated"] = datetime.now().isoformat()

        # Lưu lại file
        self.save_features_and_metadata()

        print(f"Removed employees: {removed_classes}")


def main():
    """Main function for incremental feature extraction"""
    extractor = IncrementalFeatureExtractor()
    extractor.incremental_extract()


if __name__ == "__main__":
    main()
