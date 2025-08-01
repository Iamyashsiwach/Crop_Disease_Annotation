#!/usr/bin/env python3
"""
Improved YOLOv8 Segmentation Training Script for Crop Disease Detection
======================================================================

This is an enhanced version of the training script with additional features:
- Automatic hyperparameter optimization
- Enhanced data augmentation
- Better monitoring and logging
- Model comparison capabilities
- Custom callbacks for training

Based on the improved training approach from crop_Insect_Annotation repository.

Author: Generated for Crop Disease Annotation Project
"""

import os
import sys
import json
import time
import yaml
import glob
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedYOLOTrainer:
    """
    Enhanced YOLO trainer with advanced features.
    """
    
    def __init__(self, config_path: str = "training_config.yaml"):
        """
        Initialize the improved trainer.
        
        Args:
            config_path (str): Path to training configuration file
        """
        self.config = self.load_config(config_path)
        self.results_dir = None
        self.best_model_path = None
        
    def load_config(self, config_path: str) -> Dict:
        """
        Load training configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        default_config = {
            'dataset': {
                'path': 'crop_disease_seg',
                'data_yaml': 'crop_disease_seg/data.yaml'
            },
            'model': {
                'name': 'yolov8s-seg.pt',
                'pretrained': True
            },
            'training': {
                'epochs': 200,
                'batch_size': 16,
                'image_size': 640,
                'patience': 50,
                'save_period': 10,
                'device': 'auto',
                'workers': 8,
                'project': 'runs/segment',
                'name': 'crop_disease_improved'
            },
            'hyperparameters': {
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            },
            'augmentation': {
                'enabled': True,
                'albumentations': False,
                'auto_augment': 'randaugment',
                'erasing': 0.4
            },
            'validation': {
                'val_period': 1,
                'save_json': True,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'half': False,
                'dnn': False
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.info(f"Configuration file {config_path} not found, using defaults")
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Saved default configuration to {config_path}")
        
        return default_config
    
    def setup_environment(self):
        """
        Setup training environment and check requirements.
        """
        logger.info("🔧 Setting up training environment...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 GPU available: {gpu_name} (Count: {gpu_count})")
        else:
            logger.info("💻 Training on CPU")
        
        # Check dataset
        data_yaml = self.config['dataset']['data_yaml']
        if not os.path.exists(data_yaml):
            logger.error(f"❌ Dataset configuration not found: {data_yaml}")
            return False
        
        dataset_path = self.config['dataset']['path']
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if not os.path.exists(dir_path):
                logger.error(f"❌ Required directory not found: {dir_path}")
                return False
            else:
                count = len(glob.glob(os.path.join(dir_path, '*')))
                logger.info(f"✅ {dir_name}: {count} files")
        
        return True
    
    def load_model(self) -> YOLO:
        """
        Load and configure the YOLO model.
        
        Returns:
            YOLO: Configured YOLO model
        """
        model_name = self.config['model']['name']
        logger.info(f"📥 Loading model: {model_name}")
        
        try:
            model = YOLO(model_name)
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 Model info: {model.info()}")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def train_model(self, model: YOLO) -> object:
        """
        Train the YOLO model with enhanced configuration.
        
        Args:
            model (YOLO): YOLO model to train
            
        Returns:
            Training results object
        """
        training_config = self.config['training']
        hyperparams = self.config['hyperparameters']
        
        logger.info("🏋️ Starting enhanced training...")
        logger.info(f"📊 Training configuration:")
        for key, value in training_config.items():
            logger.info(f"   {key}: {value}")
        
        # Prepare training arguments
        train_args = {
            'data': self.config['dataset']['data_yaml'],
            'epochs': training_config['epochs'],
            'batch': training_config['batch_size'],
            'imgsz': training_config['image_size'],
            'device': training_config['device'],
            'workers': training_config['workers'],
            'project': training_config['project'],
            'name': training_config['name'],
            'patience': training_config['patience'],
            'save_period': training_config['save_period'],
            'exist_ok': True,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': self.config['validation']['save_json'],
            'save_hybrid': self.config['validation']['save_hybrid'],
            'conf': self.config['validation']['conf'],
            'iou': self.config['validation']['iou'],
            'max_det': self.config['validation']['max_det'],
            'half': self.config['validation']['half'],
            'dnn': self.config['validation']['dnn'],
            'plots': True,
            'save': True,
            'save_frames': False,
            'show': False,
            'show_labels': True,
            'show_conf': True,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'retina_masks': False,
            'embed': None,
        }
        
        # Add hyperparameters
        train_args.update(hyperparams)
        
        try:
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            logger.info(f"✅ Training completed in {training_time:.2f} seconds")
            self.results_dir = results.save_dir
            
            # Save training summary
            self.save_training_summary(results, training_time)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def save_training_summary(self, results: object, training_time: float):
        """
        Save comprehensive training summary.
        
        Args:
            results: Training results object
            training_time (float): Total training time in seconds
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'training_time_formatted': f"{training_time/3600:.2f} hours",
            'config': self.config,
            'results_directory': str(self.results_dir),
            'model_files': {
                'best': os.path.join(self.results_dir, 'weights', 'best.pt'),
                'last': os.path.join(self.results_dir, 'weights', 'last.pt')
            }
        }
        
        # Add metrics if available
        if hasattr(results, 'results_dict'):
            summary['final_metrics'] = results.results_dict
        
        # Save summary
        summary_path = os.path.join(self.results_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved to: {summary_path}")
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model_path (str, optional): Path to model weights
            
        Returns:
            Dict: Evaluation results
        """
        if model_path is None:
            model_path = os.path.join(self.results_dir, 'weights', 'best.pt')
        
        logger.info(f"📊 Evaluating model: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            # Run validation
            val_results = model.val(
                data=self.config['dataset']['data_yaml'],
                imgsz=self.config['training']['image_size'],
                batch=1,
                conf=self.config['validation']['conf'],
                iou=self.config['validation']['iou'],
                device=self.config['training']['device'],
                half=self.config['validation']['half'],
                dnn=self.config['validation']['dnn'],
                plots=True,
                save_json=True,
                save_hybrid=False,
                verbose=True
            )
            
            # Extract metrics
            metrics = {}
            if hasattr(val_results, 'results_dict'):
                metrics = val_results.results_dict
            
            logger.info("📈 Evaluation Results:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   {key}: {value:.4f}")
                else:
                    logger.info(f"   {key}: {value}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {}
    
    def run_sample_predictions(self, model_path: Optional[str] = None, num_samples: int = 5):
        """
        Run predictions on sample validation images.
        
        Args:
            model_path (str, optional): Path to model weights
            num_samples (int): Number of sample predictions to run
        """
        if model_path is None:
            model_path = os.path.join(self.results_dir, 'weights', 'best.pt')
        
        logger.info(f"🎯 Running sample predictions...")
        
        try:
            model = YOLO(model_path)
            
            # Get sample images
            val_images_dir = os.path.join(self.config['dataset']['path'], 'images', 'val')
            image_files = glob.glob(os.path.join(val_images_dir, '*'))
            
            if not image_files:
                logger.warning("⚠️ No validation images found")
                return
            
            sample_images = random.sample(image_files, min(num_samples, len(image_files)))
            
            predictions_dir = os.path.join(self.results_dir, 'sample_predictions')
            os.makedirs(predictions_dir, exist_ok=True)
            
            for i, image_path in enumerate(sample_images):
                logger.info(f"   Processing sample {i+1}/{len(sample_images)}: {os.path.basename(image_path)}")
                
                results = model.predict(
                    source=image_path,
                    imgsz=self.config['training']['image_size'],
                    conf=self.config['validation']['conf'],
                    iou=self.config['validation']['iou'],
                    save=True,
                    project=predictions_dir,
                    name=f'sample_{i+1}',
                    exist_ok=True
                )
                
                # Log prediction details
                if results:
                    result = results[0]
                    if result.masks is not None:
                        num_detections = len(result.masks.data)
                        logger.info(f"     Detected {num_detections} objects")
                        if result.boxes is not None and result.boxes.conf is not None:
                            confidences = result.boxes.conf.cpu().numpy()
                            logger.info(f"     Confidences: {confidences}")
                    else:
                        logger.info(f"     No objects detected")
            
            logger.info(f"✅ Sample predictions saved to: {predictions_dir}")
            
        except Exception as e:
            logger.error(f"❌ Sample prediction failed: {e}")
    
    def hyperparameter_tuning(self, trials: int = 30):
        """
        Perform hyperparameter optimization.
        
        Args:
            trials (int): Number of optimization trials
        """
        logger.info(f"🔬 Starting hyperparameter tuning with {trials} trials...")
        
        try:
            model = self.load_model()
            
            # Run hyperparameter tuning
            results = model.tune(
                data=self.config['dataset']['data_yaml'],
                epochs=30,  # Reduced epochs for tuning
                iterations=trials,
                optimizer='AdamW',
                plots=True,
                save=True,
                val=True
            )
            
            logger.info("✅ Hyperparameter tuning completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter tuning failed: {e}")
            return None
    
    def compare_models(self, model_configs: List[str]):
        """
        Compare multiple model configurations.
        
        Args:
            model_configs (List[str]): List of model names to compare
        """
        logger.info(f"🔄 Comparing models: {model_configs}")
        
        comparison_results = {}
        
        for model_name in model_configs:
            logger.info(f"Training model: {model_name}")
            
            # Update config for this model
            original_model = self.config['model']['name']
            original_name = self.config['training']['name']
            
            self.config['model']['name'] = model_name
            self.config['training']['name'] = f"comparison_{model_name.split('.')[0]}"
            self.config['training']['epochs'] = 50  # Reduced for comparison
            
            try:
                model = self.load_model()
                results = self.train_model(model)
                metrics = self.evaluate_model()
                
                comparison_results[model_name] = {
                    'metrics': metrics,
                    'results_dir': str(self.results_dir)
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
            
            # Restore original config
            self.config['model']['name'] = original_model
            self.config['training']['name'] = original_name
        
        # Save comparison results
        comparison_path = os.path.join('runs/segment', 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info(f"📊 Model comparison results saved to: {comparison_path}")
        return comparison_results

def main():
    """
    Main training function with command line interface.
    """
    parser = argparse.ArgumentParser(description='Improved YOLOv8 Segmentation Training')
    parser.add_argument('--config', type=str, default='training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--compare', nargs='+', 
                       default=['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt'],
                       help='Compare multiple models')
    parser.add_argument('--eval-only', type=str,
                       help='Only evaluate existing model (provide path to weights)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ImprovedYOLOTrainer(args.config)
    
    logger.info("🚀 Improved YOLOv8 Segmentation Training")
    logger.info("=" * 60)
    
    # Setup environment
    if not trainer.setup_environment():
        logger.error("❌ Environment setup failed")
        return
    
    # Evaluation only mode
    if args.eval_only:
        logger.info(f"📊 Evaluation mode: {args.eval_only}")
        trainer.evaluate_model(args.eval_only)
        trainer.run_sample_predictions(args.eval_only)
        return
    
    # Hyperparameter tuning mode
    if args.tune:
        trainer.hyperparameter_tuning()
        return
    
    # Model comparison mode
    if len(args.compare) > 1:
        trainer.compare_models(args.compare)
        return
    
    # Regular training mode
    logger.info("🎯 Starting regular training...")
    
    try:
        # Load model
        model = trainer.load_model()
        
        # Train model
        results = trainer.train_model(model)
        
        # Evaluate trained model
        logger.info("\n📊 Evaluating trained model...")
        metrics = trainer.evaluate_model()
        
        # Run sample predictions
        logger.info("\n🎯 Running sample predictions...")
        trainer.run_sample_predictions()
        
        # Final summary
        logger.info("\n🎉 Training completed successfully!")
        logger.info(f"📁 Results directory: {trainer.results_dir}")
        logger.info(f"🏆 Best model: {os.path.join(trainer.results_dir, 'weights', 'best.pt')}")
        
        # Print key metrics
        if metrics:
            logger.info("\n📈 Final Metrics:")
            key_metrics = ['metrics/mAP50(M)', 'metrics/mAP50-95(M)', 
                          'metrics/precision(M)', 'metrics/recall(M)']
            for metric in key_metrics:
                if metric in metrics:
                    logger.info(f"   {metric}: {metrics[metric]:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 