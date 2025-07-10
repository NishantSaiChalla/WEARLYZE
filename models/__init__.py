"""
Fashion Detection Models Module.

This module provides comprehensive models for fashion detection, classification, and segmentation tasks.
Includes YOLOv8-based models for detection/segmentation and various CNN/Transformer models for classification.
"""

# YOLOv8 Models for Detection and Segmentation
from .yolo_config import (
    YOLOConfig,
    YOLOModelConfig,
    YOLOTrainingConfig,
    YOLOEvaluationConfig,
    YOLOExperimentConfig,
    get_default_config,
    get_fashion_config
)

from .yolo_segmentation import (
    FashionYOLOv8,
    FashionSegmentationLoss
)

from .yolo_trainer import (
    YOLOTrainer,
    EarlyStopping,
    MetricsTracker,
    create_trainer,
    train_fashion_yolo
)

from .yolo_utils import (
    YOLODataConverter,
    YOLOVisualizer,
    YOLOPostProcessor,
    calculate_iou,
    calculate_mask_iou,
    calculate_dice_score,
    create_yolo_dataset_yaml,
    load_class_names,
    save_class_names
)

# Classification Models
from .classifiers import (
    BaseFashionClassifier,
    FashionResNet,
    FashionMobileNet,
    FashionConvNeXt,
    FashionViT,
    FashionEfficientNet,
    FashionMultiScale,
    create_fashion_classifier,
    load_fashion_classifier
)

# Ensemble Models
from .ensemble import (
    BaseEnsemble,
    SoftVotingEnsemble,
    HardVotingEnsemble,
    WeightedEnsemble,
    StackedEnsemble,
    DynamicEnsemble,
    EnsembleClassifier,
    create_ensemble,
    evaluate_ensemble_diversity
)

# Loss Functions
from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    FashionTripletLoss,
    FashionContrastiveLoss,
    FashionCenterLoss,
    FashionArcFaceLoss,
    FashionMixupLoss,
    FashionCompositeLoss,
    create_fashion_loss
)

# Model Factory
from .model_factory import (
    ModelConfig,
    ModelFactory,
    OptimizationFactory,
    create_complete_training_setup
)

# Quantization and Compression
from .quantization import (
    QuantizationConfig,
    QuantizableModel,
    PostTrainingQuantizer,
    QuantizationAwareTraining,
    ModelPruner,
    ModelCompressor,
    save_compressed_model,
    load_compressed_model
)

# CLIP-based Fashion Models
from .clip_model import (
    FashionCLIPConfig,
    FashionCLIP,
    FashionCLIPTrainer,
    FashionTextEncoder,
    FashionImageEncoder,
    HardNegativeMiner,
    create_fashion_clip_model,
    load_fashion_clip_model
)

# Similarity Search and Retrieval
from .similarity_search import (
    SearchConfig,
    VectorDatabase,
    SimilaritySearchEngine,
    create_vector_database,
    optimize_index_parameters
)

# Retrieval Utilities
from .retrieval_utils import (
    RetrievalMetrics,
    HardNegativeMiner as RetrievalHardNegativeMiner,
    SimilarityMetrics,
    RetrievalEvaluator,
    EmbeddingVisualizer,
    evaluate_retrieval_system,
    mine_hard_negatives,
    compute_embedding_statistics
)

# Fashion-Specific Embeddings
from .fashion_embeddings import (
    FashionEmbeddingConfig,
    FashionAttributeEmbedding,
    TextureFeatureExtractor,
    StyleFeatureExtractor,
    ColorHistogramAnalyzer,
    CompositionAnalyzer,
    MultiModalFusion,
    FashionEmbeddingModel,
    create_fashion_embedding_model,
    extract_fashion_attributes
)

# Index Building for Large-Scale Retrieval
from .index_builder import (
    IndexBuildConfig,
    DatasetProcessor,
    IndexBuilder,
    IndexOptimizer,
    IndexValidator,
    MemoryMonitor,
    create_index_builder,
    estimate_memory_requirements,
    optimize_build_config
)

__all__ = [
    # YOLOv8 Configuration
    'YOLOConfig',
    'YOLOModelConfig',
    'YOLOTrainingConfig',
    'YOLOEvaluationConfig',
    'YOLOExperimentConfig',
    'get_default_config',
    'get_fashion_config',
    
    # YOLOv8 Models
    'FashionYOLOv8',
    'FashionSegmentationLoss',
    
    # YOLOv8 Training
    'YOLOTrainer',
    'EarlyStopping',
    'MetricsTracker',
    'create_trainer',
    'train_fashion_yolo',
    
    # YOLOv8 Utilities
    'YOLODataConverter',
    'YOLOVisualizer',
    'YOLOPostProcessor',
    'calculate_iou',
    'calculate_mask_iou',
    'calculate_dice_score',
    'create_yolo_dataset_yaml',
    'load_class_names',
    'save_class_names',
    
    # Classification Models
    'BaseFashionClassifier',
    'FashionResNet',
    'FashionMobileNet',
    'FashionConvNeXt',
    'FashionViT',
    'FashionEfficientNet',
    'FashionMultiScale',
    'create_fashion_classifier',
    'load_fashion_classifier',
    
    # Ensemble Models
    'BaseEnsemble',
    'SoftVotingEnsemble',
    'HardVotingEnsemble',
    'WeightedEnsemble',
    'StackedEnsemble',
    'DynamicEnsemble',
    'EnsembleClassifier',
    'create_ensemble',
    'evaluate_ensemble_diversity',
    
    # Loss Functions
    'FocalLoss',
    'LabelSmoothingLoss',
    'FashionTripletLoss',
    'FashionContrastiveLoss',
    'FashionCenterLoss',
    'FashionArcFaceLoss',
    'FashionMixupLoss',
    'FashionCompositeLoss',
    'create_fashion_loss',
    
    # Model Factory
    'ModelConfig',
    'ModelFactory',
    'OptimizationFactory',
    'create_complete_training_setup',
    
    # Quantization and Compression
    'QuantizationConfig',
    'QuantizableModel',
    'PostTrainingQuantizer',
    'QuantizationAwareTraining',
    'ModelPruner',
    'ModelCompressor',
    'save_compressed_model',
    'load_compressed_model',
    
    # CLIP-based Fashion Models
    'FashionCLIPConfig',
    'FashionCLIP',
    'FashionCLIPTrainer',
    'FashionTextEncoder',
    'FashionImageEncoder',
    'HardNegativeMiner',
    'create_fashion_clip_model',
    'load_fashion_clip_model',
    
    # Similarity Search and Retrieval
    'SearchConfig',
    'VectorDatabase',
    'SimilaritySearchEngine',
    'create_vector_database',
    'optimize_index_parameters',
    
    # Retrieval Utilities
    'RetrievalMetrics',
    'RetrievalHardNegativeMiner',
    'SimilarityMetrics',
    'RetrievalEvaluator',
    'EmbeddingVisualizer',
    'evaluate_retrieval_system',
    'mine_hard_negatives',
    'compute_embedding_statistics',
    
    # Fashion-Specific Embeddings
    'FashionEmbeddingConfig',
    'FashionAttributeEmbedding',
    'TextureFeatureExtractor',
    'StyleFeatureExtractor',
    'ColorHistogramAnalyzer',
    'CompositionAnalyzer',
    'MultiModalFusion',
    'FashionEmbeddingModel',
    'create_fashion_embedding_model',
    'extract_fashion_attributes',
    
    # Index Building for Large-Scale Retrieval
    'IndexBuildConfig',
    'DatasetProcessor',
    'IndexBuilder',
    'IndexOptimizer',
    'IndexValidator',
    'MemoryMonitor',
    'create_index_builder',
    'estimate_memory_requirements',
    'optimize_build_config'
]

__version__ = '1.0.0'
__author__ = 'Fashion Detection Team'
__description__ = 'Comprehensive fashion detection models including YOLOv8 for detection/segmentation and CLIP-based similarity retrieval'