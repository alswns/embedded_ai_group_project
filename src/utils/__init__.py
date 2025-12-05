"""
공통 유틸리티 모듈
"""
from .config import (
    setup_device,
    setup_matplotlib,
    get_image_transform,
    MODEL_PATH,
    TEST_IMAGE_DIR,
    CAPTIONS_FILE,
)
from .model_utils import (
    count_parameters,
    get_model_size_mb,
)
from .benchmark_utils import (
    get_peak_memory_mb,
    clear_memory,
    get_model_memory_mb,
    measure_inference_latency,
    measure_inference_latency_with_memory,
    calculate_meteor_batch,
    load_test_images_for_meteor,
    print_benchmark_result,
    calculate_model_size_mb,
    calculate_sparsity,
    calculate_size_reduction,
    run_benchmark,
)
from .metrics import (
    calculate_meteor,
    METEOR_AVAILABLE,
)
from .dataset import (
    CaptionDataset,
    load_test_data,
    prepare_calibration_dataset,
)
from .model_loader import (
    load_base_model,
)
from .pruning_utils import (
    count_nonzero_parameters,
    save_sparse_model,
    get_sparse_model_size_mb,
    get_pruning_mask,
    update_linear_layer,
    compute_channel_importance_hessian,
    apply_magnitude_pruning,
    apply_structured_pruning,
)
from .finetune_utils import (
    load_checkpoint,
    setup_training,
    save_checkpoint,
    print_checkpoint_info,
    restore_optimizer,
    load_model_checkpoint,
    load_model_from_checkpoint,
    apply_magnitude_mask,
    fine_tune_model,
)

from .quantization_utils import (
    setup_quantization_engine,
    apply_dynamic_quantization,
    apply_static_quantization,
    apply_qat,
    get_quantized_model_size_mb,
    print_quantization_stats,
)


__all__ = [
    # Config
    'setup_device',
    'setup_matplotlib',
    'get_image_transform',
    'MODEL_PATH',
    'TEST_IMAGE_DIR',
    'CAPTIONS_FILE',
    # Model utils
    'count_parameters',
    'get_model_size_mb',
    # Benchmark utils
    'get_peak_memory_mb',
    'clear_memory',
    'get_model_memory_mb',
    'measure_inference_latency',
    'measure_inference_latency_with_memory',
    'calculate_meteor_batch',
    'load_test_images_for_meteor',
    'print_benchmark_result',
    'calculate_model_size_mb',
    'calculate_sparsity',
    'calculate_size_reduction',
    'run_benchmark',
    # Metrics
    'calculate_meteor',
    'METEOR_AVAILABLE',
    # Dataset
    'CaptionDataset',
    'load_test_data',
    'prepare_calibration_dataset',
    # Model loader
    'load_base_model',
    # Pruning utils
    'count_nonzero_parameters',
    'save_sparse_model',
    'get_sparse_model_size_mb',
    'get_pruning_mask',
    'update_linear_layer',
    'compute_channel_importance_hessian',
    'apply_magnitude_pruning',
    'apply_structured_pruning',
    # Finetune utils
    'load_checkpoint',
    'setup_training',
    'save_checkpoint',
    'print_checkpoint_info',
    'restore_optimizer',
    'load_model_checkpoint',
    'load_model_from_checkpoint',
    'apply_magnitude_mask',
    'fine_tune_model',
    
    # Quantization utils
    'setup_quantization_engine',
    'apply_dynamic_quantization',
    'apply_static_quantization',
    'apply_qat',
    'get_quantized_model_size_mb',
    'print_quantization_stats',
    
]

