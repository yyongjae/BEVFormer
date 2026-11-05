from .nuscenes_dataset import CustomNuScenesDataset
from .navsim_dataset import CustomNavsimDataset

# CustomNuScenesDatasetV2 pulls in the DD3D mono dataset, hence detectron2 /
# fvcore / timm.  Optional for the same reason as projects/mmdet3d_plugin/dd3d
# (see the note there): BEVFormer v1 and the NAVSIM port do not use it.
try:
    from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2
except ImportError:  # pragma: no cover - env dependent
    CustomNuScenesDatasetV2 = None

from .builder import custom_build_dataset
__all__ = [
    'CustomNuScenesDataset',
    'CustomNuScenesDatasetV2',
    'CustomNavsimDataset',
]
