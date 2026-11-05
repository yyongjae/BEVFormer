from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .bevformer import *

# BEVFormerV2's auxiliary DD3D head needs detectron2 / fvcore / timm, which the
# NAVSIM port (BEVFormer v1, camera-only) does not use and which are absent from
# an environment built from docs/install.md steps a-e.  Importing it eagerly made
# the whole plugin unimportable, so it is optional now: with the extras installed
# nothing changes, without them only the V2 path is unavailable.
try:
    from .dd3d import *
except ImportError as _dd3d_import_error:  # pragma: no cover - env dependent
    import warnings

    warnings.warn(
        "projects.mmdet3d_plugin.dd3d is unavailable ({}); the BEVFormerV2 / DD3D "
        "path is disabled. Install detectron2, fvcore and timm (docs/install.md "
        "step f) to enable it. BEVFormer v1 and the NAVSIM port do not need "
        "it.".format(_dd3d_import_error)
    )
