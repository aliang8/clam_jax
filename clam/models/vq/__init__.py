from prompt_dtla.models.vq.fsq import FSQ
from prompt_dtla.models.vq.residual_vq import ResidualVQ
from prompt_dtla.models.vq.vq import VectorQuantize
from prompt_dtla.models.vq.vq_ema import VQEmbeddingEMA

NAME_TO_VQ_CLS = {
    "vq": VectorQuantize,
    "vq_ema": VQEmbeddingEMA,
    "fsq": FSQ,
    "residual_vq": ResidualVQ,
}
