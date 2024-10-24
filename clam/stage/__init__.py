from prompt_dtla.stage.action_decoder import LatentActionDecoder
from prompt_dtla.stage.lam.lam import LatentActionModel
from prompt_dtla.stage.lam.lam_and_action_decoder import LAMAndDecoder
from prompt_dtla.stage.latent_action_policy import LatentActionAgent

NAME_TO_STAGE_CLS = {
    "lam": LatentActionModel,
    "latent_action_decoder": LatentActionDecoder,
    "la_bc": LatentActionAgent,
    "lam_and_action_decoder": LAMAndDecoder,
}
