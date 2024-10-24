from prompt_dtla.baselines.bc.bc import BCAgent
from prompt_dtla.baselines.decision_transformer.dt import (
    DecisionTransformerAgent,
)
from prompt_dtla.baselines.vpt.vpt import VPT
from prompt_dtla.baselines.vpt.vpt_policy import VPTAgent

NAME_TO_BASELINE_CLS = {
    "bc": BCAgent,
    "dt": DecisionTransformerAgent,
    "vpt": VPT,
    "vpt_policy": VPTAgent,
    # "dt": DecisionTransformer,
}
