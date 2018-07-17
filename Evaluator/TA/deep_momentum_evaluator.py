"""
OctoBot Tentacle

$tentacle_description: {
    "package_name": "OctoBot-IA-Tentacles",
    "name": "deep_momentum_evaluator",
    "type": "Evaluator",
    "subtype": "TA",
    "version": "1.0.0",
    "requirements": [],
    "tests":[],
    "resource_files": ["test_resource.data"]
}
"""

from evaluator.TA.TA_evaluator import MomentumEvaluator


class RSIMomentumEvaluator(MomentumEvaluator):
    def __init__(self):
        super().__init__()
        self.pertinence = 1

    def eval_impl(self):
        pass
