# Implement your strategy in this file

from typing import List, Optional

from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate


class MyStrategy(SupervisedTemplate):
    """
    Implemention of MyStrategy.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

    def _before_training_exp(self, **kwargs):
        # Empty callback
        super()._before_training_exp(**kwargs)
