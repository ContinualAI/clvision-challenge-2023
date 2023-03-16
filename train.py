#######################################
#     DO NOT CHANGE THESE IMPORTS

import sys
sys.path.insert(0, "avalanche")

#######################################

import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import numpy as np

from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, LwFPlugin
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)

from benchmarks import get_cifar_based_benchmark
from models import SlimResNet18
from utils.competition_plugins import (
    GPUMemoryChecker,
    RAMChecker,
    TimeChecker
)

from strategies.my_plugin import MyPlugin
from strategies.my_strategy import MyStrategy


def main(args):
    # --- Device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- Benchmark
    benchmark = get_cifar_based_benchmark(scenario_config=args.config_file,
                                          seed=args.seed)

    # --- Model
    model = SlimResNet18(n_classes=benchmark.n_classes)

    # --- Logger and metrics
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # --- Competition Plugins
    # DO NOT REMOVE OR CHANGE THESE PLUGINS:
    competition_plugins = [
        GPUMemoryChecker(max_allowed=3500, device=device),
        RAMChecker(max_allowed=5000),
        TimeChecker(max_allowed=50)
    ]

    # --- Your Plugins
    plugins = [
        EWCPlugin(ewc_lambda=0.5),
        LwFPlugin(alpha=1.0),
        MyPlugin()
    ]

    # --- Strategy
    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=64,
        train_epochs=20,
        eval_mb_size=100,
        device=device,
        plugins=competition_plugins + plugins,
        evaluator=eval_plugin,
    )

    # --- Training Loops
    for experience in benchmark.train_stream:
        cl_strategy.train(experience, num_workers=args.num_workers)

    # --- Make prediction on test-set samples
    predictions = predict_test_set(cl_strategy.model,
                                   benchmark.test_stream[0].dataset,
                                   device)
    
    # Save predictions
    output_name = f"pred_{args.config_file.split('.')[0]}_{args.run_name}.npy"
    np.save(output_name, predictions)


def predict_test_set(model, test_set, device):
    print("Making prediction on test-set samples")

    model.eval()
    dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x, _, _) in dataloader:
            pred = model(x.to(device)).detach().cpu()
            preds.append(pred)

    preds = torch.cat(preds, dim=0)
    preds = torch.argmax(preds, dim=1).numpy()

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0,
                        help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="config_test.pkl")
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    main(args)
