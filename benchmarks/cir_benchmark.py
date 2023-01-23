import torch

from avalanche.benchmarks.utils.classification_dataset import \
    classification_subset
from avalanche.benchmarks import dataset_benchmark

from utils.generic import set_random_seed


def generate_benchmark(
        seed,
        train_set,
        test_set,
        n_classes=None,
        n_e=None,
        scenario_table=None,
        first_occurrences=None,
        n_samples_table=None,
        indices_per_class=None,
):
    set_random_seed(seed)

    # Sample indices for each class
    selected_indices = {}

    def sample_per_class(c):
        selected_indices[c] = {}
        indices_c = indices_per_class[c][torch.randperm(
            len(indices_per_class[c]))]
        exp_c = torch.where(n_samples_table[c] != 0)[0]
        for e, n in zip(exp_c, n_samples_table[c][exp_c]):
            indices_c = indices_c[torch.randperm(len(indices_c))]
            selected = indices_c[:n]
            selected_indices[c][e.item()] = selected

    _ = [sample_per_class(c) for c in range(n_classes)]

    # Create dataset for each experience
    def create_dataset_exp_i(exp_i):
        present_classes = torch.where(scenario_table[:, exp_i] != 0)[0]
        all_indices_i = torch.cat([selected_indices[c.item()][exp_i]
                                   for c in present_classes])
        ds_i = classification_subset(train_set, indices=all_indices_i)

        return ds_i, all_indices_i

    stream_items = [create_dataset_exp_i(i) for i in range(n_e)]
    train_datasets = [t[0] for t in stream_items]
    samples_per_exp = [t[1] for t in stream_items]

    # Create benchmark
    benchmark = dataset_benchmark(
        train_datasets=train_datasets,
        test_datasets=[test_set],
    )

    # Benchmark Info
    # - Scenario table, n-Samples table and n-TrainSet
    benchmark.first_occurrences = first_occurrences
    benchmark.scenario_table = scenario_table
    benchmark.n_samples_table = n_samples_table
    benchmark.n_trainset = len(train_set)

    # - Samples and number of samples in each experience
    benchmark.samples_per_exp = samples_per_exp
    benchmark.n_samples_per_exp = [len(benchmark.samples_per_exp[i]) for i in
                                   range(scenario_table.shape[1])]

    # - Total number of samples
    all_selected_indices = torch.cat(samples_per_exp)
    benchmark.n_total_samples = len(all_selected_indices)

    # - Classes in each experience:
    benchmark.present_classes_in_each_exp = [
        torch.where(benchmark.scenario_table[:, i])[0]
        for i in range(n_e)
    ]

    # - Seen classes up to each experience
    def classes_seen_sofar(i):
        seen_classes = benchmark.present_classes_in_each_exp[:i + 1]
        seen_classes = set(torch.cat(seen_classes).numpy())
        seen_classes = torch.LongTensor(list(seen_classes))

        return seen_classes

    benchmark.seen_classes = [classes_seen_sofar(i) for i in
                              range(len(benchmark.present_classes_in_each_exp))]

    benchmark.n_classes = n_classes

    return benchmark


__all__ = ["generate_benchmark"]
