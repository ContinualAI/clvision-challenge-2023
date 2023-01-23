import dill

from .cir_benchmark import generate_benchmark


def get_cifar_based_benchmark(scenario_config, seed):
    # Load challenge datasets
    with open("./data/challenge_train_set.pkl", "rb") as pkl_file:
        train_set = dill.load(pkl_file)

    with open("./data/challenge_test_set.pkl", "rb") as pkl_file:
        test_set = dill.load(pkl_file)

    # Load scenario config
    with open(f"./scenario_configs/{scenario_config}", "rb") as pkl_file:
        scenario_config = dill.load(pkl_file)

    # Benchmarks
    benchmark = generate_benchmark(seed=seed, train_set=train_set,
                                   test_set=test_set, **scenario_config)

    return benchmark


__al__ = ["get_cifar_based_benchmark"]
