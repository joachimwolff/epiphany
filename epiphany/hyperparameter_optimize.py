import numpy as np
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session
from hyperopt import tpe, hp
import os
import logging as log
from epiphany.adversarial import main as train_adversarial_main
from epiphany.predict import predict_on_chromosome
import argparse
import cooler
from scipy.stats import pearsonr
from sklearn.metrics import auc
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for adversarial training")
    parser.add_argument('--dataX', type=str, required=True, help='Path to input data X')
    parser.add_argument('--dataY', type=str, required=True, help='Path to input data Y')
    parser.add_argument('--bigwig_folder', type=str, required=True, help='Path to the folder containing bigwig files')
    parser.add_argument('--numberSamples', type=int, default=10, help='Number of optimization trials')
    parser.add_argument('--outputFolder', type=str, required=True, help='Directory to store output results')
    parser.add_argument('--plot_region', type=str, default=None, help='Region to plot (optional)')
    parser.add_argument('--continue_experiment', type=str, default=None, help='Path to continue previous experiment')
    parser.add_argument('--threads', type=int, default=1, help='Number of CPU threads to use')
    parser.add_argument('--gpu', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--train_chromosomes', type=str, required=True, help='Comma-separated list of training chromosomes')
    parser.add_argument('--validation_chromosomes', type=str, required=False, help='Comma-separated list of validation chromosomes')
    parser.add_argument('--prediction_chromosomes', type=str, required=True, help='Comma-separated list of test chromosomes')
    parser.add_argument('--output', type=str, required=True, help='Output folder for results')
    parser.add_argument('--epochs', type=int, default=55, help='Number of training epochs')
    return parser.parse_args()


def objective(config, pArgs):
    # Prepare argument parser and parse default arguments

    trial_id = session.get_trial_id()

    # Map Ray Tune config to argparse arguments
    args_list = [
        "--gpu", str(config.get("gpu", "0")),
        "--b", str(config.get("batch_size", "1")),
        "--e", str(pArgs.epochs),
        "--lr", str(config.get("learning_rate_generator", "1e-4")),
        "--v", str(trial_id),
        "--lam", str(config.get("loss_weight_adversarial", "0.95")),
        "--window_size", str(config.get("window_size", "14000")),
        "--dataX", pArgs.dataX,
        "--dataY", pArgs.dataY,
        "--outputFolder", pArgs.outputFolder,
        "--train_chromosomes", pArgs.train_chromosomes,
        "--test_chromosomes", pArgs.validation_chromosomes,
        "--plot_region", pArgs.plot_region
    ]
    if config.get("high_res", False):
        args_list.append("--high_res")
    if config.get("wandb", False):
        args_list.append("--wandb")


    # Call the training/prediction function
    result = train_adversarial_main(args_list)

    predict_on_chromosome(model=os.path.join(pArgs.outputFolder, f"model_{trial_id}.pt"),
                            chrom=pArgs.prediction_chromosomes.split(',')[0],
                            # size,
                            window_size=int(config.get("window_size", "14000")),
                            bigwig_folder=pArgs.bigwig_folder,
                            # output_folder,
                            submatrix_location=os.path.join(pArgs.outputFolder, "submatrix.txt"),
                            assemble_matrix_location=os.path.join(pArgs.outputFolder, "assemble_matrix.txt"),
                            ground_truth_file=None,
                            ground_truth_output=None,
                            cell_type="GM12878")


    def compute_distance_correlation_auc(cooler_file1, cooler_file2):
        c1 = cooler.Cooler(cooler_file1)
        c2 = cooler.Cooler(cooler_file2)
        assert c1.binsize == c2.binsize, "Cooler files must have the same bin size"
        assert c1.chromnames == c2.chromnames, "Cooler files must have the same chromosomes"

        distances = []
        correlations = []

        for chrom in c1.chromnames:
            mat1 = c1.matrix(balance=False).fetch(chrom)
            mat2 = c2.matrix(balance=False).fetch(chrom)
            n = mat1.shape[0]
            for d in range(1, n):
                vals1 = mat1.diagonal(d)
                vals2 = mat2.diagonal(d)
                mask = (~np.isnan(vals1)) & (~np.isnan(vals2))
                if np.sum(mask) > 1:
                    corr, _ = pearsonr(vals1[mask], vals2[mask])
                    distances.append(d * c1.binsize)
                    correlations.append(corr)

        # Aggregate by distance (since multiple chromosomes may contribute)
        dist_corrs = defaultdict(list)
        for d, c in zip(distances, correlations):
            dist_corrs[d].append(c)
        sorted_distances = sorted(dist_corrs.keys())
        mean_corrs = [np.mean(dist_corrs[d]) for d in sorted_distances]

        auc_value = auc(sorted_distances, mean_corrs)
        return auc_value

    # Example usage:
    auc_score = compute_distance_correlation_auc("file1.cool", "file2.cool")
    # print("AUC of distance-dependent Pearson correlation:", auc_score)
    if pArgs.genomicRegion:
        log.debug("Plot tracks")
            
        score_text = str(auc_score)
        os.makedirs(os.path.join(pArgs.outputFolder, "scores_txt"), exist_ok=True)
        score_file_path = os.path.join(pArgs.outputFolder, "scores_txt", trial_id + "_score_summary.txt")

        with open(score_file_path, 'w') as score_file:
            score_file.write(score_text)
        
        score_text = score_text.replace("\n", "; ")
        browser_tracks_with_hic = """
[hic matrix]
file = {0}
title = {2}
depth = {4}
transform = log1p
file_type = hic_matrix
show_masked_bins = false

[spacer]
height = 0.5

[TAD seperation score]
file = {5}
height = 2
type = lines
individual_color = grey
pos_score_in_bin = center
summary_color = #1f77b4
show_data_range = true
file_type = bedgraph_matrix

[spacer]
height = 1

[hic matrix]
file = {1}
title = original matrix {3}
depth = {4}
transform = log1p
file_type = hic_matrix
show_masked_bins = false
orientation = inverted

[spacer]
height = 0.5

[TAD seperation score]
file = {6}
height = 2
type = lines
individual_color = grey
pos_score_in_bin = center
summary_color = #1f77b4
show_data_range = true
file_type = bedgraph_matrix
        """.format(os.path.join(pArgs.outputFolder, trial_id, pArgs.matrixOutputName), pArgs.originalDataMatrix, score_text, pArgs.trainingCellType, 2000000, \
                os.path.join(pArgs.outputFolder, trial_id, "tads_predicted", 'tads_tad_score.bm'),
                    os.path.join(pArgs.outputFolder, trial_id, "tads_original", "tads_tad_score.bm"))
            

        tracks_path = os.path.join(
            pArgs.outputFolder, "browser_tracks_hic.ini")
        with open(tracks_path, 'w') as fh:
            fh.write(browser_tracks_with_hic)

        outfile = os.path.join(
            pArgs.outputFolder, "pygenometracks", trial_id + ".pdf")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        arguments = f"--tracks {tracks_path} --region {pArgs.genomicRegion} "\
                    f"--outFileName {outfile} --trackLabelFraction 0.1 --width 38 --height 35".split()
        try:
            pygenometracks.plotTracks.main(arguments)
        except Exception as e:
            traceback.print_exc()
            print(e)
    # Report the result to Ray Tune
    session.report({"accuracy": result})

def objective_raytune(config, pArgs, pMetric):

    score = objective(config, pArgs)
    train.report({pMetric: score})

def run_raytune(pArgs):
    os.makedirs(os.path.join(pArgs.outputFolder,
                "pygenometracks"), exist_ok=True)
    if pArgs.scoring == 'polymodel':
        if not os.path.exists(pArgs.polynomialModelPath):
            raise FileNotFoundError(f"Polynomial model file not found: {pArgs.polynomialModelPath}")

    # Create a ray tune experiment
    # Define the search space
    log.debug("Define search space")
    search_space = {
        "batch_size": tune.randint(1, 128),
        "learning_rate_generator": tune.uniform(1e-7, 1e-2),
        "loss_weight_adversarial": tune.uniform(0.5, 1.5),
        "window_size": tune.choice([14000]),
        # "high_res": tune.choice([True, False]),
        # "wandb": tune.choice([True, False]),
    }
    log.debug("Define points to evaluate")
    points_to_evaluate = [
        {   
            "loss_weight_adversarial": 0.9248942024710739,
            "learning_rate_generator": 0.0006947782705665501,
            "batch_size": 1
        }
    ]

      
        # Define the objective function
        # objective = tune.function(objective_raytune)
    metric = 'accuracy'
    mode = 'max'
    log.debug("Define objective function")
    objective_with_param = tune.with_parameters(objective_raytune, pArgs=pArgs,
                                                pMetric=metric)
    log.debug("Define objective function with resources")
    objective_with_resources = tune.with_resources(objective_with_param, resources={"cpu": pArgs.threads, "gpu": pArgs.gpu})

    if pArgs.optimizer == "hyperopt":
        log.debug("Use HyperOptSearch")
        search_algorithm = HyperOptSearch(metric=metric,
                                        mode=mode,
                                        points_to_evaluate=points_to_evaluate,
                                        )
    elif pArgs.optimizer == "optuna":
        log.debug("Use OptunaSearch")
        search_algorithm = OptunaSearch(metric=metric, 
                                    mode=mode,
                                    points_to_evaluate=points_to_evaluate)

    if pArgs.continue_experiment is None or pArgs.continue_experiment == "":
        log.debug("Start new experiment")
        tuner = tune.Tuner(
            objective_with_resources, 
            param_space=search_space, 
            tune_config=tune.TuneConfig(num_samples=pArgs.numberSamples,
                                        search_alg=search_algorithm),
        )
    else:
        log.debug("Continue experiment")
        tuner = tune.Tuner.restore(path=pArgs.continue_experiment, trainable=objective_with_resources)

    log.debug("Start tuning")    
    results = tuner.fit()

    log.debug("Get best result")
    print(results.get_best_result(metric=metric, mode=mode).config)

def run_opttuner():
    pass

def main(args=None):
    args = parse_arguments()
    run_raytune(pArgs=args)
    log.debug("Finished")

if __name__ == "__main__":
    main()