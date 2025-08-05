import numpy as np
from ray import tune
# from ray.tune.suggest.hyperopt import HyperOptSearch
# from hyperopt import hp
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session
# from hyperopt import tpe, hp
import os
import logging as log
from adversarial import train_adversarial
from predict import predict_on_chromosome
import argparse
import cooler
from scipy.stats import pearsonr
from sklearn.metrics import auc
from collections import defaultdict

import utils.generate_predictions_util as dgpu
import utils.model_architecture_util as dmau

import pygenometracks.plotTracks
from sklearn.metrics import auc
import traceback


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for adversarial training")
    parser.add_argument('--dataX', type=str, required=True, help='Path to input data X')
    parser.add_argument('--dataY', type=str, required=True, help='Path to input data Y')
    parser.add_argument('--bigwig_folder', type=str, required=True, help='Path to the folder containing bigwig files')
    parser.add_argument('--numberSamples', type=int, default=10, help='Number of optimization trials')
    parser.add_argument('--outputFolder', type=str, required=True, help='Directory to store output results')
    parser.add_argument('--genomicRegion', type=str, default=None, help='Region to plot (optional)')
    parser.add_argument('--continue_experiment', type=str, default=None, help='Path to continue previous experiment')
    parser.add_argument('--threads', type=int, default=1, help='Number of CPU threads to use')
    parser.add_argument('--gpu', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--train_chromosomes', type=str, nargs='+', required=True, help='Comma-separated list of training chromosomes')
    parser.add_argument('--validation_chromosomes', type=str, nargs='+', required=False, help='Comma-separated list of validation chromosomes')
    parser.add_argument('--prediction_chromosomes', type=str, nargs='+', required=True, help='Comma-separated list of test chromosomes')
    # parser.add_argument('--output', type=str, required=True, help='Output folder for results')
    parser.add_argument('--epochs', type=int, default=55, help='Number of training epochs')
    parser.add_argument('--comparisonMatrix', type=str, required=False, help='Path to comparison matrix cooler file')
    parser.add_argument('--chromosomeSizeFile', type=str, required=False, help='Path to chromosome size file')
    parser.add_argument('--resolution', type=int, default=10000, help='Resolution for Hi-C matrices')
    parser.add_argument('--trainingCellType', type=str, required=False, default="GM12878", help='Cell type used for training (default: GM12878)')
    return parser.parse_args()


def objective(config, pArgs):
    # Prepare argument parser and parse default arguments
    log.debug(f"Starting objective with config: {config} and trial_id: {session.get_trial_id()}")
    trial_id = session.get_trial_id()
    print("Starting train_adversarial function")
    assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print((f"Ray assigned GPU devices_hyper: {assigned_gpus}"))
  
   


    # log.debug(f"Arguments for train_adversarial_main: {args_list}")
    # Call the training/prediction function
    # Extract parameters from args_list and config
    train_adversarial(gpu = int(config.get("gpu", 1)),
                      batchSize = int(config.get("batch_size", 1)),
                      epochs = int(pArgs.epochs),
                      lr = float(config.get("learning_rate_generator", 1e-4)),
                      version = str(trial_id),
                      lam = float(config.get("loss_weight_adversarial", 0.95)),
                      windowSize = int(config.get("window_size", 14000)),
                      message = "",  # You can set this if needed
                      highRes = config.get("high_res", False),
                      wandb = config.get("wandb", False),
                      xFile = pArgs.dataX,
                      yFile = pArgs.dataY,
                      pretrainedModel = None,  # Set if you want to use a pretrained model
                      outputFolder = os.path.join(pArgs.outputFolder),
                      trainingChromosomes = pArgs.train_chromosomes,
                      validationChromosomes = pArgs.validation_chromosomes)

    # Determine model filename based on epoch count

    epochs = int(pArgs.epochs)-1
    model_name = f"{epochs:03d}.pt_model"
    net = dmau.Net(window_size=int(config.get("window_size", 14000)))
    model_path = os.path.join(pArgs.outputFolder, trial_id, f"{model_name}")
    dmau.restore(net, model_path)
    net.eval()
    for chrom in pArgs.prediction_chromosomes:
        log.debug(f"Predicting on chromosome: {chrom}")
        predict_on_chromosome(
            model=net,
            chromosome=chrom,
            window_size=int(config.get("window_size", 14000)),
            bigwig_folder=pArgs.bigwig_folder,
            submatrix_path=os.path.join(pArgs.outputFolder, trial_id, f"submatrix_location_{chrom}.txt"),
            assemble_matrix_path=os.path.join(pArgs.outputFolder, trial_id, f"assemble_matrix_location_{chrom}.txt"),
            ground_truth_path=None,
            ground_truth_output_path=None,
            cell_type="GM12878",
            chromosomeSizesFile=pArgs.chromosomeSizeFile,
            resolution=pArgs.resolution  # Default resolution if not provided
        )
    # predict_on_chromosome(
    #     model=os.path.join(pArgs.outputFolder, f"model_{trial_id}.pt"),
    #     chromosome=pArgs.prediction_chromosomes,
    #     window_size=int(config.get("window_size", 14000)),
    #     bigwig_folder=pArgs.bigwig_folder,
    #     # outputFolder=os.path.join(pArgs.outputFolder, trial_id),
    #     submatrix_path=os.path.join(pArgs.outputFolder, trial_id, "submatrix_location.txt"),
    #     assemble_matrix_path=os.path.join(pArgs.outputFolder, trial_id, "assemble_matrix_location.txt"),
    #     ground_truth_path=None,
    #     ground_truth_output_path=None,
    #     cell_type="GM12878",
    #     chromosomeSizesFile=pArgs.chromosomeSizeFile,
    #     resolution=pArgs.resolution  # Default resolution if not provided
    # )

    # log.info(f"AUC of distance-dependent Pearson correlation: {auc_score}")


    def compute_distance_correlation_auc(cooler_file1, cooler_file2, chromosome, distance=1000000):
        c1 = cooler.Cooler(cooler_file1)
        c2 = cooler.Cooler(cooler_file2)
        assert c1.binsize == c2.binsize, "Cooler files must have the same bin size"
        # assert c1.chromnames == c2.chromnames, "Cooler files must have the same chromosomes"
        
        # Ensure chromosome names match the "chr" prefix style
        # Unify chromosome name to match the style used in c1
        if c1.chromnames[0].startswith("chr") and chromosome.startswith("chr"):
            chrom1 = chromosome
        elif c1.chromnames[0].startswith("chr") and not chromosome.startswith("chr"):
            chrom1 = "chr" + chromosome
        elif not c1.chromnames[0].startswith("chr") and chromosome.startswith("chr"):
            chrom1 = chromosome.replace("chr", "", 1)

        if c2.chromnames[0].startswith("chr") and not chromosome.startswith("chr"):
            chrom2 = "chr" + chromosome
        elif not c2.chromnames[0].startswith("chr") and chromosome.startswith("chr"):
            chrom2 = chromosome.replace("chr", "", 1)
        else:
            chrom2 = chromosome

        print(c1.chromnames[0], c2.chromnames[0])
        print(chrom1, chrom2)
        distances = []
        correlations = []

        # for chrom1, chrom2 in zip(chroms1, chroms2):
        mat1 = c1.matrix(balance=False).fetch(chrom1)
        mat2 = c2.matrix(balance=False).fetch(chrom2)

        n = mat1.shape[0]
        max_bin_distance = distance // c1.binsize
        for d in range(1, min(max_bin_distance, n // 2)):
            vals1 = mat1.diagonal(d)
            vals2 = mat2.diagonal(d)
            print(f"Processing distance {d} for chromosome {chrom1} and {chrom2}")
            print(f"Length of vals1: {len(vals1)}, Length of vals2: {len(vals2)}")
            mask = (~np.isnan(vals1)) & (~np.isnan(vals2))
            if np.sum(mask) > 1:
                corr, _ = pearsonr(vals1[mask], vals2[mask])
                if np.isnan(corr):
                    corr = 0
                print(f"Pearson correlation for distance {d}: {corr}")
                distances.append(d * c1.binsize)
                correlations.append(corr)

        # Aggregate by distance (since multiple chromosomes may contribute)
        dist_corrs = defaultdict(list)
        for d, c in zip(distances, correlations):
            dist_corrs[d].append(c)
        sorted_distances = sorted(dist_corrs.keys())
        mean_corrs = [np.mean(dist_corrs[d]) for d in sorted_distances]

        auc_value = auc(sorted_distances, mean_corrs)

        return auc_value, sorted_distances, mean_corrs

    # Example usage:
    auc_score_list = []
    for chrom in pArgs.prediction_chromosomes:
        log.debug(f"Computing AUC for chromosome: {chrom}")
        auc_score, sorted_distances, mean_corrs = compute_distance_correlation_auc(
            os.path.join(pArgs.outputFolder, trial_id, f"assemble_matrix_location_{chrom}.cool"),
            pArgs.comparisonMatrix, chromosome=chrom
        )
        auc_score_list.append(auc_score)
        
        log.info(f"AUC for chromosome {chrom}: {auc_score}")
    auc_score = np.mean(auc_score_list)
    if auc_score <= 0:
        auc_score = 0.0001  # Avoid zero AUC for plotting purposes
    # auc_score = compute_distance_correlation_auc(pArgs.comparisonMatrix, os.path.join(pArgs.outputFolder, trial_id, "assemble_matrix.cool"))
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



        """.format(os.path.join(pArgs.outputFolder, trial_id, f"assemble_matrix_location_{chrom}.cool"), pArgs.comparisonMatrix, score_text, pArgs.trainingCellType, 2000000)

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
    return auc_score  # This will be used as the objective metric for Ray Tune  

def objective_raytune(config, pArgs, pMetric):
    log.debug(f"Running objective_raytune with config: {config}")
    score = objective(config, pArgs)
    log.debug(f"Objective returned score: {score}")
    train.report({pMetric: score})

def run_raytune(pArgs):
    os.makedirs(os.path.join(pArgs.outputFolder,
                "pygenometracks"), exist_ok=True)
    os.makedirs(pArgs.outputFolder, exist_ok=True)

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
            "batch_size": 1,
            "window_size": 14000
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

    # if pArgs.optimizer == "hyperopt":
    log.debug("Use HyperOptSearch")
    search_algorithm = HyperOptSearch(metric=metric,
                                    mode=mode,
                                    points_to_evaluate=points_to_evaluate,
                                    )
    # elif pArgs.optimizer == "optuna":
    #     log.debug("Use OptunaSearch")
    #     search_algorithm = OptunaSearch(metric=metric, 
    #                                 mode=mode,
    #                                 points_to_evaluate=points_to_evaluate)

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