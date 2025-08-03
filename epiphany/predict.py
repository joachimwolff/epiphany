import argparse
import os
import sys
import glob
import utils.generate_predictions_util as dgpu
import utils.model_architecture_util as dmau

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction script for epiphany")
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--window-size', type=int, required=True, help='Window size for prediction', default=14000)
    parser.add_argument('--chromosomes', nargs='+', required=True, help='List of chromosomes to predict')
    parser.add_argument('--chrom-sizes', required=False, help='File with chromosome sizes')
    parser.add_argument('--bigwig-folder', required=True, help='Folder containing input bigwig files')
    parser.add_argument('--output-folder', required=False, help='Folder to write prediction outputs')
    parser.add_argument('--submatrix-location', required=True, help='Path to the submatrix location file')
    parser.add_argument('--assemble-matrix-location', required=True, help='Path to the assemble matrix location file')
    parser.add_argument('--ground-truth-file', required=False, help='Path to the ground truth file')
    parser.add_argument('--ground-truth-output', required=False, help='Path to the output ground truth location file')
    return parser.parse_args()

# def read_chrom_sizes(chrom_sizes_file, chromosomes):
#     chrom_sizes = {}
#     with open(chrom_sizes_file) as f:
#         for line in f:
#             chrom, size = line.strip().split()[:2]
#             if chrom in chromosomes:
#                 chrom_sizes[chrom] = int(size)
#     return chrom_sizes



def predict_on_chromosome(model,
                            chrom,
                            # size,
                            window_size,
                            bigwig_folder,
                            # output_folder,
                            submatrix_location,
                            assemble_matrix_location,
                            ground_truth_file,
                            ground_truth_output,
                            cell_type="GM12878"
                        ):
    print(f"Predicting on {chrom} with window size {window_size}")

    dgpu.results_generation(
        chrom=chrom,
        net=model,
        cell_type=cell_type,
        bwfile_dir=bigwig_folder,
        submatrix_location=submatrix_location,
        assemble_matrix_location=assemble_matrix_location,
        ground_truth_file=ground_truth_file,
        ground_truth_location=ground_truth_output,
        window_size=window_size,
    )

def main():
    args = parse_args()
    # os.makedirs(args.output_folder, exist_ok=True)
    # chrom_sizes = read_chrom_sizes(args.chrom_sizes, args.chromosomes)
    # wsize = 14000
    net = dmau.Net(window_size=args.window_size)
    dmau.restore(net,args.model)
    net.eval()
    for chrom in args.chromosomes:
        # if chrom not in chrom_sizes:
        #     print(f"Warning: Chromosome {chrom} not found in sizes file.", file=sys.stderr)
        #     continue
        predict_on_chromosome(
            net,
            chrom,
            # chrom_sizes[chrom],
            args.window_size,
            args.bigwig_folder,
            # args.output_folder,
            args.submatrix_location,
            args.assemble_matrix_location,
            args.ground_truth_file,
            args.ground_truth_output,
        )

if __name__ == "__main__":
    main()