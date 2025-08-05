import argparse
import os
import sys
import glob
import utils.generate_predictions_util as dgpu
import utils.model_architecture_util as dmau

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction script for epiphany")
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--windowSize', type=int, required=True, help='Window size for prediction', default=14000)
    parser.add_argument('--chromosomes', nargs='+', required=True, help='List of chromosomes to predict')
    parser.add_argument('--chromosomeSizes', required=False, help='File with chromosome sizes')
    parser.add_argument('--bigwigFolder', required=True, help='Folder containing input bigwig files')
    parser.add_argument('--outputFolder', required=True, help='Folder to write prediction outputs')
    parser.add_argument('--submatrixName', required=False, help='Path to the submatrix location file', default='submatrix_location.txt')
    parser.add_argument('--assembleMatrixLocation', required=False, help='Path to the assemble matrix location file', default='assemble_matrix_location.txt')
    parser.add_argument('--groundTruthFile', required=False, help='Path to the ground truth file')
    parser.add_argument('--groundTruthOutput', required=False, help='Path to the output ground truth location file')
    parser.add_argument('--chromosomeSizesFile', required=False, help='Path to chromosome sizes file')
    parser.add_argument('--resolution', type=int, required=False, help='Resolution for Hi-C data', default=10000)
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
                          chromosome,
                          window_size,
                          bigwig_folder,
                          submatrix_path,
                          assemble_matrix_path,
                          ground_truth_path,
                          ground_truth_output_path,
                          cell_type="GM12878",
                          chromosomeSizesFile=None,
                          resolution=10000
                        ):
    print(f"Predicting on {chromosome} with window size {window_size}")
    
    dgpu.results_generation(
        chrom=chromosome,
        net=model,
        cell_type=cell_type,
        bwfile_dir=bigwig_folder,
        submatrix_location=submatrix_path,
        assemble_matrix_location=assemble_matrix_path,
        ground_truth_file=ground_truth_path,
        ground_truth_location=ground_truth_output_path,
        window_size=window_size,
        resolution_hic=resolution,
        chromosomeSizesFile=chromosomeSizesFile
    )

def main():
    args = parse_args()
    os.makedirs(args.outputFolder, exist_ok=True)
    submatrixName = os.path.join(args.outputFolder, args.submatrixName)
    assembleMatrixLocation = os.path.join(args.outputFolder, args.assembleMatrixLocation)
    net = dmau.Net(window_size=args.windowSize)
    dmau.restore(net, args.model)
    net.eval()
    for chromosome in args.chromosomes:
        predict_on_chromosome(
            model=net,
            chromosome=chromosome,
            window_size=args.windowSize,
            bigwig_folder=args.bigwigFolder,
            submatrix_path=submatrixName,
            assemble_matrix_path=assembleMatrixLocation,
            ground_truth_path=args.groundTruthFile,
            ground_truth_output_path=args.groundTruthOutput,
            cell_type="GM12878",
            chromosomeSizesFile=args.chromosomeSizesFile,
            resolution=args.resolution
        )

if __name__ == "__main__":
    main()