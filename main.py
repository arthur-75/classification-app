# main.py
import argparse
import os
import pandas as pd
from training.utils import test_transform, CustomFolder, met
from training.model import model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run CAN model on input data and save predictions to a CSV file.')
    parser.add_argument('--input_folder', default='training/dataset/test_images', help='Path to the input data folder.')
    parser.add_argument('--output_csv', default="output.csv", help='Path to the output CSV file.')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Get input and output paths from the command-line arguments
    input_folder = args.input_folder
    output_csv = args.output_csv

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Initialize the model
    my_model = model()

    # Perform preprocessing and run your model
    predict_dataset = CustomFolder(root=input_folder, transform=test_transform, has_labels=False)
    predict_loader = predict_dataset.predction_loader(predict_dataset)
    int_pred = met(my_model, predict_loader)
    label_pred = ['Fields' if i == 0 else 'Roads' for i in int_pred]

    df = pd.DataFrame({'file': predict_dataset.image_paths, 'pred': int_pred, 'label': label_pred})
    df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
