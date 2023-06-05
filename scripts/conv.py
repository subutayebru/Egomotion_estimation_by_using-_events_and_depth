import pandas as pd
import sys

def convert_tsv_to_csv(input_tsv_path, output_csv_path):
    # reading given tsv file
    tsv_table = pd.read_csv(input_tsv_path, sep="\t")
    
    # converting tsv file into csv
    tsv_table.to_csv(output_csv_path, sep=",", index=False)

    print("Successfully made csv file.")

# specify the paths to the input TSV file and the output CSV file
input_tsv_path = sys.argv[1]
output_csv_path = sys.argv[2]

# convert the TSV file to a CSV file

convert_tsv_to_csv(input_tsv_path, output_csv_path)
