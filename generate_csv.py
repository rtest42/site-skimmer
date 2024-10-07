import os
import sys


# Generates a CSV file for Google Vertex
def generate_csv(directory: str, label: str) -> None:
    with open(f"google-vertex-{label}.csv", 'w') as f:
        f.write("img_dir,label\n")
        files = os.listdir(directory)
        for file in files:
            f.write(f"gs://cloud-ml-data/{directory}/{file},{label}\n")


# For debugging
def main(args=sys.argv) -> None:
    if len(args) < 3:
        print("Usage: python3 generate_csv.py <directory> <label>")
        return
    
    directory = args[1]
    label = args[2]

    generate_csv(directory, label)


if __name__ == '__main__':
    main()
