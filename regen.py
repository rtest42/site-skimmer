import os
import sys


def main():
    if len(sys.argv) < 3:
        directory = input('Enter the directory/folder: ')
        label = input('Assign a label: ')
    else:
        directory = sys.argv[1]
        label = sys.argv[2]

    with open('google-vertex-' + label + '.csv', 'w') as f:
        f.write('img_dir,label\n')

        files = os.listdir(directory)
        for file in files:
            f.write('gs://cloud-ml-data/' + directory + '/' + file + ',' + label + '\n')


if __name__ == '__main__':
    main()
