import argparse
import shutil


def copy_and_replace(source_file, destination_file):
    shutil.copy2(source_file, destination_file)


parser = argparse.ArgumentParser(description='Copy a file to another location and replace if it already exists')
parser.add_argument('--source_file', help='Path to the source file')
parser.add_argument('--dest_file', help='Path to the destination file')
args = parser.parse_args()

copy_and_replace(args.source_file, args.dest_file)
