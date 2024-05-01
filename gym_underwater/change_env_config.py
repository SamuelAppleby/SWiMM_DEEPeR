import argparse
import shutil


def copy_and_replace(source_file, destination_file):
    shutil.copy2(source_file, destination_file)


parser = argparse.ArgumentParser(description="Copy a file to another location and replace if it already exists")
parser.add_argument("source_file", help="Path to the source file")
args = parser.parse_args()

dest_file = "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\configs\\env_config.yml"

copy_and_replace(args.source_file, dest_file)
