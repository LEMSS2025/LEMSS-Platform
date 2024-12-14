import json
import argparse

from competition import Competition
from utils import create_competition_folder
from utils.logger import set_competition_hash_folder


def main(config_file):
    """Main function to start the competition"""
    config = json.load(open(config_file))

    output_folder = create_competition_folder(config)
    set_competition_hash_folder(output_folder)

    competition = Competition(config)
    competition.run_competition(output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the competition")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration json")

    args = parser.parse_args()

    main(args.config_file)
