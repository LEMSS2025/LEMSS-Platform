import os
import json
import hashlib
import datetime

from constants.constants import OUTPUTS_DIR, CONFIG_FILE_NAME


def create_competition_folder(config):
    """
    Create a folder for the competition based on the configuration.

    :param config: Configuration dictionary.
    :return: Path to the output folder.
    """
    # Generate unique folder name based on config hash
    folder_name = hashlib.md5(str(config).encode()).hexdigest()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_folder = os.path.join(OUTPUTS_DIR, f"{folder_name}_{current_date}")
    os.makedirs(output_folder, exist_ok=True)

    # Save config file
    with open(os.path.join(output_folder, CONFIG_FILE_NAME), 'w') as f:
        json.dump(config, f, indent=4)

    return output_folder
