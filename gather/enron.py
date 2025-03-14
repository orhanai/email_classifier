import logging
import os
import tarfile

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_file(url: str, output_path: str, chunk_size: int = 1024) -> None:
    """
    Download a file from a given URL and save it locally.

    Args:
        url (str): URL to download the file from.
        output_path (str): Local path where the downloaded file will be saved.
        chunk_size (int): Size (in bytes) of each chunk to be written.
    """
    logging.info("Starting download from %s", url)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        logging.info("Download completed successfully and saved to %s", output_path)
    else:
        error_msg = f"Failed to download file. Status code: {response.status_code}"
        logging.error(error_msg)
        raise Exception(error_msg)


def extract_tarfile(tar_path: str, extract_to: str) -> None:
    """
    Extract a tar.gz file to a specified directory.

    Args:
        tar_path (str): Path to the tar.gz file.
        extract_to (str): Directory where the file contents will be extracted.
    """
    logging.info("Starting extraction of %s", tar_path)
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        logging.info("Created extraction directory %s", extract_to)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    logging.info(
        "Extraction completed successfully. Files are available at %s", extract_to
    )


def main():
    # Define constants for the dataset
    enron_url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
    output_tar = "enron_mail.tar.gz"
    extract_path = "enron_mail"

    try:
        download_file(enron_url, output_tar)
        extract_tarfile(output_tar, extract_path)
    except Exception as e:
        logging.exception(
            "An error occurred during the dataset download and extraction: %s", e
        )


if __name__ == "__main__":
    main()
