import logging
import typing as tp
from pathlib import Path
from py7zr import SevenZipFile
import multivolumefile
import requests
from tqdm import tqdm

_NAMED_URLS: tp.Dict[str, str] = {
    "perfcap_benchmark_server": "https://github.com/VCL3D/PerformanceCapture/releases/"
    "download/1.0/performance_capture_benchmark_server.7z",
    "perfcap_benchmark_dataset_part#1": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.001",
    "perfcap_benchmark_dataset_part#2": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.002",
    "perfcap_benchmark_dataset_part#3": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.003",
    "perfcap_benchmark_dataset_part#4": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.004",
    "perfcap_benchmark_dataset_part#5": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.005",
    "perfcap_benchmark_dataset_part#6": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.006",
    "perfcap_benchmark_dataset_part#7": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.007",
    "perfcap_benchmark_dataset_part#8": "https://github.com/VCL3D/PerformanceCapture/releases/download/dataset_1.0/data.7z.008",
}


def get_resources_folder() -> Path:
    """
    Returns the path to resources folder
    """
    return Path(__file__).parent / "resources"


def download_resource(name: str) -> Path:
    """
    Function responsible to download a resource from a url
    """
    logger = logging.getLogger("download_resource")

    if name not in _NAMED_URLS:
        raise ValueError(
            f'Resource "{name}" is not available. Please choose among:\n{list(_NAMED_URLS.keys())}'
        )

    url = _NAMED_URLS[name]
    url_name = Path(url).name

    filepath = get_resources_folder() / "downloads" / url_name
    filepath.parent.mkdir(parents=True, exist_ok=True)

    request_header = None
    current_file_size = 0
    if filepath.exists():
        current_file_size = filepath.stat().st_size
        request_header = {"Range": f"bytes={current_file_size}-"}

    logger.info("Downloading external resource %s from url: %s", str(name), str(url))

    # initial request to obtain file size
    try:
        response = requests.get(url, verify=True, stream=True)
    except requests.exceptions.SSLError:
        logger.warning("SSL verification failed for %s, requesting header without verification.", url)
        response = requests.get(url, verify=True, stream=True)
    response.close()

    resource_size = int(response.headers.get("content-length"))
    logger.info("Resource file size: %s MB", round(resource_size / (1024 * 1024), 1))

    if resource_size != current_file_size:

        if current_file_size > resource_size:
            # invalid file. re-download
            request_header = {}
            current_file_size = 0
        try:
            response = requests.get(url, headers=request_header, verify=True, stream=True)
        except requests.exceptions.SSLError:
            logger.warning("SSL verification failed for %s, downloading without verification.", url)
            response = requests.get(url, verify=False, stream=True)
        try:
            chunk_size = 1024 * 1024  # 1MB chunk size
            prog_bar = tqdm(
                total=resource_size, unit="B", unit_scale=True, unit_divisor=1024, initial=current_file_size
            )
            with filepath.open("wb" if current_file_size == 0 else "ab") as f:
                for data in response.iter_content(chunk_size=chunk_size):
                    f.write(data)
                    prog_bar.update(len(data))
            prog_bar.close()
            logger.info("Download complete")
            response.close()
        # pylint: disable=broad-except
        except Exception as ex:
            logger.warning("Failed to download resource %s. Reason: %s", name, ex)
    else:
        logger.info("Resource %s already found in download cache. Skipping", name)

    return filepath


def decompress_resource(resource_file: Path):
    """
    Decompress 7zip archive.
    """
    logger = logging.getLogger("decompress_resource")
    is_multipart = resource_file.name.endswith(".001")

    out_dir = get_resources_folder()
    logger.info("Decompressing resource file: %s", resource_file)
    try:
        if is_multipart:
            logger.info(
                "This is a multi-part file. Decompression may take a few minutes. Please be patient ..."
            )
            resource_file_ = resource_file.parent / resource_file.name.replace(".001", "")
            with multivolumefile.open(resource_file_, "rb") as zf:
                with SevenZipFile(zf, "r") as archive:
                    archive.extractall(out_dir)
        else:
            with SevenZipFile(resource_file, "r") as zf:
                zf.extractall(out_dir)
        logger.info("File decompressed successfully.")
    # pylint: disable=broad-except
    except Exception as ex:
        logger.info("File decompression failed. Reason: %s", ex)


def check_resources_to_download() -> tp.List[str]:
    """
    Checks if download of resources is required.
    It checks for existence of performanca_capture.exe and data folder in the resources directory.
    """
    resources_to_download = []
    # check if benchmark_server needs to be downloaded.
    resource_folder = get_resources_folder()
    server_exe_path = resource_folder / "performance_capture.exe"
    if not server_exe_path.exists():
        resources_to_download.append("benchmark_server")

    data_path = resource_folder / "data"
    if not data_path.exists():
        resources_to_download.append("dataset")

    return resources_to_download


def resource_needs_download(resource_name: str, resources_to_download: tp.List[str]) -> bool:
    for res_to_dl in resources_to_download:
        if res_to_dl in resource_name:
            return True
    return False


def prepare_resources():
    """
    Download and extract resources
    """
    logger = logging.getLogger("prepare_resources")
    logger.info("Checking resources ....")
    res_to_dl = check_resources_to_download()

    files_to_extract = []
    for resource_name, resource_url in _NAMED_URLS.items():
        if resource_needs_download(resource_name, res_to_dl):
            fpath = download_resource(resource_name)
            if resource_url.endswith(".7z") or resource_url.endswith(".001"):
                files_to_extract.append(fpath)

    for f in files_to_extract:
        decompress_resource(f)

    logger.info(
        "Resources OK. In case you experience an execution error consider clearing the resource folder to start from scratch: %s",
        str(get_resources_folder()),
    )


if __name__ == "__main__":
    prepare_resources()
