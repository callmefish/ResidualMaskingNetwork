import os


checkpoint_url = "https://github.com/phamquiluan/ResidualMaskingNetwork/releases/download/v0.0.1/Z_resmasking_dropout1_rot30_2019Nov30_13.32"
local_checkpoint_path = "./saved/checkpoints/"
if not os.path.exists(local_checkpoint_path):
    os.makedirs(local_checkpoint_path)
local_checkpoint_path = local_checkpoint_path + "Z_resmasking_dropout1_rot30_2019Nov30_13.32"

prototxt_url = "https://github.com/phamquiluan/ResidualMaskingNetwork/releases/download/v0.0.1/deploy.prototxt.txt"
local_prototxt_path = "deploy.prototxt.txt"

ssd_checkpoint_url = "https://github.com/phamquiluan/ResidualMaskingNetwork/releases/download/v0.0.1/res10_300x300_ssd_iter_140000.caffemodel"
local_ssd_checkpoint_path = "res10_300x300_ssd_iter_140000.caffemodel"


def download_checkpoint(remote_url, local_path):
    from tqdm import tqdm
    import requests

    response = requests.get(remote_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        desc=f"Downloading {local_path}..",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(local_path, "wb") as ref:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            ref.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


for remote_path, local_path in [
    (checkpoint_url, local_checkpoint_path),
    (prototxt_url, local_prototxt_path),
    (ssd_checkpoint_url, local_ssd_checkpoint_path),
]:
    if not os.path.exists(local_path):
        print(f"{local_path} does not exists!")
        download_checkpoint(remote_url=remote_path, local_path=local_path)
