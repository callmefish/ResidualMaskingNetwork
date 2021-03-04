import os


checkpoint_url = "https://github.com/callmefish/ResidualMaskingNetwork/releases/download/v1.0.2/resmasking_dropout1_00"
local_checkpoint_path = "./saved/checkpoints/"
if not os.path.exists(local_checkpoint_path):
    os.makedirs(local_checkpoint_path)
local_checkpoint_path = local_checkpoint_path + "resmasking_dropout1_00"


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
    (checkpoint_url, local_checkpoint_path)
]:
    if not os.path.exists(local_path):
        print(f"{local_path} does not exists!")
        download_checkpoint(remote_url=remote_path, local_path=local_path)
