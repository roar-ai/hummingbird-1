from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/mnt/minhquan/stable-diffusion-xl-base-1.0",
    repo_id="lmquan/hummingbird",
    repo_type="model",
)
