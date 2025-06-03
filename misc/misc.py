import os
import subprocess
import traceback

import torch
from azure.storage.blob import BlobClient, BlobServiceClient, ContentSettings
import mimetypes
from azureml.core import Run

def log_aml_val(name, value):
    try:
        run = Run.get_context(allow_offline=False)
        run.log(name, value)
    except:
        print('Error: failed to log on azure', name, value)
        pass

def mount_ckpts(ckpt_dir, key, container_name, cache_dir='/tmp/cache', storage_account='internreseus'):
    ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)
    cache_dir = os.path.join(os.getcwd(), cache_dir)

    print(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    command = ['blobfuse', ckpt_dir, '--container-name=' + container_name, '--tmp-path=' + cache_dir,
               '--file-cache-timeout-in-seconds=604800', '--cache-size-mb=200000', '-o', 'ro']
    env = {'AZURE_STORAGE_ACCOUNT': storage_account,
           'AZURE_STORAGE_ACCESS_KEY': key}
    completed = subprocess.run(command, env=env)
    try:
        assert completed.returncode == 0
    except:
        traceback.print_exc()

    os.system(f"ls {ckpt_dir}")


def mount_dataset(data_dir, key, container_name='videos', cache_dir='/tmp/cache', storage_account='internreseus'):
    data_dir = os.path.join(os.getcwd(), data_dir)
    cache_dir = os.path.join(os.getcwd(), cache_dir)

    print(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    command = ['blobfuse', data_dir, '--container-name=' + container_name, '--tmp-path=' + cache_dir,
               '--file-cache-timeout-in-seconds=604800', '--cache-size-mb=200000', '-o', 'ro']
    env = {'AZURE_STORAGE_ACCOUNT': storage_account,
           'AZURE_STORAGE_ACCESS_KEY': key}
    completed = subprocess.run(command, env=env)
    try:
        assert completed.returncode == 0
    except:
        traceback.print_exc()

    os.system(f"ls {data_dir}")


def metric_average(val, world_size):
    tensor = torch.tensor(val).cuda()
    torch.distributed.all_reduce(tensor)
    return tensor.item() / world_size

def upload_res(output_dir, container_name, conn_string, white_list_file_terms, item_prefix='',  black_list_terms=None,
               white_list_terms=None, dry_run=False, black_list_file_terms=None):

    if not os.path.exists(output_dir):
        return
    print('Upload outputs...')
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    container_client = blob_service_client.get_container_client(container_name)

    upload_count = 0
    if not output_dir.endswith('/'):
        output_dir += '/'
    for root, dirs, files in os.walk(output_dir):
        file_prefix = root.split(output_dir)[-1]

        if black_list_terms is not None and any([b in root for b in black_list_terms]):
            # print('Skipping', root)
            continue
        if white_list_terms is not None and not any([w in root for w in white_list_terms]):
            # print('Skipping', root)
            continue
        for name in files:
            item_name = os.path.join(item_prefix, file_prefix, name)
            item_location = os.path.join(root, name)
            if black_list_file_terms is not None and any([b in item_location for b in black_list_file_terms]):
                continue

            if white_list_file_terms is not None and not any([b in item_location for b in white_list_file_terms]):
                continue

            file_type, file_encoding = mimetypes.guess_type(str(item_name))
            if item_name.endswith('.log'):
                file_type = 'text/plain'
            elif item_name.endswith('.pth') or item_name.endswith('.pt'):
                file_type = 'application/octet-stream'
            upload_count += 1
            print('Uploading ', item_name, item_location, file_type, upload_count)

            if dry_run:
                continue

            with open(str(item_location), 'rb') as f:
                container_client.upload_blob(os.path.join(item_name), f,
                                             content_settings=ContentSettings(
                                                 content_type=file_type),
                                             overwrite=True)

def upload_res_azcopy(output_dir, ckpt, connection_string):
    for retry in range(3):
        try:
            print(f"Saving checkpoint {ckpt} to {output_dir}")
            command = ['/tmp/azcopy', 'cp', ckpt,
                       f'https://internreseus.blob.core.windows.net/minhquanle/checkpoints/experiments/{output_dir}/?{connection_string}',
                       '--recursive']
            print(command)
            completed = subprocess.run(command)
            assert completed.returncode == 0
            break
        except:
            traceback.print_exc()
            print('Retrying upload..')
    # os.remove(ckpt)
