## Hummingbird: High Fidelity Image Generation via Multimodal Context Alignment [ICLR 2025]
[Webpage](https://roar-ai.github.io/hummingbird) | [Paper](https://openreview.net/forum?id=6kPBThI6ZJ)

### Official implementation of paper: [Hummingbird: High Fidelity Image Generation via Multimodal Context Alignment](https://openreview.net/pdf?id=6kPBThI6ZJ) 

![image/png](https://roar-ai.github.io/hummingbird/static/images/teaser_comparison_v1.png)


## Prerequisites

### Installation
1. Clone this repository and navigate to hummingbird-1 folder
```
git clone https://github.com/roar-ai/hummingbird-1
cd hummingbird-1
```


2. Create `conda` virtual environment with Python 3.9, PyTorch 2.0+ is recommended:
```
conda create -n hummingbird python=3.9
conda activate hummingbird
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

3. Install additional packages for faster training and inference
```
pip install flash-attn --no-build-isolation
```

### Download necessary models
1. Clone our Hummingbird LoRA weight of UNet denoiser
```
git clone https://huggingface.co/lmquan/hummingbird
```

2. Refer to [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) to download SDXL pre-trained model and place it in the hummingbird weight directory as `./hummingbird/stable-diffusion-xl-base-1.0`.

3. Download [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main) for `feature extractor` and `image encoder` in Hummmingbird framework
```
cp -r CLIP-ViT-bigG-14-laion2B-39B-b160k ./hummingbird/stable-diffusion-xl-base-1.0/image_encoder

mv CLIP-ViT-bigG-14-laion2B-39B-b160k ./hummingbird/stable-diffusion-xl-base-1.0/feature_extractor
```

4. Replace the file `model_index.json` of pre-trained `stable-diffusion-xl-base-1.0` with our customized version for Hummingbird framework
```
cp -r ./hummingbird/model_index.json ./hummingbird/stable-diffusion-xl-base-1.0/
```
5. Download [HPSv2 weights](https://drive.google.com/file/d/1T4e6WqsS5lcs92HdmzQYonrfDH1Ub53T/view?usp=sharing) and put it here: `hpsv2/HPS_v2_compressed.pt`. 
6. Download [PickScore model weights](https://drive.google.com/file/d/1UhR0zFXiEI-spt2QdX67FY9a0dcqa9xy/view?usp=sharing) and put it here: `pickscore/pickmodel/model.safetensors`. 

### Double check if everything is all set
```
|-- hummingbird-1/
    |-- hpsv2
        |-- HPS_v2_compressed.pt
    |-- pickscore
        |-- pickmodel
            |-- config.json
            |-- model.safetensors
    |-- hummingbird
        |-- model_index.json
        |-- lora_unet_65000
            |-- adapter_config.json
            |-- adapter_model.safetensors
        |-- stable-diffusion-xl-base-1.0
            |-- model_index.json (replaced by our customized version, see step 4 above)
            |-- feature_extractor (cloned from CLIP-ViT-bigG-14-laion2B-39B-b160k)
            |-- image_encoder (cloned from CLIP-ViT-bigG-14-laion2B-39B-b160k)
            |-- text_encoder
            |-- text_encoder_2
            |-- tokenizer
            |-- tokenizer_2
            |-- unet
            |-- vae
            |-- ...
    |-- ...
```

## Quick Start
Given a reference image, Hummingbird can generate diverse variants of it and preserve specific properties/attributes, for example:
```
python3 inference.py --reference_image ./examples/image-2.jpg --attribute "color of skateboard wheels" --output_path output.jpg
```


## Training 
You can train Hummingbird with the following script: 
```
sh run_hummingbird.sh
```

## Synthetic Data Generation
You can generate synthetic data with Hummingbird framework, for e.g. with MME Perception dataset:

```
python3 image_generation.py --generator hummingbird --dataset mme --save_image_gen ./synthetic_mme
```

## Testing 
Evaluate the fidelity of generated images w.r.t reference image using Test-Time Augmentation on MLLMs (LLaVA/InternVL2):
```
python3 test_hummingbird_mme.py --dataset mme --model llava --synthetic_dir ./synthetic_mme
```


## Acknowledgement
We base on the implementation of [TextCraftor](https://github.com/snap-research/textcraftor). We thank [BLIP-2 QFormer](https://github.com/salesforce/LAVIS), [HPSv2](https://github.com/tgxs002/HPSv2), [PickScore](https://github.com/yuvalkirstain/PickScore), [Aesthetic](https://laion.ai/blog/laion-aesthetics/) for the reward models and MLLMs [LLaVA](https://github.com/haotian-liu/LLaVA), [InternVL2](https://github.com/OpenGVLab/InternVL) functioning as context descriptors in our framework.

## Citation
If you find this work helpful, please cite our paper:
```BibTeX
@inproceedings{le2025hummingbird,
    title={Hummingbird: High Fidelity Image Generation via Multimodal Context Alignment},
    author={Minh-Quan Le and Gaurav Mittal and Tianjian Meng and A S M Iftekhar and Vishwas Suryanarayanan and Barun Patra and Dimitris Samaras and Mei Chen},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=6kPBThI6ZJ}
}
```