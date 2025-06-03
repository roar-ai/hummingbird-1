# @title Modifying the StableDiffusionPipeline \_\_call\_\_ method

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch
from torch import autocast

# Taken from src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# From v0.16.1 source code: https://github.com/huggingface/diffusers/releases


@torch.no_grad()
def __call__(
    self,
    prompt: Union[str, List[str]],
    percent_noise: int,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
):

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            #######################################################################################
            # BOOMERANG CODE:
            # Skip any steps in [0, 1000] that are before (i.e., greater than) 1000 * percent noise
            if t - 1 > 1000 * percent_noise:
                continue
            # print(t)
            #######################################################################################

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == "latent":
        image = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
        has_nsfw_concept = [False] * len(image)
    else:
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


from torchvision import transforms as T


def encode_latents(pipe, img):

    # Convert image to float and preprocess it, same as in
    # huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
    transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float), T.Normalize([0.5], [0.5])])

    with torch.no_grad():
        tensor = transform(img).half().to(pipe.device)
        tensor = torch.unsqueeze(tensor, 0)

        # Project image into the latent space
        clean_z = pipe.vae.encode(
            tensor
        ).latent_dist.mode()  # From huggingface/diffusers/blob/main/src/diffusers/models/vae.py
        clean_z = 0.18215 * clean_z

    return clean_z


def boomerang_forward(pipe, percent_noise, latents):
    """
    Add noise to the latents via the pipe noise scheduler, according to percent_noise.
    """

    assert percent_noise <= 0.999
    assert percent_noise >= 0.02

    # Add noise to the latent variable
    # (this is the forward diffusion process)
    noise = torch.randn(latents.shape).to(pipe.device)
    timestep = torch.Tensor([int(pipe.scheduler.config.num_train_timesteps * percent_noise)]).to(pipe.device).long()
    z = pipe.scheduler.add_noise(latents, noise, timestep).half()

    return z


def boomerang_reverse(pipe, prompt, percent_noise, latents, num_inference_steps=100, output_type="pil"):
    """
    Denoise the noisy latents according to percent_noise.
    """

    assert percent_noise <= 0.999
    assert percent_noise >= 0.02

    # Run the reverse boomerang process
    with autocast("cuda"):
        return pipe(
            prompt=prompt,
            percent_noise=percent_noise,
            latents=latents,
            num_inference_steps=num_inference_steps,
            output_type=output_type,
        )
