import os
# 设置 Hugging Face 镜像和缓存路径
os.environ['HF_HOME'] = '/root/autodl-fs/huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import torch
import numpy as np
import safetensors.torch as sf
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
from fastapi.responses import Response

app = FastAPI()


# 加载模型
sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")
unet = UNet2DConditionModel.from_pretrained(sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

# 设置注意力处理器
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

class PromptRequest(BaseModel):
    prompt: str
    model_path: str

def generate_image(prompt: str, model_path: str):
    
    # 添加 model_path 到 sys.path，让 Python 可以找到 lib_layerdiffuse
    sys.path.append(model_path)
    from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
    # from lib_layerdiffuse.utils import download_model
    import memory_management
    from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
    models_dir = os.path.join(model_path, "models")

    # Download Mode

    path_ld_diffusers_sdxl_attn = download_model(
        url='https://hf-mirror.com/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors',
        local_path=os.path.join(models_dir, "ld_diffusers_sdxl_attn.safetensors")
    )
    
    path_ld_diffusers_sdxl_vae_transparent_encoder = download_model(
        url='https://hf-mirror.com/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        local_path=os.path.join(models_dir, "ld_diffusers_sdxl_vae_transparent_encoder.safetensors")
    )
    
    path_ld_diffusers_sdxl_vae_transparent_decoder = download_model(
        url='https://hf-mirror.com/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
        local_path=os.path.join(models_dir, "ld_diffusers_sdxl_vae_transparent_decoder.safetensors")
    )

    # 合并 UNet 权重
    sd_offset = sf.load_file(os.path.join(models_dir, "ld_diffusers_sdxl_attn.safetensors"))
    sd_origin = unet.state_dict()
    for k in sd_origin.keys():
        if k in sd_offset:
            sd_origin[k] += sd_offset[k]
    unet.load_state_dict(sd_origin, strict=True)
    
    del sd_offset, sd_origin
    sys.path.append(model_path)
    
    # 加载 TransparentVAE 编码器和解码器
    transparent_encoder = TransparentVAEEncoder(os.path.join(models_dir, "ld_diffusers_sdxl_vae_transparent_encoder.safetensors"))
    transparent_decoder = TransparentVAEDecoder(os.path.join(models_dir, "ld_diffusers_sdxl_vae_transparent_decoder.safetensors"))
    
    # 初始化 Pipeline
    pipeline = KDiffusionStableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=None,
    )
    
    guidance_scale = 7.0
    rng = torch.Generator(device=memory_management.gpu).manual_seed(12345)
    
    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
    positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(prompt)
    negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens("face asymmetry, eyes asymmetry, deformed eyes, open mouth")
    
    memory_management.load_models_to_gpu([unet])
    initial_latent = torch.zeros(size=(1, 4, 144, 112), dtype=unet.dtype, device=unet.device)
    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=25,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=guidance_scale,
    ).images
    
    memory_management.load_models_to_gpu([vae, transparent_decoder, transparent_encoder])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    result_list, vis_list = transparent_decoder(vae, latents)
    
    return result_list, vis_list

@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        result_list, vis_list = generate_image(request.prompt, request.model_path)
        img_io = BytesIO()
        for i, image in enumerate(result_list):
            Image.fromarray(image).save(img_io, format='PNG')

    
        img_io.seek(0)
        return Response(content=img_io.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)