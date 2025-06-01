# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install comfy-cli
RUN uv pip install comfy-cli --system

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.30 --cuda-version 12.6 --nvidia

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client --system

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE=flux1-dev-fp8

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/loras

# Common wget options for robustness (removed -q for debugging)
ENV WGET_OPTS="--tries=3 --timeout=30 --read-timeout=600 --retry-connrefused --waitretry=5 -c"

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      echo "Downloading flux1-dev-fp8 base checkpoint (Comfy-Org)..." && \
      wget $WGET_OPTS -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors && \
      echo "Downloading flux_text_encoders clip_l.safetensors (comfyanonymous)..." && \
      wget $WGET_OPTS -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      echo "Downloading flux_text_encoders t5xxl_fp8_e4m3fn.safetensors (comfyanonymous)..." && \
      wget $WGET_OPTS -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      echo "Downloading VAE ae.safetensors (lovis93/testllm)..." && \
      wget $WGET_OPTS -O models/vae/ae.safetensors https://huggingface.co/lovis93/testllm/resolve/ed9cf1af7465cebca4649157f118e331cf2a084f/ae.safetensors && \
      echo "Downloading Checkpoint Gemini_ILMixV5.safetensors (CuteBlueEyed/GeminiX)..." && \
      wget $WGET_OPTS -O models/checkpoints/Gemini_ILMixV5.safetensors https://huggingface.co/CuteBlueEyed/GeminiX/resolve/main/Gemini_ILMixV5.safetensors && \
      echo "Downloading Checkpoint waiNSFWIllustrious_v120.safetensors (nnnn1111/models)..." && \
      wget $WGET_OPTS -O models/checkpoints/waiNSFWIllustrious_v120.safetensors https://huggingface.co/nnnn1111/models/resolve/main/waiNSFWIllustrious_v120.safetensors && \
      echo "Downloading Checkpoint Gemini_ILMixWebtoonV2.safetensors (CuteBlueEyed/GeminiX)..." && \
      wget $WGET_OPTS -O models/checkpoints/Gemini_ILMixWebtoonV2.safetensors https://huggingface.co/CuteBlueEyed/GeminiX/resolve/main/Gemini_ILMixWebtoonV2.safetensors && \
      echo "Downloading VAE sdxl_vae.safetensors (stabilityai/sdxl-vae)..." && \
      wget $WGET_OPTS -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      echo "Downloading LoRA Detail_Tweaker_Illustrious_BSY_V3.safetensors (CuteBlueEyed/Gemini_ILMix)..." && \
      wget $WGET_OPTS -O models/loras/Detail_Tweaker_Illustrious_BSY_V3.safetensors https://huggingface.co/CuteBlueEyed/Gemini_ILMix/resolve/main/Detail_Tweaker_Illustrious_BSY_V3.safetensors && \
      echo "Downloading LoRA aidmaRealisticSkin-IL-v0.1.safetensors (CuteBlueEyed/Gemini_ILMix)..." && \
      wget $WGET_OPTS -O models/loras/aidmaRealisticSkin-IL-v0.1.safetensors https://huggingface.co/CuteBlueEyed/Gemini_ILMix/resolve/main/aidmaRealisticSkin-IL-v0.1.safetensors && \
      echo "Downloading LoRA aidmahyperrealism_IL.safetensors (CuteBlueEyed/Gemini_ILMix)..." && \
      wget $WGET_OPTS -O models/loras/aidmahyperrealism_IL.safetensors https://huggingface.co/CuteBlueEyed/Gemini_ILMix/resolve/main/aidmahyperrealism_IL.safetensors && \
      echo "Downloading LoRA Catherine_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Catherine_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Catherine_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Hanmy_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Hanmy_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Hanmy_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Kaho_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Kaho_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Kaho_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Kitakana_GMIL_TAV102.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Kitakana_GMIL_TAV102.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Kitakana_GMIL_TAV102.safetensors && \
      echo "Downloading LoRA Kudan_GMIL_TAV102.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Kudan_GMIL_TAV102.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Kudan_GMIL_TAV102.safetensors && \
      echo "Downloading LoRA Li-Duoyuan_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Li-Duoyuan_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Li-Duoyuan_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Namiko_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Namiko_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Namiko_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Nanamin_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Nanamin_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Nanamin_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Nicole_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Nicole_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Nicole_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Nova_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Nova_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Nova_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Numi_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Numi_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Numi_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Ririka_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Ririka_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Ririka_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Vicky_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Vicky_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Vicky_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Yetta_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Yetta_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Yetta_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Yudan_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Yudan_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Yudan_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA Yuina_GMIL_TAV103.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Yuina_GMIL_TAV103.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Yuina_GMIL_TAV103.safetensors && \
      echo "Downloading LoRA NanJade_GMIL_TAV101.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/NanJade_GMIL_TAV101.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/NanJade_GMIL_TAV101.safetensors && \
      echo "Downloading LoRA Mirei_GMIL_TAV102.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Mirei_GMIL_TAV102.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Mirei_GMIL_TAV102.safetensors && \
      echo "Downloading LoRA Henna_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Henna_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Henna_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Jittaya_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Jittaya_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Jittaya_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Emily_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Emily_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Emily_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Nora_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Nora_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Nora_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Lomo_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Lomo_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Lomo_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Tessa_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Tessa_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Tessa_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Eun-Bi_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Eun-Bi_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Eun-Bi_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Jennifer_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Jennifer_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Jennifer_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA JenniferElf_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/JenniferElf_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/JenniferElf_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA KURA_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/KURA_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/KURA_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Mio_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Mio_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Mio_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Tenchan_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Tenchan_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Tenchan_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Anna_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Anna_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Anna_GMIL_TAV1.safetensors && \
      echo "Downloading LoRA Fenny_GMIL_TAV1.safetensors (CuteBlueEyed/LoRAForGeminiX_IL)..." && \
      wget $WGET_OPTS -O models/loras/Fenny_GMIL_TAV1.safetensors https://huggingface.co/CuteBlueEyed/LoRAForGeminiX_IL/resolve/main/Fenny_GMIL_TAV1.safetensors && \
      echo "Finished downloading all flux1-dev-fp8 models and LoRAs."; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
