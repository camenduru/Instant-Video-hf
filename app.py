import gradio as gr
import torch
import os
import spaces
import uuid

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image

# Constants
bases = {
    "Cartoon": "frankjoshua/toonyou_beta6",
    "Realistic": "emilianJR/epiCRealism",
    "3d": "Lykon/DreamShaper",
    "Anime": "Yntec/mistoonAnime2"
}
step_loaded = None
base_loaded = "Realistic"
motion_loaded = None

# Ensure model and scheduler are initialized in GPU-enabled function
if not torch.cuda.is_available():
    raise NotImplementedError("No GPU detected!")

device = "cuda"
dtype = torch.float16
pipe = AnimateDiffPipeline.from_pretrained(bases[base_loaded], torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

# Safety checkers
from transformers import CLIPFeatureExtractor

feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

# Function 
@spaces.GPU(duration=15,enable_queue=True)
def generate_image(prompt, base, motion, step, progress=gr.Progress()):
    global step_loaded
    global base_loaded
    global motion_loaded
    print(prompt, base, step)

    if step_loaded != step:
        repo = "ByteDance/AnimateDiff-Lightning"
        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
        pipe.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device), strict=False)
        step_loaded = step

    if base_loaded != base:
        pipe.unet.load_state_dict(torch.load(hf_hub_download(bases[base], "unet/diffusion_pytorch_model.bin"), map_location=device), strict=False)
        base_loaded = base

    if motion_loaded != motion:
        pipe.unload_lora_weights()
        if motion != "":
            pipe.load_lora_weights(motion, adapter_name="motion")
            pipe.set_adapters(["motion"], [0.7])
        motion_loaded = motion

    progress((0, step))
    def progress_callback(i, t, z):
        progress((i+1, step))

    output = pipe(prompt=prompt, guidance_scale=1.2, num_inference_steps=step, callback=progress_callback, callback_steps=1)

    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"
    export_to_video(output.frames[0], path, fps=10)
    return path


# Gradio Interface
with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        "<h1><center>Instant⚡Video</center></h1>" +
        "<p><center>Lightning-fast text-to-video generation</center></p>" +
        "<p><center><span style='color: red;'>You may change the steps from 4 to 8, if you didn't get satisfied results.</center></p>" +
        "<p><center><strong> First Video Generating takes time then Videos generate faster.</p>" +
        "<p><center>Write prompts in style as Given in Example</p>"
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(
                label='Prompt'
            )
        with gr.Row():
            select_base = gr.Dropdown(
                label='Base model',
                choices=[
                    "Cartoon", 
                    "Realistic",
                    "3d",
                    "Anime",
                ],
                value=base_loaded,
                interactive=True
            )
            select_motion = gr.Dropdown(
                label='Motion',
                choices=[
                    ("Default", ""),
                    ("Zoom in", "guoyww/animatediff-motion-lora-zoom-in"),
                    ("Zoom out", "guoyww/animatediff-motion-lora-zoom-out"),
                    ("Tilt up", "guoyww/animatediff-motion-lora-tilt-up"),
                    ("Tilt down", "guoyww/animatediff-motion-lora-tilt-down"),
                    ("Pan left", "guoyww/animatediff-motion-lora-pan-left"),
                    ("Pan right", "guoyww/animatediff-motion-lora-pan-right"),
                    ("Roll left", "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
                    ("Roll right", "guoyww/animatediff-motion-lora-rolling-clockwise"),
                ],
                value="guoyww/animatediff-motion-lora-zoom-in",
                interactive=True
            )
            select_step = gr.Dropdown(
                label='Inference steps',
                choices=[
                    ('1-Step', 1), 
                    ('2-Step', 2),
                    ('4-Step', 4),
                    ('8-Step', 8),
                ],
                value=4,
                interactive=True
            )
            submit = gr.Button(
                scale=1,
                variant='primary'
            )
    video = gr.Video(
        label='AnimateDiff-Lightning',
        autoplay=True,
        height=512,
        width=512,
        elem_id="video_output"
    )

    prompt.submit(
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=video,
    )
    submit.click(
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=video,
    )

    gr.Examples(
        examples=[
        ["Focus: Eiffel Tower (Animate: Clouds moving)"], #Atmosphere Movement Example
        ["Focus: Trees In forest (Animate: Lion running)"], #Object Movement Example
        ["Focus: Astronaut in Space"], #Normal
        ["Focus: Group of Birds in sky (Animate:  Birds Moving) (Shot From distance)"], #Camera distance
        ["Focus:  Statue of liberty (Shot from Drone) (Animate: Drone coming toward statue)"], #Camera Movement
        ["Focus: Panda in Forest (Animate: Drinking Tea)"], #Doing Something
        ["Focus: Kids Playing (Season: Winter)"], #Atmosphere or Season
        {"Focus: Cars in Street (Season: Rain, Daytime) (Shot from Distance) (Movement: Cars running)"} #Mixture
    ], 
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=video,
        cache_examples=False,
)

demo.queue().launch()