import os
import argparse

import gradio as gr
import torch

import ImageBind.data as data
from ImageBind.models import imagebind_model_unnorm
from diffusers import StableUnCLIPImg2ImgPipeline
from imagebind_unclip_cache import image_generate

import llama

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="7B-beta", type=str,
    help="Name of or path to LLaMAAdapter pretrained checkpoint",
)
parser.add_argument(
    "--llama_dir", default="../huggingface/hub/models--nyanko7--LLaMA-7B/snapshots/"
, type=str,
    help="Path to LLaMA pretrained checkpoint",
)
args = parser.parse_args()


model = llama.load(args.model, args.llama_dir, knn=True)
model.eval()


pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip")
pipe = pipe.to("cuda")

imagebind = imagebind_model_unnorm.imagebind_huge(pretrained=True)
imagebind.eval()
imagebind.to("cuda")

def multimodal_generate(
        input_modality,
        output_modality,
        img_path,
        img_weight,
        text_path,
        text_weight,
        video_path,
        video_weight,
        audio_path,
        audio_weight,
        prompt,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p
):
    if len(input_modality) == 0:
        raise gr.Error('Please select at least one intput modality!')

    if len(output_modality) == 0:
        raise gr.Error('Please select at least one output modality!')

    inputs = {}
    if 'Image' in input_modality:
        if img_path is None:
            raise gr.Error('Please select an image')
        if img_weight == 0:
            raise gr.Error('Please set the weight')
        image = data.load_and_transform_vision_data([img_path], device='cuda')
        inputs['Image'] = [image, img_weight]
    if 'Text' in input_modality:
        if text_path == '':
            raise gr.Error('Please input the text')
        if text_weight == 0:
            raise gr.Error('Please set the weight')
        text = data.load_and_transform_text([text_path], device='cuda')
        inputs['Text'] = [text, text_weight]
        
    if 'Video' in input_modality:
        if video_path is None:
            raise gr.Error('Please select a video')
        if video_weight == 0:
            raise gr.Error('Please set the weight')
        video = data.load_and_transform_video_data([video_path], device='cuda')
        inputs['Video'] = [video, video_weight]
    if 'Audio' in input_modality:
        if audio_path is None:
            raise gr.Error('Please select an audio')
        if audio_weight == 0:
            raise gr.Error('Please set the weight')
        audio = data.load_and_transform_audio_data([audio_path], device='cuda')
        inputs['Audio'] = [audio, audio_weight]
    
    image_prompt = prompt
    
    prompts = [llama.format_prompt(prompt)]

    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    
    if "Text" in output_modality:
        with torch.cuda.amp.autocast():
            results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                        cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
        result = results[0].strip()
        print(result)
    
        if "Image" not in output_modality:
            return result, None
    
    if "Image" in output_modality:
        image_result = image_generate(inputs, imagebind, pipe, prompt=image_prompt, cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
        image_result[0].save("demo.png")

        if "Text" not in output_modality:
            return None, image_result[0]

    return result, image_result[0]

def create_imagebind_llm_demo():
    with gr.Blocks() as imagebind_llm_demo:
        input_modality = gr.CheckboxGroup(choices=['Image', 'Text', 'Video', 'Audio'], value='Image', interactive=True,
                                    label='Input Modalities')
        output_modality = gr.CheckboxGroup(choices=['Image', 'Text'], value='Image', interactive=True,
                                    label='Output Modalities')
        with gr.Row():
            with gr.Column() as image_input:
                img_path = gr.Image(label='Image Input', type='filepath')
                img_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
            with gr.Column() as text_input:
                text_path = gr.Textbox(label='Text Input', lines=9)
                text_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
            with gr.Column() as video_input:
                video_path = gr.Video(label='Video Input')
                video_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')
            with gr.Column() as audio_input:
                audio_path = gr.Audio(label='Audio Input', type='filepath')
                audio_weight = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, interactive=True, label='Weight')

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Textbox(lines=2, label="Question")
                with gr.Row():
                    cache_size = gr.Slider(minimum=1, maximum=100, value=10, interactive=True, label="Cache Size")
                    cache_t = gr.Slider(minimum=0.0, maximum=100, value=20, interactive=True, label="Cache Temperature")
                    cache_weight = gr.Slider(minimum=0.0, maximum=1, value=0.5, interactive=True, label="Cache Weight")
                with gr.Row():
                    max_gen_len = gr.Slider(minimum=1, maximum=512, value=128, interactive=True, label="Max Length")
                    # with gr.Accordion(label='Advanced options', open=False):
                    gen_t = gr.Slider(minimum=0, maximum=1, value=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.75, interactive=True, label="Top p")
                with gr.Row():
                    # clear_botton = gr.Button("Clear")
                    run_botton = gr.Button("Run", variant='primary')

            with gr.Column():
                text_output = gr.Textbox(lines=11, label='Text Output')

            with gr.Column():
                image_output = gr.Image(label='Image Output')

    def input_modality_select(input_modality, img, text, video, audio):
        input_modality = []
        if img is not None:
            input_modality.append('Image')
        if len(text) > 0:
            input_modality.append('Text')
        if video is not None:
            input_modality.append('Video')
        if audio is not None:
            input_modality.append('Audio')
        return input_modality

    img_path.change(input_modality_select, inputs=[input_modality, img_path, text_path, video_path, audio_path], outputs=[input_modality])
    text_path.blur(input_modality_select, inputs=[input_modality, img_path, text_path, video_path, audio_path], outputs=[input_modality])
    video_path.change(input_modality_select, inputs=[input_modality, img_path, text_path, video_path, audio_path],
                      outputs=[input_modality])
    audio_path.change(input_modality_select, inputs=[input_modality, img_path, text_path, video_path, audio_path],
                      outputs=[input_modality])

    inputs = [
        input_modality,
        output_modality,
        img_path, img_weight,
        text_path, text_weight,
        video_path, video_weight,
        audio_path, audio_weight,
        prompt,
        cache_size, cache_t, cache_weight,
        max_gen_len, gen_t, top_p
    ]
    outputs = [text_output, image_output]
    run_botton.click(fn=multimodal_generate, inputs=inputs, outputs=outputs)

    # gr.Examples(
    #     examples=examples,
    #     inputs=inputs,
    #     outputs=outputs,
    #     fn=multimodal_generate,
    #     cache_examples=False)

    return imagebind_llm_demo


description = """
# ImageBind-LLM🚀
"""

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(description)
    create_imagebind_llm_demo()

demo.queue(api_open=True, concurrency_count=1).launch(share=True)
