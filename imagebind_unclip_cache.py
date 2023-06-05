import ImageBind.data as data
import llama
from diffusers import StableUnCLIPImg2ImgPipeline
import ImageBind.data as data
from ImageBind.models import imagebind_model_unnorm
from llama.utils import sample_top_p, _download
import torch
import numpy as np
from PIL import Image

def image_generate(inputs, imagebind, pipe, prompt, cache_size, cache_t, cache_weight):

    embeddings = []
    embeddings_weights = []

    for input_type, (input, input_weight) in inputs.items():
        if input_type in ['Image', 'Video']:
            type = 'vision'
        else:
            type = input_type.lower()

        embeddings.append(imagebind({type : input})[type])
        embeddings_weights.append(input_weight)
    embeddings_weights = [x/(sum(embeddings_weights)+1e-6) for x in embeddings_weights]
    embedding = sum([embedding*embedding_weight for embedding, embedding_weight in zip(embeddings, embeddings_weights)])
    
    import faiss
    index = faiss.read_index(_download("https://huggingface.co/csuhan/knn/resolve/main/knn.index", "ckpts"))
    
    embedding_norm_scale = embedding.norm(dim=-1, keepdim=True)
    embedding = embedding / embedding_norm_scale
    embedding_ori = embedding

    sims, indices = index.search(embedding.detach().cpu(), int(cache_size))
    B = sims.shape[0]
    prototypes = [index.reconstruct(x) for x in indices.reshape(-1, ).tolist()]
    prototypes = np.vstack(prototypes).reshape(B, int(cache_size), -1)  # [N, top_k, 1024]
    sims = torch.tensor(sims, device='cuda')
    prototypes = torch.tensor(prototypes, device='cuda')

    sims = (sims * cache_t).softmax(dim=-1)
    embedding = sims @ prototypes
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    embedding = (1-cache_weight) * embedding_ori + cache_weight * embedding
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = torch.squeeze(embedding_norm_scale*embedding,0)

    images = pipe(prompt=prompt, image_embeds=embedding).images

    return images

if __name__ == "__main__":
    output = {"image"}

    inputs = {}
    image = data.load_and_transform_vision_data(["messi.png"], device='cuda')
    inputs['Image'] = [image, 0]
    audio = data.load_and_transform_audio_data(['ESC-50-master/audio/1-137-A-32.wav'], device='cuda')
    inputs['Audio'] = [audio, 1]
    text = data.load_and_transform_text(['A dog fight with a cat.'], device='cuda')
    inputs['Text'] = [text, 0]
    video = data.load_and_transform_video_data(['video_fish.mp4'], device='cuda')
    inputs['Video'] = [video, 0]

    if "text" in output:
        llama_dir = "../huggingface/hub/models--nyanko7--LLaMA-7B/snapshots/"

        model = llama.load("7B-beta", llama_dir, knn=True)
        model.eval()

        results = model.generate(
            inputs,
            #[llama.format_prompt("Guess the girl's mood based on the background music and explain the reason?")],
            [llama.format_prompt("Describe the image")],
            max_gen_len=256
        )
        result = results[0].strip()
        print(result)

    if "image" in output:
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip")
        #pipe = pipe.to("cuda")
        print ("loading")

        imagebind = imagebind_model_unnorm.imagebind_huge(pretrained=True)
        imagebind.eval()
        #imagebind.to("cuda")
        images = image_generate(inputs, imagebind, pipe, prompt='', cache_size=10, cache_t=20, cache_weight=0.5,)
        images[0].save("demo.png")
