import logging
import random
import numpy as np
import torch
from PIL import Image

from data.config import image_captioning
from functional.vocab import vocabulary, EOS_IDX
from .beheaded_inception3 import beheaded_inception_v3

class FrameCapturingPipeline():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_captions = 1
        # self.fix_seed()
        self.vocabulary = vocabulary
        self.model_type = image_captioning['general']['MODEL_TYPE']
        if self.model_type == '1_layer_simple_LSTM':
            from .lstm import LSTM_model_1_layer
            self.model = LSTM_model_1_layer
            self.state_dict = image_captioning['paths']['WEIGHTS_PATH'] + self.model_type + '.pth'
        elif self.model_type == '2_layer_simple_LSTM':
            from .lstm import LSTM_model_2_layer
            self.model = LSTM_model_2_layer
            self.state_dict = image_captioning['paths']['WEIGHTS_PATH'] + self.model_type + '.pth'
        elif self.model_type == '2_layers_attn':
            from .attn import attn_2_layers
            self.model = attn_2_layers
            self.state_dict = image_captioning['paths']['WEIGHTS_PATH'] + self.model_type + '.pth'

        logging.info(f"Model {self.model_type} have built")
        self.model.load_state_dict(torch.load(__file__[:-len('frame_capturing_pipeline.py')] + self.state_dict , map_location=torch.device(self.device)))
        self.inception = beheaded_inception_v3(device=self.device).train(False)

    def fix_seed(self):
        seed = image_captioning['general']['SEED']
        random.seed(seed)
        np.random.seed(seed)

    def return_words(self, sentences, vocab, is_list=False):
        words = []
        for idx in sentences:
            idx = int(idx.item())
            if idx != EOS_IDX:
                words.append(vocab.get_word(idx))
            else:
                break
        if is_list:
            return words
        else:
            return ' '.join(words)

    def generate_caption(self, image, top_k=1):
        image = Image.open(image).convert('RGB')
        image = np.array(image.resize((299, 299))).astype('float32') / 255;
        assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
               and np.min(image) >= 0 and image.shape[-1] == 3

        with torch.no_grad():
            image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
            vectors_8x8, vectors_neck, logits = self.inception(image[None])  # [1, 2048, 8, 8] | [1, 2048] | [1, 1000]

            outputs, probs = self.model.test(vectors_neck.to(self.device), top_k)

            # decode one oicture from output outputs and "probability" of sentence
            return (self.return_words(outputs[0], self.vocabulary), round(probs.item(), 4))

    def generate_captions(self, img, num_captions, top_k):
        captions = []
        img = img[-1] # TODO: scalability for several photos
        caption = self.generate_caption(img)
        captions.append(caption)
        if num_captions > 1:
            for _ in range(1, num_captions):
                caption = self.generate_caption(img, top_k=top_k)
                captions.append(caption)
        return captions

    def __call__(self, img, num_captions: int):
        return self.generate_captions(img, num_captions, image_captioning['caption_generator']['TOP_K'])