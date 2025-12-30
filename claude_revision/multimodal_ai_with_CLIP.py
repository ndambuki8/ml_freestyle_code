from transformers import CLIPProcessor, CLIPModel,
from PIL import Image
import torch 

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-path32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#3 Text tp image search
def search_images(query: str, image_paths: list):
    images = [Image.open(path) for path in image_paths]

    inputs = processor(
        text = [query],
        images = images,
        return_tensors="pt",
        padding=True
    )

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=0)


    # return ranke results 
    ranked = sorted(zip(image_paths, probs.squeeze().tolist()), 
                    key=lambda x: x[1], reverse=True)
    
    return ranked

# OCR + Vision for document understanding
from transformers import TcOCRProcessor, VisionEncoderDecoderModel

ocr_processor  = TcOCRProcessor.from_pretrained('microsoft/trocr-basae-handwritten')

ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-hanwritten")

def extract_text(image_path: str):
    image = Image.open(image_path).convert("RGB")
    pixel_values = ocr_processor(images=image, return_tensor="pt").pixel_values
    generated_ids = ocr_model.genenrate(pixel_values)
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text