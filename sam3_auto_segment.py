import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from dotenv import load_dotenv
from huggingface_hub import login

def main():
    # Load environment variables (e.g., HF_TOKEN)
    load_dotenv()
    
    # Authenticate to Hugging Face if a token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # 1. Prompt the user for the image location
    print("=== SAM 3 Automatic Mask Detection ===")
    image_path = input("Enter the path to the picture (e.g., data/image/test.png): ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at '{image_path}'")
        sys.exit(1)
        
    # 2. Ensure output directory exists (data/output)
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading image...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # 3. Initialize Device and Model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing SAM 3 model on {device}...")
    
    model_id = "facebook/sam3"
    try:
        processor = Sam3Processor.from_pretrained(model_id)
        
        # Determine dtype to save memory
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        model = Sam3Model.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
    except Exception as e:
        print(f"Failed to load the model: {e}")
        sys.exit(1)
        
    # 4. Generate Prompt and Run Inference
    # We use a general prompt to detect objects. Adjust as needed if the SAM 3 model handles "all" differently.
    prompt = "apple"
    print(f"Running inference with text prompt: '{prompt}'...")
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Convert floating point inputs to match model dtype
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(dtype)
            
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 5. Post-process to get instance masks
    # [height, width] of original image to rescale masks properly
    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]] 
    )[0]
    
    masks = results.get("masks", [])
    
    num_masks = len(masks)
    print(f"Found {num_masks} object masks.")
    
    # 6. Save outputs to data/output
    if num_masks == 0:
        print(f"No masks were detected. You might want to try a different text prompt.")
    else:
        import cv2
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Convert original PIL image to numpy array (RGB)
        image_np = np.array(image.copy())
        
        # Set a fixed random seed so colors are consistent across runs
        np.random.seed(42)
        
        for i, mask_tensor in enumerate(masks):
            # Create binary mask (0 and 1) for contour detection
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
            
            # Find contours of the mask
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Generate a random distinct color for this mask (RGB)
            color = tuple(int(x) for x in np.random.randint(0, 255, 3))
            
            # Draw the contour on the image with thickness 3
            cv2.drawContours(image_np, contours, -1, color, 3) 
            
        # Save as a single annotated image
        output_filepath = os.path.join(output_dir, f"{base_name}_with_contours.png")
        Image.fromarray(image_np).save(output_filepath)
            
        print(f"Success! Saved original image with {num_masks} colored contours to '{output_filepath}'.")

if __name__ == "__main__":
    main()
