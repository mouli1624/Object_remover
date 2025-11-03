import os
import base64
import requests
from pathlib import Path
from typing import Optional, Tuple
import fal_client
from PIL import Image
import io

class InpaintingService:
    """
    Service for object removal using Flux Kontext via FAL AI
    """
    
    # OPTION 1: Hardcode your API key here (not recommended for production)
    HARDCODED_API_KEY = ""  # Set your key here like: "99144ad5-8f12-42c7-ba2e-bb988b8fdbab:3f1aaac07bceb1b6d8d6020029390612"
    
    def __init__(self):
        # Load FAL_KEY: First try hardcoded, then environment variable
        self.fal_key = self.HARDCODED_API_KEY or os.environ.get("FAL_KEY")
        
        if not self.fal_key:
            print("=" * 60)
            print("WARNING: FAL_KEY not found!")
            print("Option 1 - Hardcode in inpainting_service.py:")
            print("  Set HARDCODED_API_KEY = 'your-key-here'")
            print("\nOption 2 - PowerShell:")
            print("  $env:FAL_KEY='your-key-here'")
            print("  Then restart the server")
            print("\nOption 3 - Command Prompt (cmd):")
            print("  set FAL_KEY=your-key-here")
            print("  Then restart the server")
            print("\nGet your key from: https://fal.ai/dashboard/keys")
            print("=" * 60)
        else:
            print(f"✅ FAL_KEY loaded successfully (ends with: ...{self.fal_key[-8:]})")
    
    def remove_object(
        self,
        image_path: str,
        mask_path: str,
        object_name: str = "object",
        output_path: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Remove object from image using Flux Kontext.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            object_name: Name of the object to remove (for prompt)
            output_path: Optional path to save the result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        if not self.fal_key:
            raise Exception("FAL_KEY not set. Please set your FAL API key.")
        
        try:
            print("=" * 60)
            print(f"Removing '{object_name}' from image using Flux Kontext...")
            print("=" * 60)
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
            
            # Read and encode mask
            with open(mask_path, 'rb') as f:
                mask_data = f.read()
            mask_base64 = base64.b64encode(mask_data).decode('utf-8')
            mask_url = f"data:image/png;base64,{mask_base64}"
            
            # Create prompt
            prompt = f"remove {object_name} from the image"
            
            print(f"Prompt: '{prompt}'")
            print("Sending request to FAL AI...")
            
            # Call FAL API
            handler = fal_client.submit(
                "fal-ai/flux-pro/kontext",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    # "mask_url": mask_url,
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5,
                    "output_format": "png",
                    "seed": None  # Random seed for variation
                }
            )
            
            # Wait for result
            result = handler.get()
            
            print("✅ Inpainting completed!")
            
            # Download the result image
            if result and 'images' in result and len(result['images']) > 0:
                result_url = result['images'][0]['url']
                
                # Download image
                response = requests.get(result_url)
                if response.status_code == 200:
                    # Save to output path
                    if output_path is None:
                        # Generate output path
                        image_path_obj = Path(image_path)
                        output_path = str(image_path_obj.parent / f"result_{image_path_obj.name}")
                    
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"Result saved to: {output_path}")
                    
                    result_info = {
                        "success": True,
                        "prompt": prompt,
                        "object_removed": object_name,
                        "output_path": output_path,
                        "seed": result.get('seed'),
                        "inference_time": result.get('timings', {}).get('inference', 0)
                    }
                    
                    return output_path, result_info
                else:
                    raise Exception(f"Failed to download result image: {response.status_code}")
            else:
                raise Exception("No images in result")
                
        except Exception as e:
            print("=" * 60)
            print(f"ERROR during inpainting: {e}")
            print("=" * 60)
            raise
    
    def remove_multiple_objects(
        self,
        image_path: str,
        mask_path: str,
        object_names: list,
        output_path: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Remove multiple objects from image.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the combined mask image
            object_names: List of object names to remove
            output_path: Optional path to save the result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        # Create combined prompt
        if len(object_names) == 1:
            object_desc = object_names[0]
        elif len(object_names) == 2:
            object_desc = f"{object_names[0]} and {object_names[1]}"
        else:
            object_desc = ", ".join(object_names[:-1]) + f", and {object_names[-1]}"
        
        return self.remove_object(image_path, mask_path, object_desc, output_path)


# Singleton instance
_inpainting_service = None

def get_inpainting_service() -> InpaintingService:
    """
    Get or create the inpainting service singleton.
    """
    global _inpainting_service
    if _inpainting_service is None:
        _inpainting_service = InpaintingService()
    return _inpainting_service
