import os
import base64
import replicate
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io

class InpaintingService:
    """
    Service for object removal using LaMa via Replicate
    """
    
    def __init__(self):
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            print("=" * 60)
            print("WARNING: REPLICATE_API_TOKEN not found in environment variables!")
            print("Set it with: export REPLICATE_API_TOKEN='your-token-here'")
            print("Get your token from: https://replicate.com/account/api-tokens")
            print("=" * 60)
        else:
            print("=" * 60)
            print("✅ Replicate API token found!")
            print("Using LaMa model for inpainting")
            print("=" * 60)
    
    def dilate_mask(self, mask_path: str, dilation_pixels: int = 30) -> str:
        """
        Dilate (expand) the mask by the specified number of pixels.
        
        Args:
            mask_path: Path to the mask image
            dilation_pixels: Number of pixels to expand the mask (default: 30)
            
        Returns:
            Path to the dilated mask
        """
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a circular kernel for dilation
        kernel_size = dilation_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate the mask
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Save dilated mask
        mask_path_obj = Path(mask_path)
        dilated_mask_path = str(mask_path_obj.parent / f"dilated_{mask_path_obj.name}")
        cv2.imwrite(dilated_mask_path, dilated_mask)
        
        print(f"Mask dilated by {dilation_pixels} pixels")
        
        return dilated_mask_path
    
    def remove_object(
        self,
        image_path: str,
        mask_path: str,
        object_name: str = "object",
        output_path: Optional[str] = None,
        **kwargs  # Ignore extra parameters for compatibility
    ) -> Tuple[str, dict]:
        """
        Remove object from image using LaMa via Replicate.
        
        Args:
            image_path: Path to the input image
            mask_path: Path to the mask image
            object_name: Name of the object to remove (for logging)
            output_path: Optional path to save the result
        
        Returns:
            Tuple of (output_image_path, result_info)
        """
        if not self.api_token:
            raise Exception("REPLICATE_API_TOKEN not set. Please set your Replicate API token.")
        
        try:
            import time
            start_time = time.time()
            
            print("=" * 60)
            print(f"Removing '{object_name}' from image using LaMa...")
            print("=" * 60)
            
            # Dilate the mask to expand it by 30 pixels
            dilated_mask_path = self.dilate_mask(mask_path, dilation_pixels=30)
            
            # Load image and dilated mask
            with open(image_path, 'rb') as f:
                image_data = f.read()
            with open(dilated_mask_path, 'rb') as f:
                mask_data = f.read()
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            mask_base64 = base64.b64encode(mask_data).decode('utf-8')
            
            # Create data URIs
            image_uri = f"data:image/png;base64,{image_base64}"
            mask_uri = f"data:image/png;base64,{mask_base64}"
            
            print("Sending request to Replicate LaMa model...")
            
            # Run LaMa model on Replicate
            output = replicate.run(
                "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72",
                input={
                    "image": image_uri,
                    "mask": mask_uri
                }
            )
            
            # Download the result
            result_data = output.read()
            
            # Save output
            if output_path is None:
                image_path_obj = Path(image_path)
                output_path = str(image_path_obj.parent / f"result_{image_path_obj.name}")
            
            with open(output_path, 'wb') as f:
                f.write(result_data)
            
            inference_time = time.time() - start_time
            
            print(f"✅ Inpainting completed in {inference_time:.2f}s")
            print(f"Result saved to: {output_path}")
            print("=" * 60)
            
            result_info = {
                "success": True,
                "object_removed": object_name,
                "output_path": output_path,
                "inference_time": inference_time
            }
            
            return output_path, result_info
                
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
