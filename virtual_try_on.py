import io
import argparse
from typing import List
from PIL import Image
from idm_viton_abstraction import idm_viton_abstraction

# Initialize the idm_viton instance
idm_viton_instance = idm_viton_abstraction()

def process_images(garm_img_path, human_img_path, garment_description, garment_type, is_checked):
    try:
        # Open images
        garm_img = Image.open(garm_img_path)
        human_img = Image.open(human_img_path)

        if garment_type not in ["dresses", "upper_body", "lower_body"]:
            raise Exception("Invalid Garment Type. Available options are dresses, upper_body, lower_body")

        denoise_steps = 30
        seed = 42

        # Example dictionary for your process
        dict_inputs = {
            "background": human_img,
            "layers": [human_img],  # Depending on your application logic
            "composite": None,
        }

        # Process the image through your function
        output_image, mask_image = idm_viton_instance.start_tryon(
            dict_inputs,
            garm_img,
            garment_description,
            is_checked,
            denoise_steps,
            seed,
            garment_type,
        )

        # Save the output and mask images
        output_image.save("output.png", format="PNG")
        mask_image.save("mask.png", format="PNG")

        print("Images processed and saved as 'output.png' and 'mask.png'.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for virtual try-on.")
    parser.add_argument("garm_img", type=str, help="Path to the garment image")
    parser.add_argument("human_img", type=str, help="Path to the human image")
    parser.add_argument("description", type=str, help="Description of the garment")
    parser.add_argument("garment_type", type=str, help="Type of the garment: dresses, upper_body, lower_body")
    parser.add_argument("is_checked", type=bool, help="Boolean flag for checking")
    
    args = parser.parse_args()
    
    process_images(args.garm_img, args.human_img, args.description, args.garment_type, args.is_checked)
