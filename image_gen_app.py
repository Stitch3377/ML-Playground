"""An text-to-image generation application using Stable Diffusion."""
from diffusers import StableDiffusionPipeline
import torch

MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
PIPE = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
PIPE = PIPE.to("cuda")

def generate_image(prompt):
    """Generates an image based on the provided prompt using Stable Diffusion."""
    image = PIPE(prompt).images[0]
    return image

def save_image(image):
    """Prompts the user to save the generated image."""
    save_option = input("Would you like to save this image? (yes or no): ")
    match save_option.lower():
        case 'yes' | 'y':
            filename = input("Enter the filename to save the image (default is 'my_image.png'): ")
            filename = verify_filename(filename) if filename else "my_image.png"
            image.save(filename)
            print(f"Image saved as {filename}")
        case 'no' | 'n':
            print("Image not saved.")
        case _:
            print(f"Invalid option: \"{save_option}\". Please enter yes or no.")
            save_image(image)


def verify_filename(filename):
    """Ensures the filename ends with .png extension."""
    if not filename.endswith('.png'):
        filename += '.png'
    return filename

def main():
    """Main function to run the image generation application."""
    while True:
        user_input = input("Enter a prompt for image generation (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        generated_image = generate_image(user_input)
        print("Image generated successfully!")
        generated_image.show()
        save_image(generated_image)

if __name__ == "__main__":
    main()