"""Text-to-image generation application using Stable Diffusion + Streamlit."""
from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
from image_gen_app import verify_filename

MODEL_ID = "sd-legacy/stable-diffusion-v1-5"

def load_pipeline():
    """Loads the Stable Diffusion pipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    return pipe.to("cuda")


def save_image(image):
    """UI for saving the generated image."""
    st.write("Save Options")
    filename = st.text_input(
        "Enter filename (default: my_image.png):", key="filename_input"
    )
    if st.button("💾 Save Image"):
        filename = verify_filename(filename) if filename else "my_image.png"
        image.save(filename)
        st.session_state["image_saved"] = filename
        st.session_state["show_save"] = False # Hide save options
        st.session_state.pop("generated_image", None) # Clear input field
        st.rerun() # Rerun to clear fields

def clear_input(key):
    """Clears the input field."""
    st.session_state[key] = ""

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Text-to-Image Generator", page_icon="🖼️", layout="centered")
    st.title("🖼️ Text-to-Image Generator")
    st.write("Generate images from text prompts using Stable Diffusion.")

    @st.cache_resource
    def get_cached_pipeline():
        return load_pipeline()

    model = get_cached_pipeline()

    if st.button("❌ Quit Program"):
        st.info("👋 Goodbye! The app has been stopped.")
        st.stop()

    if "image_saved" in st.session_state:
        st.success(f"Image saved as: {st.session_state['image_saved']}")
        del st.session_state["image_saved"]

    prompt = st.text_input("Enter your image prompt:", key="prompt_input")
    if st.button("✨ Generate Image",):
        if prompt.strip():
            with st.spinner("Generating image..."):
                image = model(prompt).images[0]
                st.image(image, caption="Generated Image", width="stretch")
                st.session_state['generated_image'] = image
                st.session_state['show_save'] = True  # Show save option
        else:
            st.error("Please enter a prompt to generate an image.")

    # Saving option only if image has been generated and not saved yet
    if st.session_state.get('show_save', False) and 'generated_image' in st.session_state:
        save_image(st.session_state['generated_image'])

if __name__ == "__main__":
    main()