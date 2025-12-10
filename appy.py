import streamlit as st
import google.generativeai as genai
import zipfile
import io
import time
import random
from PIL import Image

# --- CONFIGURATION ---
BATCH_SIZE = 10 
MAX_PAIRS = 300
# Use the Gemini Flash preview image model
MODEL_NAME = "gemini-2.5-flash-preview-image" 

st.set_page_config(page_title="Gemini Flash Spot-the-Difference Creator", layout="wide")

st.title("⚡ Gemini Flash Spot-the-Difference Creator")
st.markdown(f"Generating image pairs using **{MODEL_NAME}**.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password")
    theme_input = st.text_input("Theme", value="Cyberpunk Street Food Stall")
    num_pairs = st.number_input("Total Pairs", min_value=1, max_value=MAX_PAIRS, value=20)
    st.info("Files download automatically every 10 pairs.")

# --- PROMPTS ---
DIFFERENCE_OPTIONS = [
    "Remove one small object completely.",
    "Change the color of one prominent item.",
    "Shift the position of an object slightly.",
    "Change a pattern (stripes to spots).",
    "Flip an object horizontally.",
    "Remove a shadow.",
    "Change texture from smooth to rough.",
    "Make an outline thicker.",
    "Remove a reflection in a surface.",
    "Change the perspective of a background element."
]

def get_base_prompt(theme):
    return f"""
    Generate an image. Create an image with the following style: bold line cartoon
    with bright colors. The illustration should depict a detailed, wide-angle scene
    of a {theme} environment. The entire composition must be centered on
    elements, objects and characters clearly associated with {theme}.
    FULL BLEED, NO BORDERS.
    """

def get_diff_prompt(theme, diffs):
    return f"""
    Generate an image nearly identical to the previous {theme} scene.
    Style: Bold line cartoon, bright colors.
    Apply exactly these 3 changes:
    1. {diffs[0]}
    2. {diffs[1]}
    3. {diffs[2]}
    Everything else must remain as similar as possible.
    FULL BLEED, NO BORDERS.
    """

# --- LOGIC ---

def generate_pair(model, theme):
    """
    Attempts to generate two images using the GenerativeModel class.
    """
    # 1. Select 3 unique differences
    current_diffs = random.sample(DIFFERENCE_OPTIONS, 3)
    
    try:
        # Prompt 1: Base Image
        # Note: 'response.parts' or 'response.text' is for text.
        # For images in Gemini Flash, we typically ask it to output valid image data 
        # or we use the model's specific tools.
        # If the API returns a URI or base64, we handle it here.
        # This implementation assumes the standard 'generate_content' returns an image part 
        # as seen in recent experimental multimodal updates.
        
        prompt1 = get_base_prompt(theme)
        response1 = model.generate_content(prompt1)
        
        # Extract Image 1
        # Check if response contains image data (common in non-Imagen Gemini models)
        if not response1.parts or not hasattr(response1.parts[0], 'inline_data'):
             # Fallback: Some Flash versions output a URI/link if they can't render inline
             st.warning("Model returned text instead of data. It might not support direct image gen yet.")
             return None, None
             
        img1_data = response1.parts[0].inline_data.data
        img1 = Image.open(io.BytesIO(img1_data))

        # Prompt 2: Diff Image
        # We pass img1 as history/context if possible, or just prompt fresh
        prompt2 = get_diff_prompt(theme, current_diffs)
        
        # We attempt to pass the first image as input to "ground" the second one (Multimodal input)
        # This gives the best chance of consistency with Gemini Flash.
        response2 = model.generate_content([prompt2, img1])
        
        if not response2.parts or not hasattr(response2.parts[0], 'inline_data'):
             return None, None
             
        img2_data = response2.parts[0].inline_data.data
        img2 = Image.open(io.BytesIO(img2_data))

        return img1, img2

    except Exception as e:
        # Debugging helper: Print the actual error if generation fails
        print(f"Gen Error: {e}")
        return None, None

def create_zip(batch, start_idx):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (p1, p2) in enumerate(batch):
            idx = start_idx + i + 1
            
            b1 = io.BytesIO()
            p1.save(b1, format="PNG")
            zf.writestr(f"{idx}.png", b1.getvalue())
            
            b2 = io.BytesIO()
            p2.save(b2, format="PNG")
            zf.writestr(f"{idx}(1).png", b2.getvalue())
    return buf

# --- UI EXECUTION ---

if st.button("Generate Images"):
    if not api_key:
        st.error("API Key required.")
    else:
        genai.configure(api_key=api_key)
        
        try:
            # Initialize Gemini Flash
            model = genai.GenerativeModel(MODEL_NAME)
            
            batch = []
            total_count = 0
            prog_bar = st.progress(0)
            status = st.empty()

            for i in range(num_pairs):
                status.text(f"Generating pair {i+1}/{num_pairs}...")
                
                img1, img2 = generate_pair(model, theme_input)
                
                if img1 and img2:
                    batch.append((img1, img2))
                    
                    # Preview recent
                    with st.expander(f"Preview {i+1}"):
                        c1, c2 = st.columns(2)
                        c1.image(img1, caption="Base")
                        c2.image(img2, caption="Edited")
                else:
                    st.error(f"Failed to generate pair {i+1}. Skipping.")

                # Batch Download logic (Every 10 or at end)
                if len(batch) >= BATCH_SIZE or (i == num_pairs - 1 and batch):
                    zip_file = create_zip(batch, total_count)
                    
                    st.download_button(
                        f"Download Pairs {total_count+1}-{total_count+len(batch)}",
                        data=zip_file.getvalue(),
                        file_name=f"pairs_{total_count+1}_to_{total_count+len(batch)}.zip",
                        mime="application/zip",
                        key=f"btn_{total_count}"
                    )
                    
                    total_count += len(batch)
                    batch = [] # Reset batch
                
                prog_bar.progress((i+1)/num_pairs)
                time.sleep(2) # Pause to respect Flash rate limits

            status.success("Job Done.")

        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
