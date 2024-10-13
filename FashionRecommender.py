## Libraries Required
#pip install langchain-huggingface
## For API Calls
#pip install huggingface_hub
#pip install transformers
#pip install accelerate
#pip install bitsandbytes
#pip install langchain
#pip install streamlit

import streamlit as st
import cv2
import numpy as np
import cohere  # Import Cohere SDK

# Set your Cohere API key
cohere_api_key = "NxNcA2AgZ2iOJT64jhQVvXb7dIAspbffhm8QsmxZ"  # Replace with your actual Cohere API key

# Initialize Cohere client
co = cohere.Client(cohere_api_key)

# Predefined colors with specified names (RGB format)
colors = [
    ((244, 242, 245), "Light Pinkish"),
    ((236, 235, 233), "Very Light Beige (Pink Undertone)"),
    ((250, 249, 247), "Very Pale (Cream)"),
    ((253, 251, 230), "Light Beige (Yellow Undertone)"),
    ((253, 246, 230), "Light Peach"),
    ((254, 247, 229), "Warm Beige (Light Yellow Undertone)"),
    ((250, 240, 239), "Soft Pink (Light)"),
    ((243, 234, 229), "Light Beige (Neutral Undertone)"),
    ((244, 241, 234), "Very Light Neutral"),
    ((251, 252, 244), "Very Light Cream (Neutral)"),
    ((252, 248, 237), "Light Tan (Neutral)"),
    ((254, 246, 225), "Light Warm Beige (Yellow Undertone)"),
    ((255, 249, 225), "Peachy Beige"),
    ((241, 231, 195), "Light Tan (Yellow Undertone)"),
    ((239, 226, 173), "Light Yellowish (Golden Undertone)"),
    ((224, 210, 147), "Medium Yellowish (Golden Tan)"),
    ((242, 226, 151), "Soft Golden (Yellowish Beige)"),
    ((235, 214, 159), "Warm Tan (Yellow Undertone)"),
    ((235, 217, 133), "Golden Tan (Yellow)"),
    ((227, 196, 103), "Medium Tan (Golden Undertone)"),
    ((225, 193, 106), "Amber (Golden Brown)"),
    ((223, 193, 123), "Medium Tan (Yellowish Brown)"),
    ((222, 184, 119), "Sun-Kissed (Golden Tan)"),
    ((199, 164, 100), "Deep Tan (Warm Yellow Undertone)"),
    ((188, 151, 98), "Rich Tan (Warm Brown)"),
    ((158, 107, 87), "Medium Brown (Reddish Undertone)"),
    ((142, 88, 62), "Deep Brown (Chestnut Undertone)"),
    ((121, 77, 48), "Caramel Brown (Deep)"),
    ((100, 49, 22), "Deep Brown (Burnt Sienna)"),
    ((101, 48, 32), "Rich Brown (Reddish Undertone)"),
    ((96, 49, 33), "Mahogany Brown (Rich Red Undertone)"),
    ((87, 50, 41), "Burgundy Brown (Deep Red Undertone)"),
    ((64, 32, 21), "Chocolate Brown (Deep)"),
    ((49, 37, 41), "Dark Taupe (Neutral Undertone)"),
    ((27, 28, 46), "Very Deep Olive (Greenish Undertone)"),
]

def find_closest_color(rgb_color):
    """Find the closest color from predefined colors."""
    closest_color = min(colors, key=lambda color: np.linalg.norm(np.array(color[0]) - np.array(rgb_color)))
    return closest_color

def extract_skin_color(image):
    """Extract the dominant skin color from the image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    # Calculate the average skin color
    avg_skin_color = cv2.mean(skin, mask=skin_mask)[:3]
    return tuple(map(int, avg_skin_color))

def get_outfit_recommendation(skin_color, occasion, gender, time_of_day):
    """Get outfit recommendations from the Cohere language model."""
    #co = cohere.Client(cohere_api_key)

    # Prepare the prompt for the model
    prompt = (f"Considering the skin color '{skin_color}', the occasion '{occasion}', "
              f"the gender '{gender}', and the time of day '{time_of_day}', "
              "please provide a detailed outfit suggestion including suitable colors and styles.")

    # Generate outfit suggestion using Cohere
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    
    return response.generations[0].text.strip()

# Streamlit UI
st.title("Outfit Recommendation Based on Skin Color")
st.write("Upload an image of a person to extract their skin color and get outfit recommendations based on an occasion.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
occasion = st.text_input("Enter the occasion (e.g., formal dinner):")
gender = st.selectbox("Select Gender:", ["Male", "Female", "Other"])
time_of_day = st.selectbox("Select Time of Day:", ["Day", "Night"])

if uploaded_file and occasion:
    # Read and process the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Extract the dominant skin color from the image
    avg_skin_color = extract_skin_color(image)

    # Find the closest match in the predefined colors
    closest_color = find_closest_color(avg_skin_color)

    # Display extracted skin color and closest color name
    st.write(f"Extracted Skin Color (RGB): {avg_skin_color}")
    st.write(f"Closest Predefined Color (RGB): {closest_color[0]}, Name: {closest_color[1]}")

    # Get outfit recommendation using Cohere API
    outfit_recommendation = get_outfit_recommendation(closest_color[1], occasion, gender, time_of_day)
    st.write(f"Outfit Recommendation: {outfit_recommendation}")
