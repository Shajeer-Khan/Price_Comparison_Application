# import streamlit as st
# from transformers import ViTForImageClassification, ViTImageProcessor
# import torch
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json
# import requests
# from extract import *
# from scrapping import *
# from io import BytesIO

# def load_models():
#     vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
#     vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
#     custom_model = tf.keras.models.load_model("cnn_model.keras")
#     return vit_model, vit_processor, custom_model

# def predict_with_vit(image, model, processor):
#     inputs = processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predictions = outputs.logits.softmax(-1)
#     scores = predictions[0].numpy()
#     labels = [model.config.id2label[i] for i in range(len(scores))]
#     return list(zip(labels, scores))

# def predict_with_custom_model(image, model):
#     img = np.array(image.convert('L'))
#     img = cv2.resize(img, (28, 28))
#     img = img / 255.0
#     img = img.reshape(-1, 28, 28, 1)
#     predictions = model.predict(img, verbose=0)[0]
    
#     with open("metadata.json", "r") as f:
#         metadata = json.load(f)
#     labels = metadata.get("unique_product_types", [])
#     return list(zip(labels, predictions))

# def combine_predictions(vit_preds, custom_preds):
#     vit_dict = {label.lower(): score for label, score in vit_preds}
#     custom_dict = {label.lower(): score for label, score in custom_preds}
#     all_labels = set(vit_dict.keys()) | set(custom_dict.keys())
    
#     combined_scores = {}
#     for label in all_labels:
#         vit_score = vit_dict.get(label, 0)
#         custom_score = custom_dict.get(label, 0)
        
#         if vit_score > 0.3 and custom_score > 0.3:
#             combined_score = (vit_score * 0.6 + custom_score * 0.4) * 1.2
#         else:
#             combined_score = max(vit_score, custom_score) * 0.8 + min(vit_score, custom_score) * 0.2
#         combined_scores[label] = min(combined_score, 1.0)
    
#     return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

# def main():
#     st.title("Fashion Product Recognition & Similarity Search")
    
#     try:
#         vit_model, vit_processor, custom_model = load_models()
#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         return

#     uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)
        
#         with st.spinner('Analyzing image...'):
#             try:
#                 # Get predictions from both models
#                 vit_predictions = predict_with_vit(image, vit_model, vit_processor)
#                 custom_predictions = predict_with_custom_model(image, custom_model)
#                 combined_predictions = combine_predictions(vit_predictions, custom_predictions)
                
#                 # Display top predictions
#                 st.subheader("Top Predictions:")
#                 for i, (label, score) in enumerate(combined_predictions[:3], 1):
#                     st.write(f"{i}. {label.title()} (Confidence: {score:.2%})")
                
#                 # Use top prediction for similarity search
#                 predicted_label = combined_predictions[0][0]
                
#                 # Extract features for similarity search
#                 img_for_features = np.array(image.convert('RGB'))
#                 uploaded_features = extract_features(img_for_features)
                
#                 # Scrape products from multiple websites
#                 st.write("Searching similar products on Amazon and eBay...")
#                 scraped_products = []
#                 scraped_products.extend(scrape_amazon_products(predicted_label, max_products=10))
#                 scraped_products.extend(scrape_ebay_products(predicted_label, max_products=10))

#                 if not scraped_products:
#                     st.error("No products found on any platform.")
#                 else:
#                     products_with_similarity = []

#                     for product in scraped_products:
#                         try:
#                             image_urls = product.get("image_urls", [])
#                             best_similarity = 0
#                             best_image = None

#                             for image_url in image_urls:
#                                 response = requests.get(image_url)
#                                 if response.status_code != 200:
#                                     continue
                                    
#                                 scraped_img = Image.open(BytesIO(response.content))
#                                 scraped_img_array = np.array(scraped_img.convert('RGB'))
#                                 scraped_features = extract_features(scraped_img_array)
#                                 similarity_score = compute_similarity(uploaded_features, scraped_features)

#                                 if similarity_score > best_similarity:
#                                     best_similarity = similarity_score
#                                     best_image = scraped_img

#                             if best_similarity > 0.2:  
#                                 products_with_similarity.append({
#                                     "image": best_image,
#                                     "similarity": best_similarity,
#                                     "price": product["price"],
#                                     "link": product["link"],
#                                     "title": product["title"],
#                                     "source": product["source"]
#                                 })

#                         except Exception as e:
#                             continue

#                     products_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)

#                     if products_with_similarity:
#                         # Display most similar product
#                         most_similar = products_with_similarity[0]
#                         st.subheader("Most Similar Product:")
#                         col1, col2 = st.columns([1, 2])
#                         with col1:
#                             st.image(most_similar["image"], width=200)
#                         with col2:
#                             st.write(f"**Title:** {most_similar['title']}")
#                             st.write(f"**Price:** {most_similar['price']}")
#                             st.write(f"**Source:** {most_similar['source']}")
#                             st.write(f"**Similarity Score:** {most_similar['similarity']:.2f}")
#                             st.write(f"**Product Link:** [View on {most_similar['source']}]({most_similar['link']})")

#                         # Display other similar products
#                         if len(products_with_similarity) > 1:
#                             st.subheader("Other Similar Products:")
#                             for product in products_with_similarity[1:]:
#                                 col1, col2 = st.columns([1, 2])
#                                 with col1:
#                                     st.image(product["image"], width=150)
#                                 with col2:
#                                     st.write(f"**Title:** {product['title']}")
#                                     st.write(f"**Price:** {product['price']}")
#                                     st.write(f"**Source:** {product['source']}")
#                                     st.write(f"**Similarity Score:** {product['similarity']:.2f}")
#                                     st.write(f"**Product Link:** [View on {product['source']}]({product['link']})")
#                                 st.divider()
#                     else:
#                         st.warning("No products met the similarity threshold.")
                        
#             except Exception as e:
#                 st.error(f"Error during prediction: {str(e)}")

# if __name__ == "__main__":
#     main()

import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
from extract import *
from scrapping import *
from io import BytesIO
import cv2

def load_models():
    vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    custom_model = tf.keras.models.load_model("cnn_model.keras")
    return vit_model, vit_processor, custom_model

def predict_with_vit(image, model, processor):
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess the image explicitly as a numpy array, then pass to processor
    img_array = np.array(image)  # Shape: [H, W, C]
    if img_array.ndim == 2:  # Handle grayscale images
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Pass the image to the processor
    inputs = processor(images=img_array, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits.softmax(-1)
    scores = predictions[0].numpy()
    labels = [model.config.id2label[i] for i in range(len(scores))]
    return list(zip(labels, scores))

def predict_with_custom_model(image, model):
    img = np.array(image.convert('L'))
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(-1, 28, 28, 1)
    predictions = model.predict(img, verbose=0)[0]
    
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    labels = metadata.get("unique_product_types", [])
    return list(zip(labels, predictions))

def combine_predictions(vit_preds, custom_preds):
    vit_dict = {label.lower(): score for label, score in vit_preds}
    custom_dict = {label.lower(): score for label, score in custom_preds}
    all_labels = set(vit_dict.keys()) | set(custom_dict.keys())
    
    combined_scores = {}
    for label in all_labels:
        vit_score = vit_dict.get(label, 0)
        custom_score = custom_dict.get(label, 0)
        
        if vit_score > 0.3 and custom_score > 0.3:
            combined_score = (vit_score * 0.6 + custom_score * 0.4) * 1.2
        else:
            combined_score = max(vit_score, custom_score) * 0.8 + min(vit_score, custom_score) * 0.2
        combined_scores[label] = min(combined_score, 1.0)
    
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

def fetch_image_from_url(url):
    """Fetch an image from a URL and return it as a PIL Image."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error(f"Failed to fetch image from URL. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching image from URL: {str(e)}")
        return None

def main():
    st.title("Fashion Product Recognition & Similarity Search")
    
    try:
        vit_model, vit_processor, custom_model = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # Add option to choose between upload and URL
    input_method = st.radio("Choose input method:", ("Upload Image", "Enter Image URL"))

    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        url = st.text_input("Enter the URL of an image:")
        if url:
            image = fetch_image_from_url(url)

    if image is not None:
        st.image(image, caption="Input Image", use_container_width=True)
        
        with st.spinner('Analyzing image...'):
            try:
                # Get predictions from both models
                vit_predictions = predict_with_vit(image, vit_model, vit_processor)
                custom_predictions = predict_with_custom_model(image, custom_model)
                combined_predictions = combine_predictions(vit_predictions, custom_predictions)
                
                # Display top predictions
                st.subheader("Top Predictions:")
                for i, (label, score) in enumerate(combined_predictions[:3], 1):
                    st.write(f"{i}. {label.title()} (Confidence: {score:.2%})")
                
                # Use top prediction for similarity search
                predicted_label = combined_predictions[0][0]
                
                # Extract features for similarity search
                img_for_features = np.array(image.convert('RGB'))
                uploaded_features = extract_features(img_for_features)
                
                # Scrape products from multiple websites
                st.write("Searching similar products on Amazon and eBay...")
                scraped_products = []
                scraped_products.extend(scrape_amazon_products(predicted_label, max_products=10))
                scraped_products.extend(scrape_ebay_products(predicted_label, max_products=10))

                if not scraped_products:
                    st.error("No products found on any platform.")
                else:
                    products_with_similarity = []

                    for product in scraped_products:
                        try:
                            image_urls = product.get("image_urls", [])
                            best_similarity = 0
                            best_image = None

                            for image_url in image_urls:
                                response = requests.get(image_url)
                                if response.status_code != 200:
                                    continue
                                    
                                scraped_img = Image.open(BytesIO(response.content))
                                scraped_img_array = np.array(scraped_img.convert('RGB'))
                                scraped_features = extract_features(scraped_img_array)
                                similarity_score = compute_similarity(uploaded_features, scraped_features)

                                if similarity_score > best_similarity:
                                    best_similarity = similarity_score
                                    best_image = scraped_img

                            if best_similarity > 0.2:  
                                products_with_similarity.append({
                                    "image": best_image,
                                    "similarity": best_similarity,
                                    "price": product["price"],
                                    "link": product["link"],
                                    "title": product["title"],
                                    "source": product["source"]
                                })

                        except Exception as e:
                            continue

                    products_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)

                    if products_with_similarity:
                        # Display most similar product
                        most_similar = products_with_similarity[0]
                        st.subheader("Most Similar Product:")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(most_similar["image"], width=200)
                        with col2:
                            st.write(f"**Title:** {most_similar['title']}")
                            st.write(f"**Price:** {most_similar['price']}")
                            st.write(f"**Source:** {most_similar['source']}")
                            st.write(f"**Similarity Score:** {most_similar['similarity']:.2f}")
                            st.write(f"**Product Link:** [View on {most_similar['source']}]({most_similar['link']})")

                        # Display other similar products
                        if len(products_with_similarity) > 1:
                            st.subheader("Other Similar Products:")
                            for product in products_with_similarity[1:]:
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(product["image"], width=150)
                                with col2:
                                    st.write(f"**Title:** {product['title']}")
                                    st.write(f"**Price:** {product['price']}")
                                    st.write(f"**Source:** {product['source']}")
                                    st.write(f"**Similarity Score:** {product['similarity']:.2f}")
                                    st.write(f"**Product Link:** [View on {product['source']}]({product['link']})")
                                st.divider()
                    else:
                        st.warning("No products met the similarity threshold.")
                        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()