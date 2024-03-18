import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

def kmeans_clustering(image_path):
    np_image = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    segmented_image = segmented_data.reshape((image.shape))
    unique_labels, counts = np.unique(segmented_data,axis=0, return_counts=True)

    labels_hist = {
        tuple(unique_labels[0]): 'Afforestation',
        tuple(unique_labels[1]): 'Deforestation'
    }
    label_list = [labels_hist[tuple(label)] for label in unique_labels]
    custom_colors = [(r / 255, g / 255, b / 255) for r, g, b in unique_labels]
    fig, ax = plt.subplots()
    ax.bar(label_list, counts, color=custom_colors)
    ax.set_ylabel('Count scale')
    ax.set_title('Bar Chart for Comparing ratio')
    return segmented_image,fig


st.title("Deforestation Detection")

uploaded_file = st.file_uploader("Choose an image...", type=['2jpg','png','jpeg','pdf'])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    segmented_image,fig = kmeans_clustering(uploaded_file)

    st.image(segmented_image, caption="Segmented Image.", use_column_width=True)
    st.pyplot(fig)