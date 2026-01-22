# pip install streamlit scikit-image matplotlib pandas seaborn
import streamlit as st
import os
import pandas as pd
import numpy as np
from skimage import io, color, filters, morphology, measure
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plot style for consistency
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'axes.facecolor':'#f7f7f7', 'figure.facecolor':'#f7f7f7', 'axes.grid':True})

# Load your dataset once
df = pd.read_csv(r"dt_ERA_corrected - Copy.csv")
df["Image_Name"] = df["image_file"].apply(lambda x: os.path.basename(str(x)).lower())

def load_rgb_image(image_path):
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)
        img = (img * 255).astype(np.uint8)
    return img

def segment_endometrium(img):
    gray = color.rgb2gray(img) if img.ndim == 3 else img
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh
    clean_mask = morphology.remove_small_objects(binary, min_size=200)
    clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=300)
    labeled = measure.label(clean_mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return None, None, None
    region = max(regions, key=lambda r: r.area)
    coords = region.coords
    y_min, y_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    thickness_pixels = y_max - y_min
    pixel_to_mm_factor = 0.0158
    thickness_mm = thickness_pixels * pixel_to_mm_factor
    return gray, coords, thickness_mm

st.title("ERA Patient Image Segmentation and Details")

uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    temp_file_path = os.path.join("temp_uploaded_image", uploaded_file.name)
    os.makedirs("temp_uploaded_image", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = load_rgb_image(temp_file_path)
    gray, coords, thickness_mm = segment_endometrium(img)

    if gray is not None:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(gray, cmap="gray")
        ax.plot(coords[:, 1], coords[:, 0], '.', color='#1976d2', markersize=1)
        ax.set_title(f"Segmented Endometrium\nThickness approx: {thickness_mm:.2f} mm", fontsize=16, color='#374151')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No endometrium detected in image.")

    img_name = uploaded_file.name.lower()
    patient_row = df[df["Image_Name"] == img_name]

    if not patient_row.empty:
        st.subheader("Patient Details from Dataset")
        pd.set_option('display.max_columns', None)
        st.write(patient_row.drop(columns=["image_file", "Image_Name"]))

        era_status = patient_row.iloc[0]["ERA_status"]
        patient_thickness = patient_row.iloc[0]["endometrial_thickness_mm"]
        patient_cycle_day = patient_row.iloc[0]["cycle_day"]

        st.success(f"Predicted ERA Status: {era_status}")

        # ERA Status Distribution
        st.subheader("ERA Status Distribution in Dataset")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor('#21293c')  # dark background
        ax2.set_facecolor('#21293c')
        palette = {'Pre-receptive': '#157afe', 'Receptive': '#33d4fa', 'Post-receptive': '#33d4fa'}
        sns.countplot(x="ERA_status", data=df, order=df["ERA_status"].value_counts().index, ax=ax2, palette=palette)
        ax2.set_title("ERA Status Distribution", fontsize=16, color='#ffffff')
        ax2.tick_params(axis='x', colors='#eeeeee')
        ax2.tick_params(axis='y', colors='#eeeeee')
        ax2.set_xlabel("", color='#eeeeee')
        ax2.set_ylabel("", color='#eeeeee')
        ax2.grid(True, color="#34415d", alpha=0.2)
        for spine in ax2.spines.values():
            spine.set_color('#21293c')
        st.pyplot(fig2)

        # Thickness histogram for predicted ERA status
        st.subheader(f"Endometrial Thickness Distribution for ERA Status: {era_status}")
        filtered = df[df["ERA_status"] == era_status]
        fig3, ax3 = plt.subplots()
        sns.histplot(filtered["endometrial_thickness_mm"], kde=True, ax=ax3, color="#009688", bins=20)
        ax3.set_title(f"Endometrial Thickness for ERA Status {era_status}", fontsize=14, color='#374151')
        ax3.set_xlabel("Thickness (mm)", color='#374151')
        ax3.set_ylabel("Count", color='#374151')
        st.pyplot(fig3)

        # Patient vs. dataset scatter plot
        st.subheader("Patient's Endometrial Thickness vs. Entire Dataset")
        fig4, ax4 = plt.subplots()
        ax4.scatter(df["endometrial_thickness_mm"], df["cycle_day"], c="#b0bec5", alpha=0.5, label="Dataset")
        ax4.scatter(patient_row["endometrial_thickness_mm"], patient_row["cycle_day"], c="#0579f5", label="This Patient", s=120, marker="*", edgecolor="black", linewidth=1)
        ax4.set_xlabel("Endometrial Thickness (mm)", color='#374151')
        ax4.set_ylabel("Cycle Day", color='#374151')
        ax4.set_title("Thickness vs. Cycle Day", fontsize=14, color='#374151')
        ax4.legend(frameon=True, facecolor="#f7f7f7", edgecolor="#666")
        st.pyplot(fig4)

        # Additional 1: Cycle Day Distribution with patient cycle day highlight
        st.subheader("Cycle Day Distribution with Patient Cycle Day Highlight")
        fig5, ax5 = plt.subplots()
        sns.histplot(df["cycle_day"], bins=30, kde=False, color="#1976d2", ax=ax5)
        ax5.axvline(patient_cycle_day, color='red', linestyle='--', lw=2, label="Patient Cycle Day")
        ax5.set_xlabel("Cycle Day", color='#374151')
        ax5.set_ylabel("Count", color='#374151')
        ax5.set_title("Cycle Day Distribution", fontsize=14, color='#374151')
        ax5.legend()
        st.pyplot(fig5)

        # Additional 3: Thickness vs Cycle Day scatter by ERA status with patient highlight
        st.subheader("Thickness vs. Cycle Day with Patient Highlight")
        fig7, ax7 = plt.subplots()
        sns.scatterplot(data=df, x="endometrial_thickness_mm", y="cycle_day", hue="ERA_status", palette=palette, ax=ax7, alpha=0.5)
        ax7.scatter(patient_thickness, patient_cycle_day, color='red', s=150, marker='*', label='Patient')
        ax7.set_xlabel("Endometrial Thickness (mm)", color='#374151')
        ax7.set_ylabel("Cycle Day", color='#374151')
        ax7.set_title("Thickness vs. Cycle Day", fontsize=14, color='#374151')
        ax7.legend()
        st.pyplot(fig7)

    else:
        st.error(f"No matching patient record found for image: {img_name}")


