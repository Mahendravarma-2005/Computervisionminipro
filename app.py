
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

st.set_page_config(
    page_title="Smart Waste Detection",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Detection System")
st.write(
    "Upload a waste image and let AI identify "
    "different waste materials."
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Detect Waste", type="primary"):

        with st.spinner("Detecting waste objects..."):

            results = model.predict(
                source=image,
                conf=confidence,
                device="cpu",
                verbose=False
            )

            result = results[0]

            output_image = result.plot()[:, :, ::-1]

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    output_image,
                    caption="AI Detection Output",
                    use_container_width=True
                )

            if result.boxes is not None and len(result.boxes):

                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                names = [
                    result.names[class_id]
                    for class_id in class_ids
                ]

                counts = Counter(names)

                st.divider()
                st.subheader("Waste Analysis")

                c1, c2 = st.columns(2)

                c1.metric("Total Objects", len(names))
                c2.metric("Categories Detected", len(counts))

                st.write("### Category-wise Count")

                st.bar_chart(dict(counts))

                for category, count in counts.items():
                    st.write(f"♻️ {category}: {count}")

            else:
                st.warning("No waste objects detected.")
