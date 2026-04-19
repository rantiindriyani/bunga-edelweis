import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
import time
import os

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="🌼 Edelweiss Smart Detection",
    page_icon="🌼",
    layout="wide"
)

# =====================================
# BASE DIRECTORY
# =====================================
BASE_DIR = os.path.dirname(__file__)

# =====================================
# LOAD YAML
# =====================================
@st.cache_data
def load_yaml():
    yaml_path = os.path.join(BASE_DIR, "data.yaml")

    if not os.path.exists(yaml_path):
        st.error("File data.yaml tidak ditemukan")
        return {"names": []}

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data.get("names", [])

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "best.pt")

    if not os.path.exists(model_path):
        st.error("Model best.pt tidak ditemukan")
        return None

    return YOLO(model_path)

model = load_model()

# =====================================
# DETECTION FUNCTION
# =====================================
def detect_image(image):
    start = time.time()

    results = model.predict(image, verbose=False)

    infer_time = time.time() - start

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)

            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100

            class_name = (
                CLASS_NAMES[cls_id]
                if cls_id < len(CLASS_NAMES)
                else "Unknown"
            )

            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 2)
            })

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="green",
                width=4
            )

            draw.text(
                (x1, y1 - 30),
                f"{class_name} {conf:.1f}%",
                fill="green",
                font=font
            )

    return image, detections, infer_time

# =====================================
# SIDEBAR MENU
# =====================================
menu = st.sidebar.radio(
    "🌼 Menu",
    ["🏠 Beranda", "📷 Deteksi Gambar", "🌼 Informasi Fase"]
)

# =====================================
# HOME
# =====================================
if menu == "🏠 Beranda":
    st.title("🌼 Edelweiss Smart Detection")
    st.write(
        "Sistem deteksi fase pertumbuhan bunga Edelweiss "
        "berbasis YOLOv11"
    )

    st.info(
        "Kelas deteksi:\n"
        "- Mekar\n"
        "- Penyemaian\n"
        "- Sangat Mekar"
    )

# =====================================
# IMAGE DETECTION
# =====================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Upload Gambar Bunga Edelweiss")

    uploaded = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

        if model:
            result_img, detections, infer_time = detect_image(
                image.copy()
            )

            st.subheader("Hasil Deteksi")
            st.image(result_img, use_container_width=True)

            st.success(
                f"Waktu deteksi: {infer_time:.3f} detik"
            )

            if detections:
                for d in detections:
                    st.write(
                        f"🌼 {d['class_name']} "
                        f"({d['confidence']}%)"
                    )
            else:
                st.warning("Tidak ada objek terdeteksi")

# =====================================
# INFORMATION PAGE
# =====================================
elif menu == "🌼 Informasi Fase":
    st.title("🌼 Informasi Fase Edelweiss")

    st.markdown("""
    **Penyemaian**  
    Tahap awal pertumbuhan bunga.

    **Mekar**  
    Kelopak bunga mulai terbuka.

    **Sangat Mekar**  
    Fase bunga mekar sempurna.
    """)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("Edelweiss Smart Detection © 2026")
