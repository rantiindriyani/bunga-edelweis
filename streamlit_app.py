import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import yaml
import time
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌿 HerbaSmartAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# BASE PATH
# =====================================================
BASE_DIR = os.path.dirname(__file__)  # folder tempat streamlit_app.py berada

# =====================================================
# LOAD YAML DATA
# =====================================================
@st.cache_data
def load_yaml():
    yaml_path = os.path.join(BASE_DIR, "data-baru.yaml")
    if not os.path.exists(yaml_path):
        st.error("File 'data-baru.yaml' tidak ditemukan!")
        return {"names": [], "info": {}}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

yaml_data = load_yaml()
CLASS_NAMES = yaml_data.get("names", [])
CLASS_INFO = yaml_data.get("info", {})

# =====================================================
# LOAD YOLO MODEL
# =====================================================
@st.cache_resource
def load_model(model_path):
    model_file = os.path.join(BASE_DIR, model_path)
    if os.path.exists(model_file):
        return YOLO(model_file)
    return None

MODEL_PATHS = {
    "YOLOv11 Nano": "bestnano.pt",
    "YOLOv11 Small": "bestsmall.pt",
}

# =====================================================
# FUNGSI DETEKSI GAMBAR
# =====================================================
def detect_image(image: Image.Image, model):
    start = time.time()
    results = model.predict(image, verbose=False)
    infer_time = time.time() - start

    draw = ImageDraw.Draw(image)
    detections = []

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes is None:
            continue

        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) * 100
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
            info = CLASS_INFO.get(name, {})

            # ubah path gambar ke absolute
            img_file = info.get("gambar", "")
            img_path = os.path.join(BASE_DIR, img_file) if img_file else ""

            detections.append({
                "name": name,
                "confidence": round(conf, 2),
                "components": info.get("components", []),
                "benefits": info.get("benefits", []),
                "gambar_path": img_path
            })

            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
            draw.text((x1, y1 - 30), f"{name} {conf:.1f}%", fill="#00FF00", font=font)

    return image, detections, infer_time

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    menu = st.radio(
        "🌿 Menu",
        ["🏠 Beranda", "📷 Deteksi Gambar", "💊 Rekomendasi Manfaat"]
    )

# =====================================================
# BERANDA
# =====================================================
if menu == "🏠 Beranda":
    st.title("🌿 HerbaSmartAI")
    st.write("Sistem identifikasi daun herbal berbasis YOLOv11")

# =====================================================
# DETEKSI GAMBAR
# =====================================================
elif menu == "📷 Deteksi Gambar":
    st.title("📷 Deteksi Daun Herbal")

    yolo_choice = st.selectbox("Pilih Model YOLO", list(MODEL_PATHS.keys()))
    model = load_model(MODEL_PATHS[yolo_choice])

    uploaded = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])

    if uploaded and model:
        image = Image.open(uploaded).convert("RGB")
        result_img, detections, infer_time = detect_image(image.copy(), model)

        st.image(result_img, use_container_width=True)
        st.success(f"Waktu deteksi: {infer_time:.3f} detik")

        for d in detections:
            with st.expander(f"🌿 {d['name']} ({d['confidence']}%)", expanded=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    if d["gambar_path"] and os.path.exists(d["gambar_path"]):
                        st.image(d["gambar_path"], use_container_width=True)
                    else:
                        st.warning("Gambar referensi tidak ditemukan")

                with col2:
                    st.write("**Kandungan:**", ", ".join(d["components"]))
                    st.write("**Manfaat:**")
                    for b in d["benefits"]:
                        st.write(f"- {b}")

# =====================================================
# REKOMENDASI MANFAAT
# =====================================================
elif menu == "💊 Rekomendasi Manfaat":
    st.title("💊 Cari Daun Berdasarkan Manfaat")
    query = st.text_input("Masukkan kata kunci (contoh: batuk)")

    if query:
        found = False
        for leaf, info in CLASS_INFO.items():
            if any(query.lower() in b.lower() for b in info.get("benefits", [])):
                found = True
                with st.expander(f"🌿 {leaf}", expanded=True):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        img_file = info.get("gambar", "")
                        img_path = os.path.join(BASE_DIR, img_file) if img_file else ""
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        else:
                            st.warning("Gambar referensi tidak ditemukan")

                    with col2:
                        st.write("**Kandungan:**", ", ".join(info.get("components", [])))
                        for b in info.get("benefits", []):
                            st.write(f"- {b}")

        if not found:
            st.warning("❌ Tidak ditemukan tanaman herbal")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("HerbaSmartAI © 2024")
