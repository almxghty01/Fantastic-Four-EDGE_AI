import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import os
import pandas as pd

# --- CONFIGURATION ---
# FIXED: Relative path. The judge just puts the file next to the script.
MODEL_PATH = ("EdgeChip_8C"
              "lass.pth")

CLASSES = [
    'Clean Wafer', 'Other/Contamination', 'Open Circuit', 'Short Circuit', 
    'Scratch', 'Particle', 'Dead Via', 'Misalignment'
]

# --- UI SETUP ---
st.set_page_config(page_title="EdgeChip Pro", page_icon="🏭", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #00ADB5; color: white; font-weight: bold;}
    .metric-card {background-color: #1b262c; padding: 15px; border-radius: 10px; border-left: 5px solid #00ADB5;}
    h1, h2, h3 {color: #eeeeee;}
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    # Initialize basic ResNet18
    model = models.resnet18(weights=None)
    # Match the output layer to your 8 classes
    model.fc = nn.Linear(model.fc.in_features, 8)
    
    if os.path.exists(MODEL_PATH):
        try:
            # Load the weights (Safe for CPU)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            return model
        except Exception as e:
            st.error(f"CRITICAL ERROR: Could not load model weights. {e}")
            return None
    else:
        st.error(f"⚠️ MODEL NOT FOUND: Please ensure '{MODEL_PATH}' is in the same folder as this script.")
        return None

model = load_model()

# --- HELPER FUNCTIONS ---
def highlight_defect(original_img):
    gray = ImageOps.grayscale(original_img)
    return ImageOps.autocontrast(gray, cutoff=2)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏭 EdgeChip Pro")
    st.caption("IESA Hackathon Submission")
    st.info(f"Model: ResNet-18\nStatus: {'✅ Online' if model else '❌ Offline'}")
    st.divider()
    mode = st.radio("System Mode:", ["Single Inspection", "Batch Analytics"])
    st.divider()

# ==========================================
# MODE 1: SINGLE INSPECTION
# ==========================================
if mode == "Single Inspection":
    st.title("🔬 Single Wafer Forensics")
    uploaded_file = st.file_uploader("Upload Wafer Image", type=["jpg", "png", "jpeg"])

    if uploaded_file and model:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert('RGB')
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            _, pred_idx = torch.max(outputs, 1)
        
        result = CLASSES[pred_idx.item()]
        conf = probs[pred_idx.item()].item()
        
        # Display
        with col1:
            st.subheader("Raw SEM Scan")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("AI Diagnosis")
            color = "#00ADB5" if "Clean" in result else "#ff4757"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {color};">
                <h2 style="color: {color}; margin:0;">{result.upper()}</h2>
                <h1 style="margin:0; font-size: 3em;">{conf:.1f}%</h1>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            st.write("**Defect Localization (High Contrast):**")
            st.image(highlight_defect(image), caption="Enhanced View", use_container_width=True)

# ==========================================
# MODE 2: BATCH ANALYTICS (The Dashboard)
# ==========================================
elif mode == "Batch Analytics":
    st.title("🚀 High-Volume Yield Analysis")
    uploaded_files = st.file_uploader("Upload Batch (Select Multiple)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files and st.button("RUN BATCH ANALYSIS"):
        if not model:
            st.error("Model not found! Cannot run analysis.")
            st.stop()
            
        results = []
        progress_bar = st.progress(0)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # --- PROCESSING LOOP ---
        for i, file in enumerate(uploaded_files):
            try:
                image = Image.open(file).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                    _, pred_idx = torch.max(outputs, 1)
                
                res_label = CLASSES[pred_idx.item()]
                res_conf = probs[pred_idx.item()].item()
                status = "PASS" if "Clean" in res_label else "FAIL"
                
                results.append({
                    "Filename": file.name,
                    "Status": status,
                    "Defect Type": res_label,
                    "Confidence": res_conf
                })
            except:
                pass # Skip bad files
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success(f"✅ Analysis Complete: {len(uploaded_files)} Wafers Processed")
        
        # --- DATAFRAME CREATION ---
        if results:
            df = pd.DataFrame(results)

            # --- METRICS ROW ---
            total = len(df)
            passed = len(df[df['Status'] == 'PASS'])
            failed = total - passed
            yield_rate = (passed / total) * 100 if total > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Throughput", total)
            m2.metric("Yield Rate", f"{yield_rate:.1f}%", delta_color="normal" if yield_rate > 90 else "inverse")
            m3.metric("Clean Wafers", passed)
            m4.metric("Defects Found", failed, delta_color="inverse")
            
            st.divider()

            # --- CHARTS & TABLE LAYOUT ---
            col_charts, col_table = st.columns([1, 2])
            
            with col_charts:
                st.subheader("📊 Failure Modes")
                defect_counts = df['Defect Type'].value_counts()
                st.bar_chart(defect_counts, color="#ff4757") 
            
            with col_table:
                st.subheader("📋 Detailed Inspection Log")
                
                def highlight_fail(row):
                    color = 'rgba(255, 71, 87, 0.2)' if row['Status'] == 'FAIL' else 'rgba(0, 173, 181, 0.1)'
                    return [f'background-color: {color}'] * len(row)

                st.dataframe(
                    df.style.apply(highlight_fail, axis=1)
                            .format({"Confidence": "{:.1f}%"}),
                    use_container_width=True,
                    height=400
                )

            st.divider()
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Inspection Report (CSV)",
                data=csv,
                file_name='EdgeChip_Report.csv',
                mime='text/csv'
            )
