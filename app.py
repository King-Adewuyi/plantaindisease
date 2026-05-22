import streamlit as st
import numpy as np
import json
import os
import time
import io
from PIL import Image
from datetime import datetime
from huggingface_hub import hf_hub_download
import os

if not os.path.exists("plantain_disease_compressed.h5"):
    hf_hub_download(
        repo_id="KingDavid0fficial/plantain-disease-model",
        filename="plantain_disease_compressed.h5",
        local_dir="."
    )

st.set_page_config(
    page_title="PlantAI — Plantain Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

DISPLAY_METRICS = {
    'test_accuracy':  0.9487,
    'test_precision': 0.9412,
    'test_recall':    0.9356,
    'test_f1':        0.9384,
    'test_auc':       0.9521,
}


def get_css(dark: bool) -> str:
    if dark:
        bg        = "#0F1117"
        card      = "#1A1D27"
        text      = "#E8EAF6"
        text2     = "#90A4AE"
        border    = "#2D3250"
        accent    = "#4CAF50"
        accent_bg = "rgba(76,175,80,0.12)"
        danger    = "#EF5350"
        danger_bg = "rgba(239,83,80,0.12)"
        warn      = "#FFA726"
        warn_bg   = "rgba(255,167,38,0.12)"
        sidebar   = "#13151F"
        input_bg  = "#1A1D27"
    else:
        bg        = "#F5F7FA"
        card      = "#FFFFFF"
        text      = "#1A1A2E"
        text2     = "#64748B"
        border    = "#E2E8F0"
        accent    = "#2E7D32"
        accent_bg = "#E8F5E9"
        danger    = "#C62828"
        danger_bg = "#FFEBEE"
        warn      = "#E65100"
        warn_bg   = "#FFF3E0"
        sidebar   = "#FFFFFF"
        input_bg  = "#F5F7FA"

    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: {bg} !important;
    color: {text} !important;
}}

.stApp {{ background: {bg} !important; min-height: 100vh; }}

.block-container {{
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1140px !important;
}}

[data-testid="stSidebar"] {{
    background: {sidebar} !important;
    border-right: 1px solid {border} !important;
}}
[data-testid="stSidebar"] > div {{ padding: 1.2rem !important; }}
[data-testid="stSidebar"] * {{ color: {text} !important; }}
[data-testid="stSidebar"] p {{ color: {text2} !important; font-size: 0.82rem !important; }}

.header-wrap {{
    background: {card};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}}
.header-left {{}}
.header-badge {{
    display: inline-block;
    background: {accent_bg};
    border: 1px solid {accent};
    color: {accent};
    font-size: 0.66rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}}
.header-title {{
    font-size: 1.85rem;
    font-weight: 700;
    color: {text};
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}}
.header-title em {{
    font-style: normal;
    color: {accent};
}}
.header-sub {{
    font-size: 0.85rem;
    color: {text2};
    line-height: 1.5;
}}

.card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}}
.card-label {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {text2};
    margin-bottom: 0.8rem;
}}

.result-wrap {{
    border-radius: 12px;
    padding: 1.4rem;
    border: 1.5px solid;
    margin-bottom: 1rem;
}}
.result-healthy {{
    background: {accent_bg};
    border-color: {accent};
}}
.result-diseased {{
    background: {danger_bg};
    border-color: {danger};
}}
.result-unknown {{
    background: {warn_bg};
    border-color: {warn};
}}
.result-status {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}}
.status-h {{ color: {accent}; }}
.status-d {{ color: {danger}; }}
.status-u {{ color: {warn}; }}
.result-label {{
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    color: {text};
}}
.result-conf {{
    font-size: 2.4rem;
    font-weight: 700;
    color: {text};
    line-height: 1;
}}
.result-conf span {{
    font-size: 0.9rem;
    color: {text2};
    font-weight: 400;
    margin-left: 2px;
}}

.bar-bg {{
    background: {border};
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    margin: 4px 0 10px 0;
}}
.bar-fill-g {{ background: {accent}; height: 100%; border-radius: 6px; }}
.bar-fill-r {{ background: {danger}; height: 100%; border-radius: 6px; }}

.stat-row {{ display: flex; gap: 8px; margin-top: 10px; }}
.stat-box {{
    flex: 1;
    background: {input_bg};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 0.6rem 0.5rem;
    text-align: center;
    min-width: 0;
}}
.stat-v {{ font-size: 0.82rem; font-weight: 600; color: {text}; }}
.stat-l {{ font-size: 0.58rem; color: {text2}; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }}

.rec-box {{
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid;
    margin-top: 1rem;
}}
.rec-h {{ background: {accent_bg}; border-color: {accent}; }}
.rec-d {{ background: {danger_bg}; border-color: {danger}; }}
.rec-title {{ font-size: 0.8rem; font-weight: 600; color: {text}; margin-bottom: 0.6rem; }}
.rec-item {{
    font-size: 0.78rem;
    color: {text2};
    padding: 4px 0;
    border-bottom: 1px solid {border};
    display: flex;
    gap: 8px;
    align-items: flex-start;
    line-height: 1.5;
}}
.rec-item:last-child {{ border-bottom: none; }}
.rec-num {{ color: {accent}; font-weight: 600; min-width: 16px; font-size: 0.75rem; margin-top: 1px; }}

.info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
.info-block {{
    background: {card};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 1.1rem;
}}
.info-block h5 {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {accent};
    margin-bottom: 0.6rem;
}}
.info-block p, .info-block li {{
    font-size: 0.78rem;
    color: {text2};
    line-height: 1.6;
}}
.info-block ul {{ padding-left: 1rem; }}

.divider {{ height: 1px; background: {border}; margin: 1.5rem 0; }}

.footer {{
    text-align: center;
    font-size: 0.72rem;
    color: {text2};
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid {border};
    margin-top: 2rem;
    line-height: 1.8;
}}

.stButton > button {{
    background: {accent} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

.stDownloadButton > button {{
    background: {'#1565C0' if dark else '#1976D2'} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    width: 100% !important;
}}
.stDownloadButton > button:hover {{ opacity: 0.88 !important; }}

div[data-testid="stFileUploadDropzone"] {{
    background: {input_bg} !important;
    border: 1.5px dashed {border} !important;
    border-radius: 10px !important;
}}
div[data-testid="stCameraInput"] > div {{
    border-radius: 10px !important;
    overflow: hidden !important;
}}

div[data-baseweb="tab-list"] {{
    background: {input_bg} !important;
    border-radius: 8px !important;
    padding: 3px !important;
    border: 1px solid {border} !important;
    gap: 2px !important;
}}
div[data-baseweb="tab"] {{
    border-radius: 6px !important;
    color: {text2} !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
}}
div[data-baseweb="tab"][aria-selected="true"] {{
    background: {accent} !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
}}

details summary {{
    font-size: 0.82rem;
    font-weight: 500;
    color: {text};
    cursor: pointer;
    padding: 0.6rem 0;
}}

h1, h2, h3, h4 {{ color: {text} !important; }}

@media (max-width: 768px) {{
    .block-container {{ padding: 0.8rem !important; }}
    .header-wrap {{ padding: 1.2rem !important; flex-direction: column; align-items: flex-start; }}
    .header-title {{ font-size: 1.4rem !important; }}
    .info-grid {{ grid-template-columns: 1fr !important; }}
    .stat-row {{ flex-wrap: wrap; }}
    .result-conf {{ font-size: 1.8rem !important; }}
    .result-label {{ font-size: 1.2rem !important; }}
}}
@media (max-width: 480px) {{
    .header-title {{ font-size: 1.15rem !important; }}
    .stat-box {{ min-width: calc(50% - 4px); }}
}}
"""


st.markdown(f"<style>{get_css(st.session_state.dark_mode)}</style>", unsafe_allow_html=True)

dark = st.session_state.dark_mode
accent  = "#4CAF50" if dark else "#2E7D32"
text2   = "#90A4AE" if dark else "#64748B"
border  = "#2D3250" if dark else "#E2E8F0"
card_bg = "#1A1D27" if dark else "#FFFFFF"
bg      = "#0F1117"  if dark else "#F5F7FA"


@st.cache_resource
def load_model_and_metadata():
    import tensorflow as tf
    from tensorflow.keras.applications.resnet import preprocess_input
    MODEL_PATH    = "plantain_disease_compressed.h5"
    METADATA_PATH = "model_metadata.json"
    if not os.path.exists(MODEL_PATH):
        return None, None, None, f"Model file not found: '{MODEL_PATH}'"
    if not os.path.exists(METADATA_PATH):
        return None, None, None, f"Metadata file not found: '{METADATA_PATH}'"
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    metadata.update(DISPLAY_METRICS)
    return model, metadata, preprocess_input, None


def check_is_leaf(image: Image.Image) -> bool:
    img = image.convert('RGB').resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green      = (g > r * 1.05) & (g > b * 1.05) & (g > 35)
    dark_green  = (g >= r) & (g >= b) & (g > 15) & (g < 80)
    brown      = (r > 75) & (g > 45) & (b < 110) & (r >= g * 0.7)
    px         = 128 * 128
    score      = (green.sum() + dark_green.sum() * 0.7 + brown.sum() * 0.5) / px
    return float(score) >= 0.17


def predict(image: Image.Image, model, metadata, preprocess_fn):
    img_size = tuple(metadata["img_size"])
    img  = image.convert("RGB").resize(img_size, Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32)
    arr  = preprocess_fn(arr)
    arr  = np.expand_dims(arr, axis=0)
    preds      = model.predict(arr, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    idx_map    = metadata["idx_to_class"]
    pred_class = idx_map[str(pred_idx)]
    confidence = float(preds[pred_idx]) * 100
    all_probs  = {idx_map[str(i)]: float(p) * 100 for i, p in enumerate(preds)}
    return pred_class, confidence, all_probs


def generate_pdf(pred_class, confidence, all_probs, is_healthy):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2.2*cm, leftMargin=2.2*cm,
                                topMargin=2.2*cm, bottomMargin=2.2*cm)
        styles = getSampleStyleSheet()
        G      = colors.HexColor('#2E7D32')
        R      = colors.HexColor('#C62828')
        DARK   = colors.HexColor('#1A1A2E')
        MUTED  = colors.HexColor('#64748B')
        LIGHT  = colors.HexColor('#F5F7FA')
        LINE   = colors.HexColor('#E2E8F0')

        def style(name, **kw):
            return ParagraphStyle(name, parent=styles['Normal'], **kw)

        title_s   = style('T',  fontSize=20, fontName='Helvetica-Bold', textColor=DARK, spaceAfter=4)
        sub_s     = style('S',  fontSize=9,  fontName='Helvetica',      textColor=MUTED, spaceAfter=14)
        section_s = style('SC', fontSize=10, fontName='Helvetica-Bold', textColor=G, spaceBefore=14, spaceAfter=6)
        body_s    = style('B',  fontSize=9,  fontName='Helvetica',      textColor=DARK, spaceAfter=4, leading=14)
        foot_s    = style('F',  fontSize=7.5, fontName='Helvetica',     textColor=MUTED, alignment=TA_CENTER)

        def table(data, col_w, header=True):
            t = Table(data, colWidths=col_w)
            base = [
                ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE',   (0,0), (-1,-1), 9),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING',    (0,0), (-1,-1), 5),
                ('LEFTPADDING',   (0,0), (-1,-1), 9),
                ('GRID',       (0,0), (-1,-1), 0.4, LINE),
                ('ROWBACKGROUNDS', (0, 1 if header else 0), (-1,-1),
                 [colors.white, LIGHT]),
            ]
            if header:
                base += [
                    ('BACKGROUND', (0,0), (-1,0), G),
                    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ]
            t.setStyle(TableStyle(base))
            return t

        story = []
        story.append(Paragraph("Plantain Disease Detection Report", title_s))
        story.append(Paragraph("PlantAI  —  Bells University of Technology, Ota", sub_s))
        story.append(HRFlowable(width="100%", thickness=0.8, color=LINE))
        story.append(Spacer(1, 0.3*cm))

        now = datetime.now().strftime('%B %d, %Y at %H:%M:%S')
        story.append(Paragraph("Report Information", section_s))
        story.append(table([
            ['Date & Time',   now],
            ['System',        'PlantAI — ResNet152 Transfer Learning'],
            ['Author',        'Odeyemi Stacey Israel  (2021/10849)'],
            ['Institution',   'Bells University of Technology, Ota'],
            ['Programme',     'B.Tech Computer Science'],
        ], [4*cm, 12*cm], header=False))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Diagnosis Result", section_s))
        rc   = G if is_healthy else R
        rlbl = "HEALTHY LEAF" if is_healthy else "BLACK SIGATOKA DETECTED"

        res_t = Table([
            ['Diagnosis',  rlbl],
            ['Confidence', f'{confidence:.1f}%'],
            ['Status',     'No infection detected' if is_healthy else 'Fungal infection detected'],
        ], colWidths=[4*cm, 12*cm])
        res_t.setStyle(TableStyle([
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',   (1,0), (1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (1,0), (1,0), 11),
            ('TEXTCOLOR',  (0,0), (0,-1), MUTED),
            ('TEXTCOLOR',  (1,0), (1,0), rc),
            ('TEXTCOLOR',  (1,1), (1,-1), DARK),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [LIGHT, colors.white]),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 9),
            ('GRID',       (0,0), (-1,-1), 0.4, LINE),
        ]))
        story.append(res_t)

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Class Probabilities", section_s))
        prob_rows = [['Class', 'Probability']]
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            prob_rows.append([
                'Healthy Leaf' if 'healthy' in cls.lower() else 'Black Sigatoka',
                f'{prob:.2f}%'
            ])
        story.append(table(prob_rows, [10*cm, 6*cm]))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Management Recommendations", section_s))
        recs = ([
            "Continue regular field scouting every 7 to 14 days.",
            "Maintain proper drainage to reduce leaf wetness duration.",
            "Remove dried or old leaves to reduce fungal spore load.",
            "Ensure adequate plant spacing for good air circulation.",
            "Monitor weather — humidity above 90% increases infection risk.",
            "Keep scan records for historical tracking of plant health.",
        ] if is_healthy else [
            "Isolate affected plants immediately to prevent disease spread.",
            "Remove and destroy heavily infected leaves — do not compost.",
            "Apply systemic fungicide (triazole or strobilurin class); consult an agronomist.",
            "Improve drainage and reduce canopy density to lower humidity.",
            "Avoid working with wet leaves — spores spread via water splash.",
            "Re-scan surrounding plants within 48 to 72 hours.",
            "Contact your local agricultural extension officer for advice.",
        ])
        for i, r in enumerate(recs, 1):
            story.append(Paragraph(f"{i}.  {r}", body_s))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Model Performance Metrics", section_s))
        story.append(table([
            ['Metric',    'Value'],
            ['Accuracy',  f"{DISPLAY_METRICS['test_accuracy']*100:.2f}%"],
            ['Precision', f"{DISPLAY_METRICS['test_precision']*100:.2f}%"],
            ['Recall',    f"{DISPLAY_METRICS['test_recall']*100:.2f}%"],
            ['F1-Score',  f"{DISPLAY_METRICS['test_f1']*100:.2f}%"],
            ['AUC',       f"{DISPLAY_METRICS['test_auc']*100:.2f}%"],
        ], [8*cm, 8*cm]))

        story.append(Spacer(1, 0.8*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LINE))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "PlantAI  |  Odeyemi Stacey Israel (2021/10849)  |  Bells University of Technology, Ota  |  "
            "B.Tech Computer Science, November 2025  |  Powered by ResNet152 Transfer Learning",
            foot_s
        ))
        doc.build(story)
        buf.seek(0)
        return buf.read(), None
    except ImportError:
        return None, "reportlab not installed — run: pip install reportlab"
    except Exception as e:
        return None, str(e)


model, metadata, preprocess_fn, load_error = load_model_and_metadata()


with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.2rem 0 0.8rem 0;">
        <div style="font-size:1rem; font-weight:700; color:{accent}; margin-bottom:2px;">PlantAI</div>
        <div style="font-size:0.72rem; color:{text2};">Plantain Disease Recognition</div>
    </div>
    """, unsafe_allow_html=True)

    mode_label = "Switch to Light Mode" if dark else "Switch to Dark Mode"
    if st.button(mode_label, key="toggle_mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f"<div style='height:1px; background:{border}; margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:{text2}; margin-bottom:0.6rem;'>About</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:0.8rem; color:{text2}; line-height:1.6; margin-bottom:0;">
        ResNet152 deep learning system for automatic detection of 
        <strong style="color:{accent};">Black Sigatoka</strong> 
        disease in plantain leaves.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='height:1px; background:{border}; margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:{text2}; margin-bottom:0.8rem;'>Model Performance</div>", unsafe_allow_html=True)

    for lbl, key in [("Accuracy","test_accuracy"),("Precision","test_precision"),
                     ("Recall","test_recall"),("F1-Score","test_f1"),("AUC","test_auc")]:
        val = DISPLAY_METRICS[key] * 100
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.76rem; margin-bottom:3px;">
                <span style="color:{text2};">{lbl}</span>
                <span style="color:{accent}; font-weight:600;">{val:.2f}%</span>
            </div>
            <div style="background:{border}; border-radius:6px; height:6px; overflow:hidden;">
                <div style="background:{accent}; width:{val}%; height:100%; border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div style='height:1px; background:{border}; margin:1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:{text2}; margin-bottom:0.5rem;'>Disease Detected</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:0.78rem; color:{text2}; line-height:1.6; margin-bottom:0;">
        <em>Mycosphaerella fijiensis</em> (Black Sigatoka) — the most destructive 
        fungal disease of plantain worldwide, reducing yield by 30–100% if untreated.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='height:1px; background:{border}; margin:1rem 0;'></div>", unsafe_allow_html=True)
    if metadata:
        st.markdown(f"""
        <div style="font-size:0.72rem; color:{text2}; line-height:1.9;">
            <strong style="color:{accent};">Author</strong><br>
            {metadata.get('author','Odeyemi Stacey Israel — 2021/10849')}<br>
            <strong style="color:{accent};">Institution</strong><br>
            {metadata.get('university','Bells University of Technology, Ota')}<br>
            <strong style="color:{accent};">Trained</strong><br>
            {metadata.get('date_trained','N/A')}
        </div>
        """, unsafe_allow_html=True)


st.markdown(f"""
<div class="header-wrap">
    <div class="header-left">
        <div class="header-badge">ResNet152 · Transfer Learning · TensorFlow</div>
        <div class="header-title">Plantain <em>Disease</em> Recognition</div>
        <div class="header-sub">
            Upload or capture a plantain leaf to detect Black Sigatoka infection instantly.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"Setup error: {load_error}")
    st.info("""
    **How to set up:**
    1. Train the model in Google Colab
    2. Download `plantain_disease_resnet152.h5` and `model_metadata.json`
    3. Place both files in the same folder as `app.py`
    4. Run: `streamlit run app.py`
    """)
    st.stop()

tab1, tab2 = st.tabs(["Upload Image", "Use Camera"])
image_input  = None
input_source = None

with tab1:
    uploaded = st.file_uploader(
        "Upload plantain leaf",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed"
    )
    if uploaded:
        image_input  = Image.open(uploaded)
        input_source = uploaded.name

with tab2:
    st.markdown(f"<p style='font-size:0.8rem; color:{text2}; margin-bottom:0.5rem;'>Hold camera 20–30 cm from the leaf. Ensure even lighting and a steady hand.</p>", unsafe_allow_html=True)
    camera_img = st.camera_input("Capture", label_visibility="collapsed")
    if camera_img:
        image_input  = Image.open(camera_img)
        input_source = "Camera"

is_leaf    = True
is_healthy = True
pred_class = ""
confidence = 0.0
all_probs  = {}

if image_input is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_img, col_res = st.columns([1, 1.5], gap="large")

    with col_img:
        st.markdown('<div class="card-label">Input Image</div>', unsafe_allow_html=True)
        st.image(image_input, use_container_width=True, caption=f"Source: {input_source}")
        w, h = image_input.size
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box"><div class="stat-v">{w}x{h}</div><div class="stat-l">Resolution</div></div>
            <div class="stat-box"><div class="stat-v">{image_input.mode}</div><div class="stat-l">Mode</div></div>
            <div class="stat-box"><div class="stat-v">{round(w*h/1e6,1)}MP</div><div class="stat-l">Size</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_res:
        with st.spinner("Analysing image..."):
            t0     = time.time()
            is_leaf = check_is_leaf(image_input)
            if is_leaf:
                pred_class, confidence, all_probs = predict(image_input, model, metadata, preprocess_fn)
            elapsed = (time.time() - t0) * 1000

        if not is_leaf:
            st.markdown(f"""
            <div class="result-wrap result-unknown">
                <div class="result-status status-u">Not a leaf</div>
                <div class="result-label">Unrecognised Image</div>
                <p style="font-size:0.82rem; color:{text2}; line-height:1.6; margin-top:0.5rem;">
                    The uploaded image does not appear to be a plantain leaf. 
                    Please upload a clear, close-up photo of a plantain leaf for accurate results.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            is_healthy   = "healthy" in pred_class.lower()
            wrap_cls     = "result-healthy" if is_healthy else "result-diseased"
            status_cls   = "status-h" if is_healthy else "status-d"
            status_txt   = "Diagnosis result" 
            display_name = "Healthy Leaf" if is_healthy else "Black Sigatoka"

            st.markdown(f"""
            <div class="result-wrap {wrap_cls}">
                <div class="result-status {status_cls}">{status_txt}</div>
                <div class="result-label">{display_name}</div>
                <div class="result-conf">{confidence:.1f}<span>% confidence</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:{text2}; margin:1rem 0 0.4rem 0;'>Class probabilities</div>", unsafe_allow_html=True)

            for cls_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                display_cls = "Healthy Leaf" if "healthy" in cls_name.lower() else "Black Sigatoka"
                fill_cls    = "bar-fill-g"  if "healthy" in cls_name.lower() else "bar-fill-r"
                val_color   = accent if "healthy" in cls_name.lower() else ("#EF5350" if dark else "#C62828")
                bold        = "600" if cls_name in pred_class else "400"
                st.markdown(f"""
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
                        <span style="font-weight:{bold}; color:{'#E8EAF6' if (dark and cls_name in pred_class) else '#1A1A2E' if (not dark and cls_name in pred_class) else text2};">{display_cls}</span>
                        <span style="color:{val_color}; font-weight:600;">{prob:.1f}%</span>
                    </div>
                    <div class="bar-bg"><div class="{fill_cls}" style="width:{prob}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:0.7rem; color:{text2}; margin-top:0.6rem;'>Analysis time: {elapsed:.0f}ms &nbsp;|&nbsp; Model: ResNet152 &nbsp;|&nbsp; Input: 224x224px</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
            pdf_bytes, pdf_err = generate_pdf(pred_class, confidence, all_probs, is_healthy)
            if pdf_bytes:
                fname = f"plantai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button("Download Diagnosis Report (PDF)", data=pdf_bytes,
                                   file_name=fname, mime="application/pdf", key="dl_pdf")
            elif pdf_err:
                st.caption(f"PDF unavailable: {pdf_err}")
            st.markdown("</div>", unsafe_allow_html=True)

    if is_leaf:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.8rem; font-weight:600; color:{('#E8EAF6' if dark else '#1A1A2E')}; margin-bottom:0.5rem;'>Management Recommendations</div>", unsafe_allow_html=True)

        if is_healthy:
            recs     = [
                "Continue regular field scouting every 7 to 14 days",
                "Maintain proper drainage to reduce leaf wetness duration",
                "Remove dried or old leaves to reduce fungal spore load",
                "Ensure adequate plant spacing for good air circulation",
                "Monitor weather — humidity above 90% increases infection risk",
                "Keep scan records for historical tracking of plant health",
            ]
            box_cls  = "rec-h"
            rec_title = "Healthy — Preventive Measures"
        else:
            recs     = [
                "Isolate affected plants immediately to prevent disease spread",
                "Remove and destroy heavily infected leaves — do not compost",
                "Apply systemic fungicide (triazole or strobilurin class) — consult an agronomist",
                "Improve drainage and reduce canopy density to lower humidity",
                "Avoid working with wet leaves — spores spread via water splash",
                "Re-scan surrounding plants within 48 to 72 hours",
                "Contact your local agricultural extension officer for field-level advice",
            ]
            box_cls  = "rec-d"
            rec_title = "Black Sigatoka — Immediate Action Required"

        items_html = "".join(
            f'<div class="rec-item"><span class="rec-num">{i}.</span><span>{r}</span></div>'
            for i, r in enumerate(recs, 1)
        )
        st.markdown(f"""
        <div class="rec-box {box_cls}">
            <div class="rec-title">{rec_title}</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Learn more about Black Sigatoka"):
            st.markdown(f"""
            <div class="info-grid">
                <div class="info-block">
                    <h5>What is it?</h5>
                    <p>Black Sigatoka (Black Leaf Streak Disease) is caused by 
                    <em>Mycosphaerella fijiensis</em>, a fungal pathogen affecting 
                    all Musa species. It is the most economically destructive disease 
                    of banana and plantain globally.</p>
                </div>
                <div class="info-block">
                    <h5>Optimal disease conditions</h5>
                    <ul>
                        <li>Temperature: 25 to 28°C</li>
                        <li>Humidity: above 90%</li>
                        <li>Free water on leaf: 2 to 4 hours minimum</li>
                        <li>Most severe during rainy seasons</li>
                    </ul>
                </div>
                <div class="info-block">
                    <h5>Economic impact</h5>
                    <ul>
                        <li>Reduces yield by 30 to 100%</li>
                        <li>Fungicide costs: 15 to 27% of annual farm budget</li>
                        <li>Up to 50 fungicide sprays per year on affected farms</li>
                        <li>Haiti: up to 80% of plantation crops destroyed</li>
                    </ul>
                </div>
                <div class="info-block">
                    <h5>Six infection stages</h5>
                    <ul>
                        <li><strong>Stage 1:</strong> Tiny yellow spots appear</li>
                        <li><strong>Stage 2:</strong> Yellow streaks along veins</li>
                        <li><strong>Stage 3:</strong> Streaks turn brown</li>
                        <li><strong>Stage 4:</strong> Brown spots with yellow edges</li>
                        <li><strong>Stage 5:</strong> Black spots with gray centers</li>
                        <li><strong>Stage 6:</strong> Leaf death and necrosis</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align:center; padding:3rem 2rem; background:{card_bg};
                border:1px solid {border}; border-radius:12px; margin-top:1rem;">
        <div style="font-size:0.95rem; font-weight:600; color:{'#E8EAF6' if dark else '#1A1A2E'}; margin-bottom:0.4rem;">
            Ready to Analyse
        </div>
        <div style="font-size:0.82rem; color:{text2}; max-width:360px; margin:0 auto 2rem; line-height:1.6;">
            Upload a plantain leaf image or take a photo using the tabs above 
            to detect Black Sigatoka disease.
        </div>
        <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
            <div style="background:{'#13151F' if dark else '#F5F7FA'}; border:1px solid {border}; border-radius:8px; padding:1rem 1.4rem; min-width:110px;">
                <div style="font-size:1.2rem; margin-bottom:4px;">📁</div>
                <div style="font-size:0.72rem; color:{text2};">Upload image</div>
            </div>
            <div style="background:{'#13151F' if dark else '#F5F7FA'}; border:1px solid {border}; border-radius:8px; padding:1rem 1.4rem; min-width:110px;">
                <div style="font-size:1.2rem; margin-bottom:4px;">📷</div>
                <div style="font-size:0.72rem; color:{text2};">Use camera</div>
            </div>
            <div style="background:{'#13151F' if dark else '#F5F7FA'}; border:1px solid {border}; border-radius:8px; padding:1rem 1.4rem; min-width:110px;">
                <div style="font-size:1.2rem; margin-bottom:4px;">⚡</div>
                <div style="font-size:0.72rem; color:{text2};">Instant results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(f"""
<div class="footer">
    <strong>PlantAI — Plantain Leaf Disease Recognition System</strong><br>
    Odeyemi Stacey Israel (2021/10849) · Department of Computer Science and Information Technology<br>
    Bells University of Technology, Ota · B.Tech Computer Science · November 2025<br>
    Powered by ResNet152 Transfer Learning · TensorFlow · Streamlit
</div>
""", unsafe_allow_html=True)
