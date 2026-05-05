import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import EXPERIMENTS
from models import build_model
from pokemon_data import get_korean, get_types, get_type_color

RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Pokémon Classifier", layout="wide", page_icon="🔴")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background-color: #f2f4f8; }
#MainMenu, footer, header { visibility: hidden; }

.page-title {
    text-align: center; font-size: 2.6rem; font-weight: 700;
    color: #1a1a2e; margin-bottom: 4px;
}
.page-subtitle {
    text-align: center; font-size: 1rem; color: #888; margin-bottom: 32px;
}

/* ── Selectbox: 드롭다운처럼 ── */
[data-baseweb="select"] input { caret-color: transparent !important; cursor: pointer !important; }
[data-baseweb="select"] > div  { cursor: pointer !important; }

/* ── 섹션 헤더 (이미지 업로드 / 분류 결과 동일 스타일) ── */
.section-label {
    font-size: 1rem; font-weight: 600; color: #555;
    margin-bottom: 10px;
}

/* ── 파일 업로더 label → section-label 동일하게 ── */
[data-testid="stFileUploader"] label {
    font-size: 1rem !important; font-weight: 600 !important;
    color: #555 !important; margin-bottom: 10px !important;
}

/* ── 드롭존: 커스텀 디자인 ── */
[data-testid="stFileUploaderDropzone"] {
    height: 380px !important;
    min-height: 380px !important;
    background: white !important;
    border: 2px dashed #d0d5e8 !important;
    border-radius: 20px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    cursor: pointer !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #a5a0ff !important;
}
/* Browse files 버튼: 완전히 레이아웃 밖으로 제거 (DOM 유지로 기능은 유지) */
[data-testid="stFileUploaderDropzone"] button {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    padding: 0 !important; margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0,0,0,0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}
/* 영어 안내문 완전 제거 */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }

/* ::before = 보라 아이콘 박스 */
[data-testid="stFileUploaderDropzone"]::before {
    content: '';
    display: block;
    width: 64px; height: 64px;
    background-color: #ebe8ff;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%236c63ff' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 28px;
    border-radius: 16px;
    margin-bottom: 16px;
    flex-shrink: 0;
    pointer-events: none;
}
/* ::after = 텍스트 두 줄 */
[data-testid="stFileUploaderDropzone"]::after {
    content: '이미지 업로드\A클릭하거나 드래그하여 파일 선택';
    white-space: pre;
    text-align: center;
    font-size: 0.95rem;
    color: #555;
    line-height: 2;
    pointer-events: none;
}

/* ── 이미지 표시 박스 (st.image 컨테이너 직접 스타일링) ── */
[data-testid="stImage"] {
    background: white; border-radius: 20px;
    padding: 24px;
    display: flex; justify-content: center; align-items: center;
    height: 380px; min-height: 380px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    box-sizing: border-box;
}
[data-testid="stImage"] img {
    max-height: 320px;
    object-fit: contain;
    border-radius: 8px;
}

/* ── 결과 패널 ── */
.result-box {
    background: white; border-radius: 20px;
    padding: 24px; height: 380px; min-height: 380px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    box-sizing: border-box;
    overflow-y: auto;
}
.result-placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%; text-align: center;
}
.result-placeholder svg { margin-bottom: 14px; }
.result-placeholder-text { font-size: 0.88rem; color: #bbb; line-height: 1.7; }

/* ── 1위 카드 ── */
.top-card {
    background: linear-gradient(135deg, #6c63ff 0%, #8b5cf6 50%, #7c3aed 100%);
    border-radius: 18px; padding: 22px 24px; color: white; margin-bottom: 12px;
}
.top-card-inner { display: flex; align-items: center; justify-content: space-between; }
.top-card-left  { display: flex; flex-direction: column; gap: 5px; }
.rank-badge {
    display: inline-block; padding: 2px 10px;
    background: rgba(255,255,255,0.22); border-radius: 99px;
    font-size: 0.75rem; font-weight: 700; width: fit-content;
}
.top-name-kr { font-size: 1.4rem; font-weight: 700; }
.top-name-en { font-size: 0.85rem; opacity: 0.75; }
.top-pct     { font-size: 1.7rem; font-weight: 700; }
.top-progress {
    margin-top: 14px; background: rgba(255,255,255,0.2);
    border-radius: 99px; height: 5px;
}
.top-progress-fill { background: white; height: 5px; border-radius: 99px; }

/* ── 2~5위 카드 ── */
.other-card {
    background: white; border: 1px solid #f0f0f5; border-radius: 14px;
    padding: 14px 18px; display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.other-left  { display: flex; align-items: center; gap: 14px; }
.other-rank  { font-size: 0.8rem; font-weight: 600; color: #bbb; min-width: 26px; }
.other-kr    { font-size: 1rem; font-weight: 600; color: #222; }
.other-en    { font-size: 0.78rem; color: #bbb; }
.other-pct   { font-size: 1rem; font-weight: 600; color: #444; }

/* ── 타입 뱃지 ── */
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 99px;
    font-size: 0.7rem; font-weight: 600; color: white; margin-top: 3px;
}

/* ── 버튼 ── */
.stButton > button {
    width: 100%; background: #f5f5f8; border: none; border-radius: 12px;
    padding: 12px; font-size: 0.9rem; color: #555; cursor: pointer;
    margin-top: 12px; font-weight: 500; transition: background 0.15s;
}
.stButton > button:hover { background: #eaeaf0; color: #333; }

[data-testid="stSidebar"] { background: white; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(exp_name):
    exp_dir = RESULTS_DIR / exp_name
    if not (exp_dir / "best_model.pth").exists():
        return None, None
    with open(exp_dir / "class_names.json") as f:
        class_names = json.load(f)
    cfg = next(e for e in EXPERIMENTS if e.name == exp_name)
    cfg.num_classes = len(class_names)
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(exp_dir / "best_model.pth", map_location=DEVICE))
    model.eval()
    return model, class_names

def preprocess(image):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict_top5(model, tensor, class_names):
    probs = F.softmax(model(tensor), dim=1)[0]
    top5_probs, top5_idx = probs.topk(5)
    return [(class_names[i], p.item()) for i, p in zip(top5_idx, top5_probs)]

def badges_html(name):
    return " ".join(
        f'<span class="badge" style="background:{get_type_color(t)};">{t}</span>'
        for t in get_types(name)
    )


# ── Session state ──────────────────────────────────────────────────────────

if "image" not in st.session_state:
    st.session_state.image = None


# ── Sidebar ────────────────────────────────────────────────────────────────

available = [d.name for d in sorted(RESULTS_DIR.iterdir())
             if d.is_dir() and (d / "best_model.pth").exists()] if RESULTS_DIR.exists() else []

LABELS = {
    "exp1_resnet18_scratch":  "Exp 1 — ResNet18 (scratch)",
    "exp2_resnet18_frozen":   "Exp 2 — ResNet18 (frozen)",
    "exp3_resnet50_partial":  "Exp 3 — ResNet50 (partial)",
    "exp4_efficientnet_full": "Exp 4 — EfficientNet-B0 (full)",
}

with st.sidebar:
    st.markdown("### 모델 선택")
    if not available:
        st.warning("학습된 모델이 없습니다.\ntrain.py를 먼저 실행하세요.")
        st.stop()
    selected = st.selectbox("", available,
                            format_func=lambda x: LABELS.get(x, x),
                            label_visibility="collapsed")
    rp = RESULTS_DIR / selected / "results.json"
    if rp.exists():
        with open(rp) as f:
            res = json.load(f)
        st.metric("Test Accuracy",     f"{res['test_acc']*100:.1f}%")
        st.metric("Best Val Accuracy", f"{res['best_val_acc']*100:.1f}%")
    st.markdown("---")
    cmp = RESULTS_DIR / "comparison.png"
    if cmp.exists():
        st.image(str(cmp), caption="실험 비교")


# ── Load model ─────────────────────────────────────────────────────────────

model, class_names = load_model(selected)
if model is None:
    st.error(f"`{selected}` 모델을 찾을 수 없습니다.")
    st.stop()


# ── Page header ────────────────────────────────────────────────────────────

st.markdown('<div class="page-title">Pokémon Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">AI 기반 포켓몬 이미지 분류 시스템</div>', unsafe_allow_html=True)


# ── Layout ─────────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2, gap="large")

# ── Left: 업로드 or 이미지 ─────────────────────────────────────────────────
with col_left:
    if st.session_state.image is None:
        # 파일 업로더 (CSS로 커스텀 디자인 적용)
        uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            st.session_state.image = Image.open(uploaded).convert("RGB")
            st.rerun()
    else:
        # 이미지 표시
        st.markdown('<div class="section-label">이미지 업로드</div>', unsafe_allow_html=True)
        st.image(st.session_state.image, width=300)
        if st.button("다른 포켓몬 분석하기"):
            st.session_state.image = None
            st.rerun()

# ── Right: 결과 ────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">분류 결과</div>', unsafe_allow_html=True)

    if st.session_state.image is None:
        st.markdown("""
        <div class="result-box">
          <div class="result-placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
                 xmlns="http://www.w3.org/2000/svg">
              <circle cx="11" cy="11" r="7" stroke="#d0d0e0" stroke-width="1.8"/>
              <path d="M16.5 16.5L21 21" stroke="#d0d0e0" stroke-width="1.8"
                    stroke-linecap="round"/>
            </svg>
            <div class="result-placeholder-text">
              이미지를 업로드하면<br>분석 결과가 여기에 표시됩니다
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("분석 중..."):
            tensor = preprocess(st.session_state.image)
            preds  = predict_top5(model, tensor, class_names)

        name1, prob1 = preds[0]
        kr1   = get_korean(name1)
        b1    = badges_html(name1)
        pct1  = prob1 * 100
        bar_w = min(int(pct1), 100)

        st.markdown(f"""
        <div class="top-card">
          <div class="top-card-inner">
            <div class="top-card-left">
              <span class="rank-badge">1st</span>
              <span class="top-name-kr">{kr1}</span>
              <span class="top-name-en">{name1}</span>
              <div style="margin-top:3px">{b1}</div>
            </div>
            <span class="top-pct">{pct1:.2f}%</span>
          </div>
          <div class="top-progress">
            <div class="top-progress-fill" style="width:{bar_w}%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        for rank, (name, prob) in enumerate(preds[1:], 2):
            kr     = get_korean(name)
            badges = badges_html(name)
            pct    = prob * 100
            st.markdown(f"""
            <div class="other-card">
              <div class="other-left">
                <span class="other-rank">#{rank}</span>
                <div>
                  <div class="other-kr">{kr}</div>
                  <div class="other-en">{name}</div>
                  <div style="margin-top:3px">{badges}</div>
                </div>
              </div>
              <span class="other-pct">{pct:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
