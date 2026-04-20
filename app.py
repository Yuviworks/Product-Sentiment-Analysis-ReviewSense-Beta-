import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReviewSense · Sentiment AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "review_text"  not in st.session_state: st.session_state.review_text  = ""
if "history"      not in st.session_state: st.session_state.history      = []
if "last_result"  not in st.session_state: st.session_state.last_result  = None
if "analyzed"     not in st.session_state: st.session_state.analyzed     = False
if "batch_text"   not in st.session_state: st.session_state.batch_text   = ""

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ── Clean text ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def analyze(text):
    cleaned    = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probs      = model.predict_proba(vectorized)[0] if hasattr(model, "predict_proba") else None
    labels     = model.classes_               if hasattr(model, "predict_proba") else None
    feature_names = vectorizer.get_feature_names_out()
    vec_arr    = vectorized.toarray()[0]
    top_indices = np.argsort(vec_arr)[-8:]
    keywords   = [feature_names[i] for i in top_indices if vec_arr[i] > 0]
    return prediction, probs, labels, keywords

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ═══ BASE ═══ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #05080f !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    overflow-x: hidden;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"] { display: none !important; }

[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"]      { gap: 0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #05080f; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,.4); border-radius: 99px; }

/* ═══ ORB BACKGROUND ═══ */
.bg-mesh { position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden; }
.orb { position:absolute;border-radius:50%;filter:blur(100px);animation:drift linear infinite;opacity:.16; }
.orb-1 { width:550px;height:550px;background:#6366f1;top:-180px;left:-100px;animation-duration:24s; }
.orb-2 { width:480px;height:480px;background:#ec4899;bottom:-120px;right:-80px;animation-duration:30s;animation-direction:reverse; }
.orb-3 { width:340px;height:340px;background:#06b6d4;top:45%;left:55%;animation-duration:20s;animation-delay:-10s; }
.orb-4 { width:260px;height:260px;background:#8b5cf6;top:15%;right:25%;animation-duration:36s;animation-delay:-5s; }
@keyframes drift {
    0%   { transform:translate(0,0) scale(1); }
    33%  { transform:translate(40px,-30px) scale(1.05); }
    66%  { transform:translate(-25px,35px) scale(.95); }
    100% { transform:translate(0,0) scale(1); }
}

/* ═══ HERO ═══ */
.hero {
    position:relative;z-index:1;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    min-height:65vh;padding:80px 24px 50px;text-align:center;
}

.hero-title {
    font-family:'Syne',sans-serif;font-size:clamp(46px,8vw,106px);
    font-weight:800;line-height:.95;letter-spacing:-.03em;color:#f1f5f9;
    margin-bottom:20px;animation:fadeUp .8s .1s ease both;
}
.hero-title .glow {
    background:linear-gradient(135deg,#f472b6 0%,#a78bfa 45%,#38bdf8 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    filter:drop-shadow(0 0 28px rgba(236,72,153,.3));
}
.hero-sub {
    font-size:17px;font-weight:300;color:rgba(226,232,240,.42);
    letter-spacing:.02em;max-width:500px;margin-bottom:44px;
    animation:fadeUp .9s .2s ease both;
}
.stat-row{display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-bottom:16px;}
.stat-chip {
    display:flex;align-items:center;gap:8px;
    background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
    backdrop-filter:blur(12px);border-radius:10px;padding:8px 16px;
    font-size:12px;color:rgba(226,232,240,.55);
}
.stat-chip strong{color:#f9a8d4;font-size:14px;font-weight:600;}

/* ═══ GLASS PANEL ═══ */
.glass {
    background:rgba(255,255,255,.04);
    backdrop-filter:blur(24px) saturate(180%);
    -webkit-backdrop-filter:blur(24px) saturate(180%);
    border:1px solid rgba(255,255,255,.08);
    border-radius:20px;padding:28px;
    box-shadow:0 8px 48px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.06);
    animation:fadeUp .6s ease both;
}
.glass-sm {
    background:rgba(255,255,255,.04);
    backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.08);
    border-radius:14px;padding:18px 20px;
    box-shadow:0 4px 24px rgba(0,0,0,.25);
}

/* ═══ SECTION LABEL ═══ */
.section-label {
    font-family:'Syne',sans-serif;font-size:11px;font-weight:700;
    letter-spacing:.18em;text-transform:uppercase;
    color:rgba(249,168,212,.5);margin-bottom:14px;
}

/* ═══ BUTTONS ═══ */
.stButton > button {
    font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:500!important;
    letter-spacing:.08em!important;border-radius:12px!important;
    transition:all .25s cubic-bezier(.23,1,.32,1)!important;cursor:pointer!important;
    background:rgba(255,255,255,.06)!important;color:#cbd5e1!important;
    border:1px solid rgba(255,255,255,.1)!important;padding:12px 20px!important;
}
.stButton > button:hover {
    background:rgba(255,255,255,.1)!important;border-color:rgba(255,255,255,.18)!important;
    transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(0,0,0,.25)!important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#ec4899 0%,#8b5cf6 100%)!important;
    color:#fff!important;border:none!important;padding:15px 28px!important;
    box-shadow:0 4px 24px rgba(236,72,153,.35)!important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow:0 8px 40px rgba(236,72,153,.55)!important;
    transform:translateY(-3px)!important;filter:brightness(1.1)!important;
}

/* ═══ TEXT AREA ═══ */
.stTextArea textarea {
    background:rgba(255,255,255,.05)!important;
    border:1.5px solid rgba(255,255,255,.12)!important;
    border-radius:18px!important;
    color:#f1f5f9!important;
    font-family:'Inter',sans-serif!important;
    font-size:18px!important;
    font-weight:300!important;
    line-height:1.75!important;
    padding:24px 26px!important;
    backdrop-filter:blur(16px);
    transition:all .35s ease!important;
    caret-color:#f472b6;
    box-shadow:0 4px 32px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.06)!important;
    resize:vertical!important;
}
.stTextArea textarea:focus {
    border-color:rgba(236,72,153,.55)!important;
    box-shadow:0 0 0 4px rgba(236,72,153,.12),0 12px 48px rgba(0,0,0,.35)!important;
    background:rgba(255,255,255,.08)!important;
}
.stTextArea textarea::placeholder{
    color:rgba(226,232,240,.2)!important;
    font-size:17px!important;
    font-style:italic!important;
}
.stTextArea label{
    font-family:'Syne',sans-serif!important;
    font-size:20px!important;
    font-weight:700!important;
    letter-spacing:-.01em!important;
    text-transform:none!important;
    color:#f1f5f9!important;
    margin-bottom:12px!important;
}
/* ═══ INPUT SECTION HEADING ═══ */
.input-heading{font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#f1f5f9;letter-spacing:-.01em;margin-bottom:4px;}
.input-subheading{font-size:14px;font-weight:300;color:rgba(226,232,240,.38);margin-bottom:18px;letter-spacing:.01em;}

/* ═══ TABS ═══ */
[data-baseweb="tab-list"] {
    background:rgba(255,255,255,.04)!important;
    border-radius:12px!important;border:1px solid rgba(255,255,255,.08)!important;
    gap:4px!important;padding:4px!important;
}
[data-baseweb="tab"] {
    font-family:'Inter',sans-serif!important;font-size:13px!important;font-weight:500!important;
    color:rgba(226,232,240,.4)!important;border-radius:9px!important;
    padding:8px 20px!important;transition:all .2s!important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background:rgba(236,72,153,.15)!important;
    color:#f9a8d4!important;
    box-shadow:0 2px 12px rgba(236,72,153,.2)!important;
}
[data-baseweb="tab-highlight"]{display:none!important;}
[data-baseweb="tab-border"]{display:none!important;}

/* ═══ RESULT CARDS ═══ */
.result-card {
    border-radius:18px;padding:28px 32px;margin-bottom:20px;
    backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.1);
    animation:cardIn .5s ease both;
    box-shadow:0 12px 48px rgba(0,0,0,.35);
}
.result-positive { background:rgba(34,197,94,.08);border-color:rgba(34,197,94,.2); }
.result-negative { background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.2); }
.result-neutral  { background:rgba(148,163,184,.08);border-color:rgba(148,163,184,.2); }

.result-emoji { font-size:52px;margin-bottom:12px; }
.result-label {
    font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
    letter-spacing:-.02em;margin-bottom:8px;
}
.result-label-pos{color:#4ade80;}
.result-label-neg{color:#f87171;}
.result-label-neu{color:#94a3b8;}
.result-desc{font-size:14px;color:rgba(226,232,240,.5);font-weight:300;}

/* ═══ CONFIDENCE BAR ═══ */
.conf-wrap{margin-top:20px;}
.conf-label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:rgba(226,232,240,.35);margin-bottom:8px;}
.conf-bar-bg{height:6px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden;margin-bottom:6px;}
.conf-bar-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.23,1,.32,1);}
.conf-bar-pos{background:linear-gradient(90deg,#22c55e,#86efac);box-shadow:0 0 8px rgba(34,197,94,.5);}
.conf-bar-neg{background:linear-gradient(90deg,#ef4444,#fca5a5);box-shadow:0 0 8px rgba(239,68,68,.5);}
.conf-bar-neu{background:linear-gradient(90deg,#64748b,#94a3b8);box-shadow:0 0 8px rgba(148,163,184,.4);}
.conf-pct{font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#f1f5f9;}

/* ═══ PROB BARS ═══ */
.prob-grid{display:flex;flex-direction:column;gap:10px;margin-top:16px;}
.prob-row{display:flex;align-items:center;gap:12px;}
.prob-name{font-size:12px;font-weight:500;letter-spacing:.05em;text-transform:capitalize;width:72px;color:rgba(226,232,240,.6);}
.prob-bar-bg{flex:1;height:8px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;}
.prob-bar-inner{height:100%;border-radius:99px;}
.prob-pct{font-size:12px;font-weight:600;color:#f1f5f9;width:40px;text-align:right;}

/* ═══ KEYWORD CHIPS ═══ */
.kw-wrap{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;}
.kw-chip {
    background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.25);
    border-radius:8px;padding:5px 14px;font-size:12px;color:#c4b5fd;
    transition:all .2s;cursor:default;
}
.kw-chip:hover{background:rgba(139,92,246,.2);transform:translateY(-1px);}

/* ═══ HISTORY ITEM ═══ */
.hist-item {
    display:flex;align-items:flex-start;gap:14px;
    background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
    border-radius:14px;padding:16px 18px;margin-bottom:10px;
    transition:all .25s;animation:cardIn .4s ease both;
}
.hist-item:hover{background:rgba(255,255,255,.07);border-color:rgba(255,255,255,.12);}
.hist-emoji{font-size:22px;flex-shrink:0;margin-top:1px;}
.hist-body{flex:1;min-width:0;}
.hist-text{font-size:13px;color:rgba(226,232,240,.65);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hist-meta{font-size:11px;color:rgba(226,232,240,.3);margin-top:4px;letter-spacing:.04em;}
.hist-badge {
    font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
    border-radius:6px;padding:3px 10px;flex-shrink:0;align-self:center;
}
.hist-badge-pos{background:rgba(34,197,94,.12);color:#4ade80;border:1px solid rgba(34,197,94,.2);}
.hist-badge-neg{background:rgba(239,68,68,.12);color:#f87171;border:1px solid rgba(239,68,68,.2);}
.hist-badge-neu{background:rgba(148,163,184,.12);color:#94a3b8;border:1px solid rgba(148,163,184,.2);}

/* ═══ EXAMPLES ═══ */
.ex-pill-wrap{display:flex;gap:8px;flex-wrap:wrap;}
.ex-pill {
    background:rgba(99,179,237,.08);border:1px solid rgba(99,179,237,.2);
    border-radius:99px;padding:6px 16px;font-size:12px;color:#93c5fd;cursor:pointer;
    transition:all .2s;
}
.ex-pill:hover{background:rgba(99,179,237,.18);transform:translateY(-1px);}

/* ═══ BATCH ═══ */
.batch-result-row{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid rgba(255,255,255,.06);}
.batch-result-row:last-child{border-bottom:none;}
.batch-text-preview{flex:1;font-size:13px;color:rgba(226,232,240,.6);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.batch-badge{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;border-radius:6px;padding:3px 10px;flex-shrink:0;}

/* ═══ WARN ═══ */
.warn-box{display:flex;align-items:center;gap:12px;background:rgba(234,179,8,.07);border:1px solid rgba(234,179,8,.2);border-radius:14px;padding:14px 20px;font-size:13px;color:rgba(253,224,71,.75);backdrop-filter:blur(12px);}

/* ═══ FOOTER ═══ */
.site-footer{text-align:center;padding:40px 24px;border-top:1px solid rgba(255,255,255,.05);color:rgba(226,232,240,.15);font-size:11px;letter-spacing:.1em;text-transform:uppercase;}

/* ═══ KEYFRAMES ═══ */
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
@keyframes cardIn{from{opacity:0;transform:translateY(16px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}

[data-testid="stMarkdownContainer"] p{color:inherit!important;}
div.stSpinner > div{border-top-color:#ec4899!important;}
</style>
""", unsafe_allow_html=True)

# ── Background orbs ───────────────────────────────────────────────────────────
st.markdown("""
<div class="bg-mesh">
  <div class="orb orb-1"></div><div class="orb orb-2"></div>
  <div class="orb orb-3"></div><div class="orb orb-4"></div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Review<span class="glow">Sense</span></div>
  <div class="hero-sub">Paste any product review and instantly understand the emotion behind every word.</div>
  <div class="stat-row">
    <div class="stat-chip">⚡ <strong>Instant</strong>&nbsp;results</div>
    <div class="stat-chip">📊 <strong>Confidence</strong>&nbsp;scoring</div>
    <div class="stat-chip">🔑 <strong>Keyword</strong>&nbsp;extraction</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
_, main, _ = st.columns([0.15, 3, 0.15])
with main:

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_single, tab_batch, tab_history = st.tabs(["✦ Single Review", "📋 Batch Analysis", "🕒 History"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SINGLE REVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # Examples
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">💡 Try an example</div>', unsafe_allow_html=True)
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        if ecol1.button("😊 Positive", use_container_width=True):
            st.session_state.review_text = "Absolutely amazing product! Works perfectly and exceeded all my expectations."
        if ecol2.button("😠 Negative", use_container_width=True):
            st.session_state.review_text = "Terrible quality, broke after two days. Complete waste of money."
        if ecol3.button("😐 Neutral",  use_container_width=True):
            st.session_state.review_text = "It's okay, does what it says but nothing particularly special about it."
        if ecol4.button("🎲 Random",   use_container_width=True):
            import random
            samples = [
                "Incredible experience from start to finish. Highly recommend!",
                "The product arrived damaged and customer service was unhelpful.",
                "Average product. Works as described but feels cheap.",
                "Best purchase I've made this year. Absolutely love it!",
                "Would not buy again. The quality is far below expectations.",
            ]
            st.session_state.review_text = random.choice(samples)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Text input + controls
        st.markdown("""
        <div class="input-heading">✍️ Write or paste your review</div>
        <div class="input-subheading">Tell us what the customer said — we'll decode the emotion instantly.</div>
        """, unsafe_allow_html=True)

        left_col, right_col = st.columns([3, 1])
        with left_col:
            review = st.text_area(
                "",
                value=st.session_state.review_text,
                height=240,
                placeholder='"Absolutely loved this product, it exceeded every expectation…"',
                label_visibility="collapsed",
            )
            st.session_state.review_text = review
        with right_col:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            analyze_btn = st.button("🚀  Analyze", use_container_width=True, type="primary")
            clear_btn   = st.button("✕  Clear",    use_container_width=True)
            if clear_btn:
                st.session_state.review_text = ""
                st.session_state.analyzed    = False
                st.session_state.last_result = None
                st.rerun()

            # Live char count
            char_count = len(review)
            word_count = len(review.split()) if review.strip() else 0
            st.markdown(f"""
            <div class="glass-sm" style="margin-top:12px;text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#f9a8d4;">{word_count}</div>
              <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:rgba(226,232,240,.3);">words</div>
              <div style="margin-top:8px;font-family:'Syne',sans-serif;font-size:16px;font-weight:600;color:rgba(226,232,240,.5);">{char_count}</div>
              <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:rgba(226,232,240,.25);">chars</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Run analysis ──────────────────────────────────────────────────────
        if analyze_btn:
            if not review.strip():
                st.markdown('<div class="warn-box">⚠️ &nbsp; Please enter a review before analyzing.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Reading between the lines…"):
                    prediction, probs, labels, keywords = analyze(review)
                st.session_state.last_result = (prediction, probs, labels, keywords, review)
                st.session_state.analyzed    = True
                # Save to history
                st.session_state.history.insert(0, {
                    "text": review, "prediction": prediction,
                    "confidence": float(max(probs)) if probs is not None else None,
                })

        # ── Results ───────────────────────────────────────────────────────────
        if st.session_state.analyzed and st.session_state.last_result:
            prediction, probs, labels, keywords, _ = st.session_state.last_result

            emoji_map  = {"positive": "😊", "negative": "😠", "neutral": "😐"}
            cls_map    = {"positive": "result-positive", "negative": "result-negative", "neutral": "result-neutral"}
            label_cls  = {"positive": "result-label-pos", "negative": "result-label-neg", "neutral": "result-label-neu"}
            bar_cls    = {"positive": "conf-bar-pos", "negative": "conf-bar-neg", "neutral": "conf-bar-neu"}
            desc_map   = {
                "positive": "This review expresses a positive sentiment — the customer is satisfied.",
                "negative": "This review expresses a negative sentiment — the customer is dissatisfied.",
                "neutral":  "This review is neutral — neither clearly positive nor negative.",
            }
            prob_colors = {"positive":"#22c55e", "negative":"#ef4444", "neutral":"#64748b"}

            emoji     = emoji_map.get(prediction, "🤔")
            conf      = float(max(probs)) * 100 if probs is not None else 0
            conf_bar  = bar_cls.get(prediction, "conf-bar-neu")

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            res_col, meta_col = st.columns([1.4, 1])

            with res_col:
                st.markdown(f"""
                <div class="result-card {cls_map.get(prediction,'result-neutral')}">
                  <div class="result-emoji">{emoji}</div>
                  <div class="result-label {label_cls.get(prediction,'result-label-neu')}">{prediction.capitalize()}</div>
                  <div class="result-desc">{desc_map.get(prediction,'')}</div>
                  <div class="conf-wrap">
                    <div class="conf-label">Confidence</div>
                    <div style="display:flex;align-items:center;gap:14px;">
                      <div class="conf-bar-bg" style="flex:1;">
                        <div class="conf-bar-fill {conf_bar}" style="width:{conf:.1f}%"></div>
                      </div>
                      <div class="conf-pct">{conf:.1f}%</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with meta_col:
                # Probability breakdown
                if probs is not None:
                    st.markdown('<div class="glass" style="margin-bottom:14px;">', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">📊 Probability Breakdown</div>', unsafe_allow_html=True)
                    rows_html = ""
                    for lbl, pval in sorted(zip(labels, probs), key=lambda x: -x[1]):
                        color = prob_colors.get(lbl, "#94a3b8")
                        rows_html += f"""
                        <div class="prob-row">
                          <div class="prob-name">{lbl.capitalize()}</div>
                          <div class="prob-bar-bg">
                            <div class="prob-bar-inner" style="width:{pval*100:.1f}%;background:{color};box-shadow:0 0 6px {color}88;"></div>
                          </div>
                          <div class="prob-pct">{pval*100:.0f}%</div>
                        </div>"""
                    st.markdown(f'<div class="prob-grid">{rows_html}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Keywords
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">🔑 Key Signals</div>', unsafe_allow_html=True)
                if keywords:
                    chips = "".join(f'<span class="kw-chip">{k}</span>' for k in keywords)
                    st.markdown(f'<div class="kw-wrap">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:13px;color:rgba(226,232,240,.3);">No strong keywords detected.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — BATCH ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Batch Review Analyzer</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;color:rgba(226,232,240,.4);margin-bottom:16px;">Enter one review per line. Each line is analyzed independently.</div>', unsafe_allow_html=True)

        batch_input = st.text_area(
            "Paste reviews (one per line)",
            height=200,
            placeholder="Great product, love it!\nTerrible experience, never again.\nIt's alright, nothing special.",
            label_visibility="collapsed",
        )

        batch_btn = st.button("⚡  Analyze All", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if batch_btn and batch_input.strip():
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            with st.spinner(f"Analyzing {len(lines)} reviews…"):
                batch_results = []
                counts = {"positive": 0, "negative": 0, "neutral": 0}
                for line in lines:
                    pred, probs_, labels_, kws_ = analyze(line)
                    conf_ = float(max(probs_)) * 100 if probs_ is not None else 0
                    batch_results.append((line, pred, conf_))
                    counts[pred] = counts.get(pred, 0) + 1

            # Summary strip
            total = len(batch_results)
            emoji_map2 = {"positive":"😊","negative":"😠","neutral":"😐"}
            color_map2 = {"positive":"#22c55e","negative":"#ef4444","neutral":"#94a3b8"}
            badge_cls2 = {"positive":"hist-badge-pos","negative":"hist-badge-neg","neutral":"hist-badge-neu"}

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            for col, sentiment in zip([s1, s2, s3], ["positive","negative","neutral"]):
                pct = counts[sentiment] / total * 100 if total else 0
                col.markdown(f"""
                <div class="glass-sm" style="text-align:center;">
                  <div style="font-size:28px;">{emoji_map2[sentiment]}</div>
                  <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{color_map2[sentiment]};">{counts[sentiment]}</div>
                  <div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:rgba(226,232,240,.35);">{sentiment} · {pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Matplotlib donut
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='none')
            sizes  = [counts["positive"], counts["negative"], counts["neutral"]]
            colors = ["#22c55e", "#ef4444", "#94a3b8"]
            wedges, _ = ax.pie(
                sizes, colors=colors, startangle=90,
                wedgeprops=dict(width=0.55, edgecolor='#05080f', linewidth=3)
            )
            ax.set_facecolor('none')
            legend = ax.legend(
                wedges, ["Positive","Negative","Neutral"],
                loc="center left", bbox_to_anchor=(1,0.5),
                frameon=False, labelcolor='white', fontsize=9
            )
            fig.patch.set_alpha(0.0)
            st.pyplot(fig, use_container_width=False)

            # Individual results
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Individual Results</div>', unsafe_allow_html=True)
            for text_, pred_, conf_ in batch_results:
                st.markdown(f"""
                <div class="batch-result-row">
                  <div style="font-size:18px;">{emoji_map2.get(pred_,"🤔")}</div>
                  <div class="batch-text-preview">{text_}</div>
                  <span class="batch-badge {badge_cls2.get(pred_,'hist-badge-neu')}">{pred_}</span>
                  <span style="font-size:11px;color:rgba(226,232,240,.3);width:44px;text-align:right;">{conf_:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="text-align:center;padding:60px 0;color:rgba(226,232,240,.2);font-family:\'Syne\',sans-serif;font-size:20px;font-weight:600;">🕒 No analyses yet</div>', unsafe_allow_html=True)
        else:
            badge_cls3  = {"positive":"hist-badge-pos","negative":"hist-badge-neg","neutral":"hist-badge-neu"}
            emoji_map3  = {"positive":"😊","negative":"😠","neutral":"😐"}
            clear_hist  = st.button("🗑 Clear History", use_container_width=False)
            if clear_hist:
                st.session_state.history = []
                st.rerun()
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            for idx, item in enumerate(st.session_state.history[:20]):
                pred  = item["prediction"]
                conf  = f"{item['confidence']*100:.0f}% confidence" if item["confidence"] else ""
                st.markdown(f"""
                <div class="hist-item">
                  <div class="hist-emoji">{emoji_map3.get(pred,'🤔')}</div>
                  <div class="hist-body">
                    <div class="hist-text">"{item['text']}"</div>
                    <div class="hist-meta">#{len(st.session_state.history)-idx} &nbsp;·&nbsp; {conf}</div>
                  </div>
                  <span class="hist-badge {badge_cls3.get(pred,'hist-badge-neu')}">{pred}</span>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='height:60px'></div>
<div class="site-footer">ReviewSense &nbsp;·&nbsp; NLP Sentiment Engine &nbsp;·&nbsp; Built with Streamlit</div>
""", unsafe_allow_html=True)
