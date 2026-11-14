"""CorroVision - Streamlit interface (English / Arabic).

Run from the repository root:   streamlit run app/app.py
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from texts import T, TYPES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
APP_MODEL = 'ResNet50 (frozen)'                       # row name of this app's model in the result tables
TEAL, NAVY, RED, GREEN, GREY = '#0f8b8d', '#0b3954', '#c62828', '#2e7d32', '#9aa5b1'

st.set_page_config(page_title='CorroVision', page_icon='⚓', layout='wide')

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    lang = 'ar' if st.radio(T['en']['lang'], ['English', 'العربية'], horizontal=True, key='lang') == 'العربية' else 'en'
    t = T[lang]
    st.markdown(f"### ⚙️ {t['settings']}")
    threshold = st.slider(t['threshold'], 0.05, 0.95, 0.50, 0.05, help=t['threshold_help'], key='threshold')
    use_tta = st.toggle(t['tta'], value=False, help=t['tta_help'], key='tta')
    use_cam = st.toggle(t['cam'], value=True, key='cam')
    st.markdown(f"**{t['model_title']}**")
    st.caption(t['model_text'])
    st.caption(t['disclaimer'])

rtl = lang == 'ar'
st.markdown(f"""
<style>
.block-container {{padding-top: 3.6rem; max-width: 1180px;}}
{'.stApp, .stMarkdown, .stCaption, [data-testid="stSidebar"] {direction: rtl; text-align: right;}' if rtl else ''}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #0e6377 60%, {TEAL} 100%); border-radius: 14px;
        padding: 22px 28px; color: #fff; margin-bottom: 14px;}}
.hero h1 {{color: #fff; font-size: 1.9rem; margin: 0; padding: 0;}}
.hero p {{color: #d6ecee; margin: 4px 0 0; font-size: 1rem;}}
.verdict {{border-radius: 12px; padding: 16px 20px; color: #fff; margin-bottom: 6px;}}
.verdict .big {{font-size: 1.45rem; font-weight: 700;}}
.verdict.bad {{background: linear-gradient(120deg, #b71c1c, #d32f2f);}}
.verdict.ok {{background: linear-gradient(120deg, #1b5e20, #2e7d32);}}
.typecard {{border: 1px solid #d9e0e6; border-radius: 12px; padding: 14px 18px; background: #fff;}}
.typecard .name {{color: {TEAL}; font-weight: 700; font-size: 1.1rem;}}
.typecard p {{margin: 6px 0 0;}}
.kpi {{border: 1px solid #d9e0e6; border-radius: 12px; padding: 14px 16px; background: #fff;}}
.kpi .v {{font-size: 1.7rem; font-weight: 700; color: {TEAL};}}
.kpi .l {{font-size: .85rem; color: #5b6770;}}
</style>
<div class="hero"><h1>⚓ CorroVision</h1><p>{t['tagline']}</p></div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading model...')
def get_predictor():
    from predictor import Predictor
    return Predictor()


def bar_chart(names, values, highlight=None, errors=None, height=None):
    names, values = list(names), np.asarray(values, float)
    colors = [TEAL if (highlight is None or n == highlight) else GREY for n in names]
    fig = go.Figure(go.Bar(
        x=values * 100, y=names, orientation='h', marker_color=colors,
        text=[f'{v * 100:.1f}%' for v in values], textposition='inside', insidetextanchor='start',
        error_x=None if errors is None else dict(type='data', symmetric=False, array=errors[1] * 100,
                                                 arrayminus=errors[0] * 100, color='#44505c', thickness=1.2)))
    fig.update_layout(height=height or (60 + 34 * len(names)), margin=dict(l=0, r=10, t=6, b=6),
                      xaxis=dict(range=[0, 100], ticksuffix='%', gridcolor='#e6eaee'),
                      yaxis=dict(autorange='reversed'), plot_bgcolor='#fff', showlegend=False)
    return fig


tab_analyze, tab_results, tab_about = st.tabs([t['tab_analyze'], t['tab_results'], t['tab_about']])

# ============================================================================= 1. analyze an image
with tab_analyze:
    c_up, c_cam = st.columns([3, 1])
    with c_up:
        uploaded = st.file_uploader(t['upload'], type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], key='uploader')
    with c_cam:
        st.write('')
        shot = st.camera_input(t['camera_label'], key='camera') if st.toggle(f"📷 {t['camera']}", key='use_camera') else None
    source = uploaded or shot
    if source is not None:                       # keep the image when the language or a setting changes
        st.session_state['image_bytes'] = source.getvalue()
    if 'image_bytes' not in st.session_state:
        st.info(t['waiting'])
    else:
        rgb = np.asarray(Image.open(io.BytesIO(st.session_state['image_bytes'])).convert('RGB'))
        res = get_predictor().predict(rgb, threshold=threshold, tta=use_tta, cam=use_cam)

        with st.expander(f"🧪 {t['steps_title']}"):
            for col, key in zip(st.columns(4), ['resize', 'denoise', 'color', 'clahe']):
                col.image(res['steps'][key], caption=t[f'step_{key}'], width='stretch')

        left, right = st.columns([2, 3], gap='large')
        with left:
            st.image(rgb, caption=t['original'], width='stretch')
            if res['gradcam'] is not None:
                st.image(res['gradcam'], caption=t['cam_caption'], width='stretch')
        with right:
            # ---- Stage 1
            st.subheader(t['s1_title'])
            p = res['p_corroded']
            if res['corroded']:
                st.markdown(f'<div class="verdict bad"><div class="big">⚠️ {t["corroded"]}</div>'
                            f'{t["confidence"]}: {p * 100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict ok"><div class="big">✅ {t["healthy"]}</div>'
                            f'{t["confidence"]}: {(1 - p) * 100:.1f}%</div>', unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode='gauge+number', value=p * 100, number=dict(suffix='%', font=dict(size=34)),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=RED if res['corroded'] else GREEN),
                           threshold=dict(line=dict(color='#111', width=3), value=threshold * 100))))
            gauge.update_layout(height=230, margin=dict(l=40, r=40, t=20, b=10))
            st.caption(t['gauge'])
            st.plotly_chart(gauge, width='stretch')

            # ---- Stage 2
            st.subheader(t['s2_title'])
            probs = pd.Series(res['type_probs']).sort_values(ascending=False)
            if not res['corroded']:
                st.info(t['s2_skipped'])
            else:
                name, what, action = TYPES[res['top_type']][lang]
                st.markdown(f'<div class="typecard"><div class="name">{name}</div>'
                            f'<div>{t["type_conf"]}: {probs.iloc[0] * 100:.1f}%</div>'
                            f'<p><b>{t["what"]}</b> {what}</p><p><b>{t["action"]}:</b> {action}</p></div>',
                            unsafe_allow_html=True)
                if probs.iloc[0] - probs.iloc[1] < 0.15:
                    st.warning(t['low_conf'])
                st.caption(t['top3'])
                top3 = probs.head(3)
                st.plotly_chart(bar_chart([TYPES[k][lang][0] for k in top3.index], top3.values),
                                width='stretch')
                with st.expander(t['all_types']):
                    st.plotly_chart(bar_chart([TYPES[k][lang][0] for k in probs.index], probs.values),
                                    width='stretch')

            report = {'stage1': {'label': 'Corroded' if res['corroded'] else 'Healthy',
                                 'probability_corroded': round(p, 4), 'threshold': threshold},
                      'stage2': ({'type': res['top_type'],
                                  'probabilities': {k: round(v, 4) for k, v in probs.items()}}
                                 if res['corroded'] else None),
                      'settings': {'test_time_augmentation': use_tta}, 'model': APP_MODEL}
            st.download_button(t['download'], json.dumps(report, indent=2), 'corrovision_result.json', 'application/json')

# ============================================================================= 2. project results
with tab_results:
    st.subheader(t['res_title'])
    st.caption(t['res_sub'])
    s1 = pd.read_csv(RESULTS / 'stage1_results.csv', index_col=0)
    s2 = pd.read_csv(RESULTS / 'stage2_results.csv', index_col=0)
    e2e = pd.read_csv(RESULTS / 'end_to_end_results.csv', index_col=0)
    rec = pd.read_csv(RESULTS / 'stage2_per_class_recall.csv', index_col=0)

    kpis = [(f"{s1.loc[APP_MODEL, 'Accuracy'] * 100:.1f}%", t['m_s1']), (f"{s1.loc[APP_MODEL, 'ROC-AUC']:.4f}", t['m_auc']),
            (f"{s2.loc[APP_MODEL, 'Accuracy'] * 100:.1f}%", t['m_s2']), (f"{e2e.loc[APP_MODEL, 'Accuracy'] * 100:.1f}%", t['m_e2e'])]
    for col, (v, l) in zip(st.columns(4), kpis):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.caption(t['m_note'])

    levels = {t['lv_e2e']: e2e, t['lv_s1']: s1, t['lv_s2']: s2}
    df = levels[st.radio(t['level'], list(levels), horizontal=True)].sort_values('Accuracy', ascending=False)
    st.markdown(f"**{t['acc_chart']}**")
    st.plotly_chart(bar_chart(df.index, df['Accuracy'], highlight=APP_MODEL,
                              errors=((df['Accuracy'] - df['Acc CI low']).values,
                                      (df['Acc CI high'] - df['Accuracy']).values)), width='stretch')
    st.info(t['why'])

    st.markdown(f"**{t['recall_chart']}**")
    method = st.selectbox(t['method'], list(rec.index), index=list(rec.index).index(APP_MODEL))
    r = rec.loc[method].sort_values(ascending=False)
    st.plotly_chart(bar_chart([TYPES[k][lang][0] for k in r.index], r.values), width='stretch')

    st.markdown(f"### {t['figures']}")
    figs = sorted((RESULTS / 'figures').glob('*.png'))
    if figs:
        pick = st.selectbox(' ', figs, format_func=lambda f: f.stem.replace('_', ' ').title(), label_visibility='collapsed')
        st.image(str(pick), width='stretch')
    with st.expander(f"📋 {t['tables']}"):
        for title, d in (('Stage 1', s1), ('Stage 2', s2), ('End to end', e2e),
                         ('McNemar', pd.read_csv(RESULTS / 'mcnemar.csv'))):
            st.markdown(f'**{title}**'); st.dataframe(d.round(4), width='stretch')

# ============================================================================= 3. about
with tab_about:
    st.markdown(t['about_md'])
