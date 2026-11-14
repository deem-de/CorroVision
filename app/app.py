"""CorroVision demo app.

Upload an image of a marine structure:
  Stage 1 decides Healthy / Corroded, and if corroded,
  Stage 2 names the corrosion type (one of eight).

Model: frozen ImageNet ResNet50 + the two small heads trained in this project
(test accuracy: Stage 1 99.56%, Stage 2 74.06%).

Run:  python app.py            (then open the printed link)
      python app.py --share    (also prints a public link; use this on Google Colab)
"""
import json
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from tensorflow import keras

HERE = Path(__file__).resolve().parent
IMG_SIZE = 224
CLASSES = json.loads((HERE / 'models' / 'labels.json').read_text())['stage2_classes']

# What each corrosion type means, shown under the prediction.
ABOUT = {
    'Crevice': 'Localised attack inside narrow gaps (under gaskets, washers, lap joints) where seawater stagnates.',
    'Erosion': 'Metal loss sped up by fast or turbulent flow, often at bends, elbows and pump parts.',
    'Galvanic': 'One metal corrodes faster because it touches a different, more noble metal in seawater.',
    'MIC': 'Microbiologically influenced corrosion: bacteria and biofilms create aggressive local conditions.',
    'Pitting': 'Small, deep holes in the surface. Hard to see and dangerous because they can perforate the wall.',
    'Stress': 'Cracks that grow under tensile stress in a corrosive environment (stress corrosion cracking).',
    'Under Insulation': 'Corrosion hidden under insulation or coatings where moisture has been trapped.',
    'Uniform': 'Even metal loss over a large area. The most common and most predictable form.',
}

# ----------------------------------------------------------------------------- models (loaded once)
backbone = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg',
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
head1 = keras.models.load_model(HERE / 'models' / 'head_stage1.keras')
head2 = keras.models.load_model(HERE / 'models' / 'head_stage2.keras')


# ----------------------------------------------------------------------------- same preprocessing as training
def preprocess(rgb):
    """Resize -> bilateral filter -> Gray-World colour correction -> CLAHE (identical to the training pipeline)."""
    x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = cv2.bilateralFilter(x, 5, 20, 3)
    f = x.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    x = np.clip(f * np.clip(means.mean() / means, 0.5, 3.0), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def card(kind, title, line):
    return f'<div class="verdict {kind}"><div class="v-title">{title}</div><div class="v-line">{line}</div></div>'


def predict(image):
    if image is None:
        return card('idle', 'No image yet', 'Upload a photo of a structure, then press Inspect.'), None, None, ''
    rgb = np.asarray(image.convert('RGB') if hasattr(image, 'convert') else image)[..., :3].astype(np.uint8)
    proc = preprocess(rgb)
    feat = backbone.predict_on_batch(keras.applications.resnet50.preprocess_input(proc[None].astype(np.float32)))
    p_cor = float(head1.predict_on_batch(feat).reshape(-1)[0])
    if p_cor < 0.5:                                         # Stage 1 says Healthy: the pipeline stops here
        return (card('ok', 'Healthy', f'No corrosion detected &nbsp;·&nbsp; confidence {(1 - p_cor) * 100:.1f}%'),
                None, proc, 'Stage 2 was skipped because Stage 1 classified the structure as healthy.')
    p_type = head2.predict_on_batch(feat).reshape(-1)
    top = CLASSES[int(p_type.argmax())]
    return (card('bad', 'Corroded', f'Corrosion detected &nbsp;·&nbsp; confidence {p_cor * 100:.1f}%'),
            {c: float(p) for c, p in zip(CLASSES, p_type)}, proc,
            f'**Most likely type: {top} corrosion.** {ABOUT[top]}')


# ----------------------------------------------------------------------------- interface
CSS = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important;}
#hero {background: linear-gradient(135deg, #062a4f 0%, #0b5a8a 55%, #14879b 100%); border-radius: 18px;
       padding: 30px 34px; color: #fff; margin-bottom: 6px;}
#hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 4px; letter-spacing: .3px;}
#hero p {color: #d9ecf7; margin: 0; font-size: 1.02rem;}
#hero .chips {margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap;}
#hero .chip {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
             padding: 5px 14px; font-size: .88rem; color: #fff;}
.verdict {border-radius: 14px; padding: 18px 22px; border: 1px solid transparent;}
.verdict .v-title {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.verdict .v-line {font-size: 1rem; margin-top: 4px; opacity: .92;}
.verdict.ok {background: #e6f6ec; border-color: #9bd4b0; color: #0f5a2e;}
.verdict.bad {background: #fdebe6; border-color: #f0b3a2; color: #8a2a12;}
.verdict.idle {background: #eef2f6; border-color: #d5dde6; color: #3d4a5a;}
#inspect {background: #0b5a8a !important; color: #fff !important; font-weight: 600; border: none !important;}
#inspect:hover {background: #094a73 !important;}
footer {display: none !important;}
"""

HERO = """
<div id="hero">
  <h1>CorroVision</h1>
  <p>AI inspection of marine structures: detects corrosion and names its type.</p>
  <div class="chips">
    <span class="chip">Stage 1 · Healthy vs. Corroded · 99.6% test accuracy</span>
    <span class="chip">Stage 2 · 8 corrosion types · 74.1% test accuracy</span>
    <span class="chip">ResNet50 transfer learning</span>
  </div>
</div>
"""

examples_dir = HERE / 'examples'
examples = sorted(str(p) for p in examples_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}) \
    if examples_dir.exists() else []

with gr.Blocks(title='CorroVision') as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            inp = gr.Image(type='pil', label='Structure image', height=380, sources=['upload', 'clipboard'])
            btn = gr.Button('Inspect', elem_id='inspect', size='lg')
            if examples:
                gr.Examples(examples=examples, inputs=inp, label='Try an example')
        with gr.Column(scale=5):
            verdict = gr.HTML(card('idle', 'No image yet', 'Upload a photo of a structure, then press Inspect.'))
            types = gr.Label(num_top_classes=8, label='Stage 2 · corrosion type probabilities')
            note = gr.Markdown()
    with gr.Accordion('What the model sees (after preprocessing)', open=False):
        proc = gr.Image(label='224 x 224 · denoised · colour corrected · contrast enhanced', height=260)
    with gr.Accordion('How it works and limits', open=False):
        gr.Markdown(
            'The image is resized to 224 x 224, denoised with a bilateral filter, colour corrected with the '
            'Gray-World method and contrast enhanced with CLAHE. A ResNet50 pretrained on ImageNet turns it into '
            '2,048 features. **Stage 1** decides Healthy or Corroded. If corroded, **Stage 2** scores the eight '
            'corrosion types.\n\n'
            '**Limits.** The models were trained on the Marine Corrosion Dataset from Kaggle. Stage 2 is right '
            'about 3 times out of 4 on that dataset, and results on real underwater footage may be lower. '
            'This is a course project demo, not an inspection tool.')
    btn.click(predict, inp, [verdict, types, proc, note])
    inp.upload(predict, inp, [verdict, types, proc, note])

if __name__ == '__main__':
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue='blue', neutral_hue='slate'), share='--share' in sys.argv)
