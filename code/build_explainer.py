"""Fills explainer_template.html with the real result tables and figures (no number is typed by hand)."""
import base64, io, json, re
from pathlib import Path
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent; ROOT = HERE.parent; OUT = ROOT / 'outputs'; FIG = OUT / 'figures'
html = (HERE / 'explainer_template.html').read_text(encoding='utf-8')
pct = lambda v: f'{v * 100:.1f}%'


def table(df, cols, fmt):
    h = '<div class="tw"><table class="ltr"><tr>' + ''.join(f'<th>{c}</th>' for c in ['Method'] + [c for c, _ in cols]) + '</tr>'
    for name, r in df.iterrows():
        h += f'<tr><td>{name}</td>' + ''.join(f'<td>{fmt.get(k, "{:.4f}").format(r[k]) if not callable(fmt.get(k)) else fmt[k](r[k])}</td>' for _, k in cols) + '</tr>'
    return h + '</table></div>'


s1 = pd.read_csv(OUT / 'stage1_results.csv', index_col=0)
s2 = pd.read_csv(OUT / 'stage2_results.csv', index_col=0)
e2e = pd.read_csv(OUT / 'end_to_end_results.csv', index_col=0)
rec = pd.read_csv(OUT / 'stage2_per_class_recall.csv', index_col=0)
thr = pd.read_csv(OUT / 'stage1_threshold.csv')
mc = pd.read_csv(OUT / 'mcnemar.csv')
I = '{:d}'
T_S1 = table(s1, [('Accuracy', 'Accuracy'), ('F1 (weighted)', 'F1 (w)'), ('F1 (macro)', 'F1 (macro)'),
                  ('Healthy recall', 'Healthy recall'), ('Corroded recall', 'Corroded recall'),
                  ('Balanced acc.', 'Balanced accuracy'), ('ROC-AUC', 'ROC-AUC'),
                  ('H→H', 'TN (H->H)'), ('H→C', 'FP (H->C)'), ('C→H', 'FN (C->H)'), ('C→C', 'TP (C->C)')],
             {'Healthy recall': pct, 'Corroded recall': pct, 'TN (H->H)': lambda v: str(int(v)), 'FP (H->C)': lambda v: str(int(v)),
              'FN (C->H)': lambda v: str(int(v)), 'TP (C->C)': lambda v: str(int(v))})
ci = lambda df: [f'{pct(a)} – {pct(b)}' for a, b in zip(df['Acc CI low'], df['Acc CI high'])]
s2 = s2.assign(CI=ci(s2)); e2e = e2e.assign(CI=ci(e2e))
T_S2 = table(s2, [('Accuracy', 'Accuracy'), ('95% CI', 'CI'), ('Precision (w)', 'Precision (w)'), ('Recall (w)', 'Recall (w)'),
                  ('F1 (weighted)', 'F1 (w)'), ('F1 (macro)', 'F1 (macro)')], {'CI': str})
T_E2E = table(e2e, [('Accuracy', 'Accuracy'), ('95% CI', 'CI'), ('F1 (weighted)', 'F1 (w)'), ('F1 (macro)', 'F1 (macro)'),
                    ('Stage 1 errors', 'Stage-1 errors'), ('Stage 2 errors', 'Stage-2 errors (after correct Stage 1)')],
              {'CI': str, 'Stage-1 errors': lambda v: str(int(v)), 'Stage-2 errors (after correct Stage 1)': lambda v: str(int(v))})
T_REC = table(rec, [(c, c) for c in rec.columns], {c: (lambda v: f'{v * 100:.0f}') for c in rec.columns})
t = thr.assign(Method=thr['Method'] + ' — ' + thr['Threshold']).set_index('Method')
T_THR = table(t, [('Threshold value', 'Value'), ('Accuracy', 'Accuracy'), ('F1 (macro)', 'F1 (macro)'),
                  ('Healthy recall', 'Healthy recall'), ('Corroded recall', 'Corroded recall')],
              {'Value': '{:.3f}', 'Healthy recall': pct, 'Corroded recall': pct})
h = ''
for comp, g in mc.groupby('Comparison', sort=False):
    h += f'<h4><span class="en">{comp}</span></h4><div class="tw"><table class="ltr"><tr><th>Method A</th><th>Method B</th><th>A right, B wrong</th><th>A wrong, B right</th><th>p (Holm)</th><th>Significant?</th></tr>'
    for _, r in g.iterrows():
        p = r['p (Holm)']; ps = '&lt; 0.0001' if p < 1e-4 else f'{p:.4f}'
        h += f"<tr><td>{r['Method A']}</td><td>{r['Method B']}</td><td>{r['A right, B wrong']}</td><td>{r['A wrong, B right']}</td><td>{ps}</td><td>{'Yes' if r['Significant (0.05)'] else 'No'}</td></tr>"
    h += '</table></div>'
T_MC = h
bb = pd.read_csv(OUT / 'backbone_comparison.csv').set_index('Backbone(s)')
T_BB = table(bb, [('Features', 'Features'), ('Stage 1 val', 'S1 val'), ('Stage 2 val', 'S2 val'), ('Stage 2 val + TTA', 'S2 val TTA'),
                  ('Stage 2 test', 'S2 test'), ('Stage 2 test + TTA', 'S2 test TTA')],
             {'Features': lambda v: str(int(v)), **{k: pct for k in ('S1 val', 'S2 val', 'S2 val TTA', 'S2 test', 'S2 test TTA')}})


def fig(m):
    name, cap = m.group(1), m.group(2)
    p = FIG / f'{name}.png'
    if not p.exists():
        return ''
    im = Image.open(p).convert('RGB')
    if im.width > 1800:
        im = im.resize((1800, int(im.height * 1800 / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=86)
    return f'<img alt="{cap}" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"><div class="cap">{cap}</div>'


html = re.sub(r'\{\{FIG:([a-z0-9_]+)\|([^}]*)\}\}', fig, html)
for k, v in {'T_S1': T_S1, 'T_S2': T_S2, 'T_E2E': T_E2E, 'T_REC': T_REC, 'T_THR': T_THR, 'T_MC': T_MC, 'T_BB': T_BB}.items():
    html = html.replace('{{' + k + '}}', v)
for k in ('READING', 'QA', 'CNN2_INFO'):
    f = HERE / f'explainer_{k.lower()}.html'
    html = html.replace('{{' + k + '}}', f.read_text(encoding='utf-8') if f.exists() else '')
assert '{{' not in html, re.findall(r'\{\{[^}]*\}\}', html)
out = ROOT / 'CorroVision_Technical_Explanation_AR.html'
out.write_text(html, encoding='utf-8')
print('wrote', out, f'{out.stat().st_size / 1e6:.1f} MB')
