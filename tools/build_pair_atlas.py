"""Build a private, original-pixel atlas from the intersection of training splits.

This is a qualitative reading aid, not a benchmark or manual identity annotation.
All images and the identity manifest must remain outside Git.
"""
import argparse
from collections import defaultdict
import hashlib
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from track6.data import decode_image, sha256, write_json
from track6.protocol import load_split


def font(size):
    for candidate in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def threshold_geometry(pixels, threshold):
    """Describe bright pixels; this does not estimate physical ship geometry."""
    y, x = np.where(pixels.max(axis=2) > threshold)
    result = {'threshold': threshold, 'pixels': len(x), 'fraction': len(x) / (256 * 256),
              'bbox_width': 0, 'bbox_height': 0, 'axis_degrees_mod180': None,
              'moment_anisotropy': None}
    if not len(x):
        return result
    result.update(bbox_width=int(x.max() - x.min() + 1),
                  bbox_height=int(y.max() - y.min() + 1))
    if len(x) > 1:
        points = np.column_stack((x, y)).astype(np.float64)
        points -= points.mean(axis=0)
        values, vectors = np.linalg.eigh(points.T @ points / len(points))
        if values.sum() > 0:
            anisotropy = float((values[-1] - values[0]) / values.sum())
            result['moment_anisotropy'] = anisotropy
            # A circular/isotropic mask has no defined principal direction.
            if anisotropy > 1e-12:
                v = vectors[:, -1]
                result['axis_degrees_mod180'] = float(np.degrees(np.arctan2(v[1], v[0])) % 180)
    return result


def build_atlas(source, split_paths, output, seed=20261002, per_group=6, census=False):
    if per_group < 1 or len(split_paths) < 2:
        raise ValueError('Use at least two splits and a positive sample count')
    loaded = [load_split(source, p) for p in split_paths]
    store, records, _ = loaded[0]
    allowed = set.intersection(*(set(s['train_images']) for _, _, s in loaded))
    validation_ids = set.union(*(set(s['validation_identities']) for _, _, s in loaded))
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.image_id in allowed:
            assert r.ship_id not in validation_ids and r.role == 'train'
            grouped[r.ship_id][r.modality].append(r)
    pools = {
        'one_optical_one_sar': sorted(k for k, v in grouped.items()
                                     if len(v['optical']) == len(v['sar']) == 1),
        'multiple_optical': sorted(k for k, v in grouped.items()
                                   if len(v['optical']) >= 2 and len(v['sar']) >= 1),
    }
    census_summary = {}
    if census:
        for group, pool in pools.items():
            fractions = [float((decode_image(store.read(r.image_path)).max(axis=2) > 8).mean())
                         for ship in pool for r in grouped[ship]['optical']]
            census_summary[group] = {
                'identities': len(pool), 'optical_images': len(fractions),
                'threshold8_fraction_lt_0_3': sum(v < .3 for v in fractions),
                'threshold8_fraction_ge_0_9': sum(v >= .9 for v in fractions),
                'threshold8_fraction_median': float(np.median(fractions)),
                'scope': 'all optical images in this eligible common-training stratum',
            }
    # Hash ordering makes the exact sample independent of PRNG/library versions.
    # Sample sizes deliberately overrepresent multiple-view identities.
    def order(ship):
        # Separate this order from make_split's seed:component hash, including
        # the different treatment of non-canonical members of near-duplicate groups.
        return hashlib.sha256(f'pair-atlas-v1:{seed}:{ship}'.encode('utf-8')).hexdigest()
    chosen = []
    for group, pool in pools.items():
        if len(pool) < per_group:
            raise ValueError(f'Insufficient examples in {group}')
        chosen.extend((group, ship) for ship in sorted(pool, key=order)[:per_group])
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / 'images'
    image_dir.mkdir(exist_ok=True)
    private, snippets, canvas_rows = [], [], []
    for index, (group, ship) in enumerate(chosen, 1):
        case = f'P{index:02}'
        modal = {m: sorted(v, key=lambda r: r.image_id) for m, v in grouped[ship].items()}
        slots = [('Optical 1', modal['optical'][0]), ('SAR 1', modal['sar'][0]),
                 ('Optical 2', modal['optical'][1] if len(modal['optical']) > 1 else None),
                 ('SAR 2', modal['sar'][1] if len(modal['sar']) > 1 else None)]
        row, pictures, details = [], [], []
        for col, (label, record) in enumerate(slots):
            if record is None:
                row.append((label, None))
                continue
            pixels = decode_image(store.read(record.image_path))
            destination = image_dir / f'{case}-{col + 1}.png'
            Image.fromarray(pixels).save(destination)
            with Image.open(destination) as saved:
                assert np.array_equal(np.asarray(saved), pixels)
            row.append((label, pixels))
            fraction = float((pixels.max(axis=2) > 8).mean())
            details.append({'slot': label, 'image_id': record.image_id,
                            'image_path': record.image_path,
                            'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                            'png_sha256': sha256(destination),
                            'threshold8_fraction': fraction,
                            'threshold_geometry': [threshold_geometry(pixels, t) for t in (8, 32, 64, 128)]})
            pictures.append(f'<figure><img src="images/{destination.name}" width="256" height="256" '
                            f'alt="{case} {label}"><figcaption>{label} · 像素&gt;8占比 '
                            f'{fraction:.1%}</figcaption></figure>')
        private.append({'case': case, 'group': group, 'ship_id': ship, 'images': details})
        canvas_rows.append((case, group, row))
        snippets.append(f'<section><h2>{case} · {html.escape(group)}</h2><div class="images">'
                        + ''.join(pictures) + '</div><p>观察：哪些部分可见？有哪些共同关系？'
                        '有哪些差异只能解释为“未知”？有没有证据足以排除一艘相似船？</p></section>')
    panels = []
    for start in range(0, len(canvas_rows), 3):
        selected = canvas_rows[start:start + 3]
        canvas = Image.new('RGB', (1120, 75 + 324 * len(selected)), '#edf1f4')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), 'Training-only pairs | original 256x256 pixels | same identity per official labels',
                  fill='#163148', font=font(20))
        draw.text((16, 42), 'Illustrative stratified sample. No model output, rotation, crop or enhancement.',
                  fill='#45586b', font=font(17))
        for rownum, (case, group, row) in enumerate(selected):
            y = 75 + rownum * 324
            draw.text((16, y), f'{case} | {group}', fill='#163148', font=font(20))
            for col, (label, pixels) in enumerate(row):
                x = 16 + 276 * col
                draw.text((x, y + 28), label, fill='#45586b', font=font(17))
                if pixels is not None:
                    canvas.paste(Image.fromarray(pixels), (x, y + 53))
                else:
                    draw.rectangle((x, y + 53, x + 255, y + 308), fill='#dce3e9')
                    draw.text((x + 12, y + 165), 'No additional view', fill='#45586b', font=font(17))
        path = output / f'panel-{len(panels) + 1}.png'
        canvas.save(path)
        panels.append({'file': path.name, 'sha256': sha256(path)})
    summary = {'method': 'training-intersection-original-pixel-atlas-v1',
               'seed': seed, 'source_sha256': loaded[0][2]['source_sha256'],
               'split_sha256': [sha256(p) for p in split_paths],
               'tool_sha256': sha256(__file__),
               'selection': 'smallest SHA256(pair-atlas-v1:seed:ship_id) within each predeclared stratum',
               'eligible_identity_counts': {k: len(v) for k, v in pools.items()},
               'identities_per_group': per_group, 'identities': len(chosen),
               'common_training_optical_census': census_summary,
               'images': sum(len(c['images']) for c in private),
               'validation_identity_overlap': 0, 'test_used': False,
               'identity_is_official_label_not_manual_confirmation': True,
               'representative_prevalence_sample': False,
               'threshold8_is_segmentation': False, 'exported_pixels_equal_source': True,
               'moment_anisotropy_is_calibrated_confidence': False,
               'panels': panels}
    write_json(output / 'private-cases.json', private)
    write_json(output / 'summary.json', summary)
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>真实配对观察册</title>
<style>body{max-width:1160px;margin:32px auto;padding:0 20px;background:#edf1f4;color:#163148;
font:16px/1.7 system-ui,sans-serif}h1{font-size:28px}h2{font-size:20px}section{background:white;
padding:20px;margin:24px 0;border-radius:12px}.images{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:0}img{display:block}figcaption{font-size:14px;color:#45586b}a{color:#125aad}</style>
<h1>从真实配对观察“同一艘船”</h1><p>本册只从两组划分共同的训练身份抽取样例。
同船关系来自正式标签，尚无逐例独立身份核验。多视图样例被刻意多取，不用于估计全体发生率。</p>
<p>图像为原始 TIFF 解码后的无损 PNG；保持 256×256 像素、原方向、原亮度。没有生成、修复或裁剪。
像素占比只描述大于阈值的区域，不是船体面积。原图、身份映射和本页均留在 Git 之外。</p>
'''
    (output / 'index.html').write_text(document + '\n'.join(snippets) + '</html>\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--splits', type=Path, nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--per-group', type=int, default=6)
    p.add_argument('--census', action='store_true',
                   help='Also summarize brightness occupancy of all eligible training optical images')
    a = p.parse_args()
    result = build_atlas(a.source, a.splits, a.output_dir, a.seed, a.per_group, a.census)
    print({k: result[k] for k in ('identities', 'images', 'eligible_identity_counts',
                                'validation_identity_overlap', 'exported_pixels_equal_source')})
