import argparse
import torch
import os
from PIL import Image, ImageDraw
from script.dataset import BBoxDataset


def generate_output_dir(checkpoint_path, base_dir='detections'):
    """
    Generate output directory based on checkpoint path.
    
    Example:
        checkpoints/detection/colab_run10/stage2/best.pth
        -> detections/colab_run10/stage2/infer1
        
        checkpoints/stage2/best.pth
        -> detections/stage2/infer1
    """
    # Parse checkpoint path to extract run identifier and stage
    ckpt_parts = os.path.normpath(checkpoint_path).split(os.sep)
    
    # Find 'checkpoints' in path and extract everything after it (excluding the filename)
    try:
        ckpt_idx = ckpt_parts.index('checkpoints')
        # Get path components after 'checkpoints' but before filename
        run_parts = ckpt_parts[ckpt_idx + 1:-1]  # Exclude 'checkpoints' and filename
        # Remove 'detection' or 'classification' from run_parts if present
        run_parts = [p for p in run_parts if p not in ('detection', 'classification')]
    except (ValueError, IndexError):
        # If 'checkpoints' not in path, use parent directory of checkpoint file
        run_parts = [os.path.basename(os.path.dirname(checkpoint_path))]
    
    # Build base path for this run
    if run_parts:
        run_path = os.path.join(base_dir, *run_parts)
    else:
        run_path = base_dir
    
    # Find next available infer number
    os.makedirs(run_path, exist_ok=True)
    existing = [d for d in os.listdir(run_path) if d.startswith('infer') and os.path.isdir(os.path.join(run_path, d))]
    
    # Extract numbers from existing infer folders
    numbers = []
    for d in existing:
        try:
            num = int(d.replace('infer', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(run_path, f'infer{next_num}')


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model_state = ck['model_state']
    classes = ck.get('classes', [])
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, classes


def draw_predictions(img_path, model, classes, device='cpu', thresh=0.9,
                     y_thresh_mult=0.5, x_thresh_mult=0.6, min_thresh_px=10):
    img = Image.open(img_path).convert('RGB')
    import torchvision.transforms as T
    tensor = T.ToTensor()(img).to(device)
    with torch.no_grad():
        outputs = model([tensor])
    out = outputs[0]
    
    # Calculate scaled font size and line width based on image dimensions
    img_width, img_height = img.size
    img_diagonal = (img_width ** 2 + img_height ** 2) ** 0.5
    # Scale font size: base size 12 for ~1000px diagonal, scale proportionally
    font_size = max(10, int(img_diagonal / 80))
    line_width = max(1, int(img_diagonal / 500))
    
    # Try to load a scalable font, fall back to default if not available
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            from PIL import ImageFont
            # Try default font with size
            font = ImageFont.load_default()
        except:
            font = None
    
    draw = ImageDraw.Draw(img)
    preds = []  # list of dicts with box, centers, size, label_name, score
    for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
        if score < thresh:
            continue
        x1, y1, x2, y2 = map(float, box)
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        cname = classes[label - 1] if label > 0 and label - 1 < len(classes) else str(int(label))
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line_width)
        # Draw text with scaled font
        text = f"{cname}:{score:.2f}"
        text_offset = max(2, int(img_diagonal / 400))
        if font:
            draw.text((x1 + text_offset, y1 + text_offset), text, fill='red', font=font)
        else:
            draw.text((x1 + text_offset, y1 + text_offset), text, fill='red')
        preds.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'xc': xc, 'yc': yc, 'w': w, 'h': h, 'label': cname, 'score': float(score)})

    # If no preds, return empty structures (6-tuple)
    if not preds:
        return img, [], '', [], [], []

    # compute average sizes to set clustering thresholds
    avg_h = sum(p['h'] for p in preds) / len(preds)
    avg_w = sum(p['w'] for p in preds) / len(preds)

    # cluster into lines by y-center (top -> down)
    preds_sorted_y = sorted(preds, key=lambda p: p['yc'])
    lines = []  # list of lists of preds
    current_line = [preds_sorted_y[0]]
    for p in preds_sorted_y[1:]:
        if abs(p['yc'] - current_line[-1]['yc']) > max(y_thresh_mult * avg_h, min_thresh_px):
            lines.append(current_line)
            current_line = [p]
        else:
            current_line.append(p)
    lines.append(current_line)

    # within each line, cluster by x to form symbol positions (left->right)
    import itertools
    all_line_permutations = []  # list of lists-of-candidate-labels per line
    annotated_positions = []
    for line in lines:
        # sort by x center
        line_sorted = sorted(line, key=lambda q: q['xc'])
        positions = []  # each position is list of preds (alternatives)
        current_pos = [line_sorted[0]]
        for q in line_sorted[1:]:
            if abs(q['xc'] - current_pos[-1]['xc']) > max(x_thresh_mult * avg_w, min_thresh_px):
                positions.append(current_pos)
                current_pos = [q]
            else:
                current_pos.append(q)
        positions.append(current_pos)

        # for each position, get candidate labels ordered by score desc
        line_candidates = []
        for pos in positions:
            sorted_pos = sorted(pos, key=lambda r: r['score'], reverse=True)
            labels = [r['label'] for r in sorted_pos]
            line_candidates.append(labels)
        all_line_permutations.append(line_candidates)
        # keep positions for per-image text output with coords
        annotated_positions.append(positions)

    # produce sentence permutations: Cartesian product of choices per position within each line,
    # then join lines top->down with ' <NL> '
    line_permutation_strings = []  # list of lists (each list contains permutations for that line)
    for line_choices in all_line_permutations:
        # line_choices: list of lists of labels for that line positions
        # produce all combinations for this line
        combos = list(itertools.product(*line_choices)) if line_choices else [()]
        line_strings = [' '.join(c) for c in combos]
        line_permutation_strings.append(line_strings)

    # Now combine lines (top->down). For each combination pick one permutation per line and join with ' <NL> '
    overall_sentences = []
    for prod in itertools.product(*line_permutation_strings):
        overall_sentences.append(' <NL> '.join(prod))

    # Score permutations: approximate by summing the score of the chosen candidate per position.
    # We'll need to map label choices back to position candidate scores.
    # Build per-line per-position candidate score lists in the same shape as all_line_permutations
    per_line_pos_scores = []
    for positions in annotated_positions:
        line_scores = []
        for pos in positions:
            sorted_pos = sorted(pos, key=lambda r: r['score'], reverse=True)
            scores = [r['score'] for r in sorted_pos]
            line_scores.append(scores)
        per_line_pos_scores.append(line_scores)

    # helper to compute sentence score for a chosen combination per line
    import math
    def score_sentence_by_choice(line_choice_lists):
        # line_choice_lists: list (per line) of chosen labels tuple for positions
        total = 0.0
        for li, chosen in enumerate(line_choice_lists):
            # for each position, find the candidate index for the chosen label and add its score
            positions = annotated_positions[li]
            for pi, lab in enumerate(chosen):
                # find matching candidate in positions[pi]
                candidates = positions[pi]
                matched = None
                for c in candidates:
                    if c['label'] == lab:
                        matched = c
                        break
                if matched is not None:
                    total += matched['score']
                else:
                    # penalty if label not found (shouldn't happen)
                    total -= 0.5
        return total

    # evaluate all overall_sentences and pick best by score
    best_sentence = ''
    sentence_scores = []
    if overall_sentences:
        # To re-create the chosen labels per line for each overall sentence,
        # split by ' <NL> ' then split per-line string into tokens
        for s in overall_sentences:
            per_line = [ln.split() if ln.strip() != '' else [] for ln in s.split(' <NL> ')]
            sc = score_sentence_by_choice(per_line)
            sentence_scores.append((sc, s))
        # pick highest score
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        best_sentence = sentence_scores[0][1]

    # prepare preds_sorted as list of (x1,label,score) for compatibility
    flat_preds_sorted = sorted(preds, key=lambda x: (x['yc'], x['xc']))
    preds_sorted = [(p['x1'], p['label'], p['score']) for p in flat_preds_sorted]

    return img, preds_sorted, best_sentence, overall_sentences, annotated_positions, flat_preds_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input', required=True, help='Image file or folder')
    parser.add_argument('--out', default=None, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--thresh', type=float, default=0.9)
    parser.add_argument('--no-permutations', action='store_true', help='Do not write all permutations to per-image txt')
    parser.add_argument('--global-format', choices=['txt', 'csv', 'json'], default='txt', help='Format for global predictions file')
    parser.add_argument('--y-thresh-mult', type=float, default=0.5, help='Multiplier for y clustering threshold (relative to avg height)')
    parser.add_argument('--x-thresh-mult', type=float, default=0.6, help='Multiplier for x clustering threshold (relative to avg width)')
    parser.add_argument('--min-thresh-px', type=int, default=10, help='Minimum pixel threshold for clustering')
    parser.add_argument('--compile-inferred', action='store_true', help='Produce a single compiled inferred text file (compiled_inferred.txt) instead of separate per-image inference txts')
    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.out is None:
        args.out = generate_output_dir(args.ckpt, base_dir='detections')
        print(f'Auto-generated output directory: {args.out}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, classes = load_checkpoint(args.ckpt, device=device)
    os.makedirs(args.out, exist_ok=True)
    paths = []
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(args.input, f))
    else:
        paths = [args.input]
    # global predictions file
    global_preds_path = os.path.join(args.out, 'predictions.txt')
    # prepare global predictions writer depending on format
    import json, csv
    if args.global_format == 'txt':
        gfh = open(global_preds_path, 'w', encoding='utf8')
        write_global = lambda name, sent: gfh.write(f"{name}\t{sent}\n")
    elif args.global_format == 'csv':
        gfh = open(global_preds_path, 'w', encoding='utf8', newline='')
        gcsv = csv.writer(gfh)
        gcsv.writerow(['image', 'best_sentence'])
        write_global = lambda name, sent: gcsv.writerow([name, sent])
    else:
        # json: accumulate and write at the end
        g_entries = []
        write_global = lambda name, sent: g_entries.append({'image': name, 'best_sentence': sent})

    compiled_entries = []
    compiled_annotations = []
    for p in paths:
        img_out, preds_sorted, best_sentence, overall_sentences, annotated_positions, raw_preds = draw_predictions(
            p, model, classes, device=device, thresh=args.thresh,
            y_thresh_mult=args.y_thresh_mult, x_thresh_mult=args.x_thresh_mult, min_thresh_px=args.min_thresh_px
        )
        out_img_path = os.path.join(args.out, os.path.basename(p))
        img_out.save(out_img_path)
        print('wrote', out_img_path)
        # write per-image text file with label and score
        base = os.path.splitext(os.path.basename(p))[0]
        txt_path = os.path.join(args.out, base + '.txt')
        with open(txt_path, 'w', encoding='utf8') as fh:
            for x1, cname, score in preds_sorted:
                fh.write(f"{cname}\t{score:.4f}\n")
            fh.write('\n')
            fh.write('BEST_SENTENCE:\n')
            fh.write(best_sentence + '\n')
            fh.write('\n')
            if not args.no_permutations:
                fh.write('PERMUTATIONS:\n')
                for s in overall_sentences:
                    fh.write(s + '\n')
            fh.write('\n')
            fh.write('POSITIONS (per-line, left->right, with candidates):\n')
            # annotated_positions is list of lines -> list of positions -> list of preds
            for li, line in enumerate(annotated_positions):
                fh.write(f'Line {li+1}:\n')
                for pi, pos in enumerate(line):
                    labels = [pp['label'] for pp in sorted(pos, key=lambda r: r['score'], reverse=True)]
                    fh.write(f'  Pos {pi+1}: ' + ','.join(labels) + '\n')
        # append to global predictions file / structure
        write_global(os.path.basename(p), best_sentence)
        # optionally aggregate into a compiled inferred file content
        if args.compile_inferred:
            # we will append a block per image to compiled_entries
            compiled_entries.append({'image': os.path.basename(p), 'best_sentence': best_sentence, 'permutations': overall_sentences})
            # append raw detections for CSV
            for det in raw_preds:
                compiled_annotations.append({
                    'image': os.path.basename(p),
                    'x1': det.get('x1'), 'y1': det.get('y1'), 'x2': det.get('x2'), 'y2': det.get('y2'),
                    'label': det.get('label'), 'score': det.get('score')
                })

    # finalize json if needed
    if args.global_format == 'json':
        with open(global_preds_path, 'w', encoding='utf8') as gj:
            json.dump(g_entries, gj, ensure_ascii=False, indent=2)
    else:
        gfh.close()
    # write compiled inferred file if requested
    if args.compile_inferred:
        compiled_path = os.path.join(args.out, 'compiled_inferred.txt')
        with open(compiled_path, 'w', encoding='utf8') as cf:
            for ent in compiled_entries:
                cf.write(f"# {ent['image']}\n")
                cf.write("BEST: " + ent['best_sentence'] + '\n')
                if not args.no_permutations:
                    cf.write('PERMUTATIONS:\n')
                    for s in ent['permutations']:
                        cf.write(s + '\n')
                cf.write('\n')
        print('wrote', compiled_path)
        # write compiled CSV of raw detections
        csv_path = os.path.join(args.out, 'compiled_annotations.csv')
        with open(csv_path, 'w', encoding='utf8', newline='') as cf:
            import csv as _csv
            writer = _csv.writer(cf)
            writer.writerow(['image_path', 'x1', 'y1', 'x2', 'y2', 'label', 'confidence_score'])
            for r in compiled_annotations:
                writer.writerow([r['image'], r['x1'], r['y1'], r['x2'], r['y2'], r['label'], f"{r['score']:.6f}"])
        print('wrote', csv_path)
    print('wrote', global_preds_path)


if __name__ == '__main__':
    main()
