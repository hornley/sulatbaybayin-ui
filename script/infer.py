import argparse
import torch
import os
from PIL import Image, ImageDraw, ImageOps
from src.detection.dataset import BBoxDataset
import itertools
import re

# Optional OpenAI integration (module may be missing at runtime)
try:
    import openai
except Exception:
    openai = None


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
                     y_thresh_mult=0.5, x_thresh_mult=0.6, min_thresh_px=10,
                     max_symbol_width_mult=1.6):
    img = Image.open(img_path)
    # Respect EXIF orientation tags for photos taken on phones/cameras
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert('RGB')

    # Try multiple rotations (0,90,180,270) and pick the one with the most
    # confident detections. This helps with scanned/rotated pages. This is
    # best-effort and increases inference time up to 4x.
    import torchvision.transforms as T
    best_metric = (-1, -1.0)  # (count, sum_scores)
    best_angle = 0
    best_out = None
    best_rot_img = img
    for angle in (0, 90, 180, 270):
        # rotate and expand so content isn't cropped
        rot = img.rotate(angle, expand=True)
        tensor = T.ToTensor()(rot).to(device)
        with torch.no_grad():
            try:
                outputs = model([tensor])
            except Exception:
                # If model errors for a rotation, skip it
                continue
        out_candidate = outputs[0]
        # count predictions above threshold
        scores = [float(s) for s in out_candidate.get('scores', [])]
        cnt = sum(1 for s in scores if s >= thresh)
        sum_scores = sum(s for s in scores if s >= thresh)
        metric = (cnt, sum_scores)
        if metric > best_metric:
            best_metric = metric
            best_angle = angle
            best_out = out_candidate
            best_rot_img = rot

    # If model never produced outputs (best_out is None), run once without rotation
    if best_out is None:
        tensor = T.ToTensor()(img).to(device)
        with torch.no_grad():
            outputs = model([tensor])
        out = outputs[0]
        used_img = img
    else:
        out = best_out
        used_img = best_rot_img
    
    # Calculate scaled font size and line width based on image dimensions
    img_width, img_height = used_img.size
    img_diagonal = (img_width ** 2 + img_height ** 2) ** 0.5
    # Increase font size scale so labels are more readable on typical images.
    # Use a larger minimum to avoid tiny text on small images.
    font_size = max(14, int(img_diagonal / 60))
    line_width = max(1, int(img_diagonal / 400))

    # Try to load a scalable TrueType font. Try several common fonts, then fall
    # back to PIL's default font (which cannot be resized).
    try:
        from PIL import ImageFont
        font = None
        for candidate in ("DejaVuSans.ttf", "arial.ttf", "/System/Library/Fonts/SFNSDisplay.ttf"):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                font = None
        if font is None:
            # final fallback
            font = ImageFont.load_default()
    except Exception:
        font = None
    
    draw = ImageDraw.Draw(used_img)
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
        # Draw text with scaled font and a contrasting stroke for readability
        text = f"{cname}:{score:.2f}"
        text_offset = max(2, int(img_diagonal / 400))
        stroke_w = max(1, int(line_width / 2))
        try:
            if font:
                draw.text((x1 + text_offset, y1 + text_offset), text, fill='red', font=font, stroke_width=stroke_w, stroke_fill='black')
            else:
                draw.text((x1 + text_offset, y1 + text_offset), text, fill='red')
        except TypeError:
            # Older Pillow may not support stroke parameters; fallback gracefully
            if font:
                draw.text((x1 + text_offset, y1 + text_offset), text, fill='red', font=font)
            else:
                draw.text((x1 + text_offset, y1 + text_offset), text, fill='red')
        preds.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'xc': xc, 'yc': yc, 'w': w, 'h': h, 'label': cname, 'score': float(score)})

    # If no preds, return empty structures (7-tuple)
    if not preds:
        return used_img, [], '', [], [], [], []

    # compute average sizes to set clustering thresholds
    avg_h = sum(p['h'] for p in preds) / len(preds)
    avg_w = sum(p['w'] for p in preds) / len(preds)

    # Detect and remove invalid detections: boxes that are much wider than
    # the average symbol width (likely contain two symbols). These will be
    # excluded from clustering and from the final sentence permutations.
    invalid_boxes = []
    filtered_preds = []
    for p in preds:
        if p['w'] > max_symbol_width_mult * avg_w:
            p_copy = p.copy()
            p_copy['reason'] = f'width>{max_symbol_width_mult}x_avg'
            invalid_boxes.append(p_copy)
        else:
            filtered_preds.append(p)

    # If after filtering there are no valid preds, return but include invalids
    if not filtered_preds:
        flat_invalid = sorted(invalid_boxes, key=lambda x: (x['yc'], x['xc']))
        preds_sorted_invalid = [(v['x1'], v['label'], v['score']) for v in flat_invalid]
        return img, preds_sorted_invalid, '', [], [], flat_invalid, invalid_boxes

    # use filtered preds for the remainder of the pipeline
    preds = filtered_preds

    # recompute averages using filtered preds
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
    import re

    # Optional OpenAI integration
    try:
        import openai
    except Exception:
        openai = None
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
    # Also compute per-line position centers for spacing decisions
    per_line_centers = []
    for line_idx, line_choices in enumerate(all_line_permutations):
        # compute centers for this line positions from annotated_positions
        centers = []
        for pos in annotated_positions[line_idx]:
            # representative center for this position: mean of candidate x-centers
            mean_x = sum([c['xc'] for c in pos]) / len(pos)
            centers.append(mean_x)
        per_line_centers.append(centers)

        # line_choices: list of lists of labels for that line positions
        combos = list(itertools.product(*line_choices)) if line_choices else [()]
        line_strings = []
        # decide separator threshold to mark an inter-word gap (larger than clustering threshold)
        sep_threshold = 1.8 * max(x_thresh_mult * avg_w, min_thresh_px)
        for combo in combos:
            # build string by concatenating symbols within a word (no separator), and
            # inserting a single space between positions whose center gap exceeds sep_threshold
            pieces = []
            for pi, token in enumerate(combo):
                pieces.append(token)
            if len(centers) <= 1:
                # single position, just use the token
                line_strings.append(''.join(pieces))
            else:
                s = ''
                for i, tok in enumerate(pieces):
                    s += tok
                    if i < len(centers) - 1:
                        gap = centers[i+1] - centers[i]
                        if gap > sep_threshold:
                            s += ' '
                line_strings.append(s)
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

    # Always return invalid_boxes as the final element so callers can log or inspect
    # any detections that were filtered out earlier (e.g. boxes that were too wide).
    # Return the annotated image in the chosen rotation (upright according to
    # model's best orientation). Caller should display the returned image as-is.
    return used_img, preds_sorted, best_sentence, overall_sentences, annotated_positions, flat_preds_sorted, invalid_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input', required=True, help='Image file or folder')
    parser.add_argument('--out', default=None, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--thresh', type=float, default=0.85)
    parser.add_argument('--no-permutations', action='store_true', help='Do not write all permutations to per-image txt')
    parser.add_argument('--global-format', choices=['txt', 'csv', 'json'], default='txt', help='Format for global predictions file')
    parser.add_argument('--use-openai', action='store_true', help='Call OpenAI to identify Tagalog words from permutations (requires OPENAI_API_KEY)')
    parser.add_argument('--openai-model', default='gpt-4o-mini', help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--openai-key', default=None, help='OpenAI API key (optional, overrides OPENAI_API_KEY env var)')
    parser.add_argument('--y-thresh-mult', type=float, default=0.5, help='Multiplier for y clustering threshold (relative to avg height)')
    parser.add_argument('--x-thresh-mult', type=float, default=0.6, help='Multiplier for x clustering threshold (relative to avg width)')
    parser.add_argument('--min-thresh-px', type=int, default=10, help='Minimum pixel threshold for clustering')
    parser.add_argument('--compile-inferred', action='store_true', help='Produce a single compiled inferred text file (compiled_inferred.txt) instead of separate per-image inference txts')
    parser.add_argument('--max-symbol-width-mult', type=float, default=1.6, help='Max allowed symbol box width as multiple of avg symbol width; wider boxes are marked invalid')
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
    import json, csv, re
    if args.global_format == 'txt':
        # accumulate entries and write at the end so we can place a summary at the top
        g_entries = []
        write_global = lambda name, sent, openai_resp, recon: g_entries.append({'image': name, 'best_sentence': sent, 'openai': openai_resp, 'openai_reconstructed': recon})
    elif args.global_format == 'csv':
        gfh = open(global_preds_path, 'w', encoding='utf8', newline='')
        gcsv = csv.writer(gfh)
        gcsv.writerow(['image', 'best_sentence', 'openai_response', 'openai_reconstructed'])
        write_global = lambda name, sent, openai_resp, recon: gcsv.writerow([name, sent, openai_resp, recon])
    else:
        # json: accumulate and write at the end
        g_entries = []
        write_global = lambda name, sent, openai_resp, recon: g_entries.append({'image': name, 'best_sentence': sent, 'openai': openai_resp, 'openai_reconstructed': recon})

    compiled_entries = []
    compiled_annotations = []
    for p in paths:
        img_out, preds_sorted, best_sentence, overall_sentences, annotated_positions, raw_preds, invalid_boxes = draw_predictions(
            p, model, classes, device=device, thresh=args.thresh,
            y_thresh_mult=args.y_thresh_mult, x_thresh_mult=args.x_thresh_mult, min_thresh_px=args.min_thresh_px,
            max_symbol_width_mult=getattr(args, 'max_symbol_width_mult', 1.6)
        )
        out_img_path = os.path.join(args.out, os.path.basename(p))
        img_out.save(out_img_path)
        print('wrote', out_img_path)
        # write per-image text file with label and score
        base = os.path.splitext(os.path.basename(p))[0]
        txt_path = os.path.join(args.out, base + '.txt')
        # Expand vowel permutations (i -> [i,e], u -> [u,o]) for all overall permutations
        def expand_vowel_permutations_sentence(s):
            # s may contain ' <NL> ' joins for lines
            lines = s.split(' <NL> ')
            expanded_lines = []
            for line in lines:
                tokens = [t for t in line.split(' ') if t != '']
                # per-token alternatives
                alt_lists = []
                for tok in tokens:
                    m = re.match(r"^([b-df-hj-np-tv-z])(i|u)$", tok, flags=re.IGNORECASE)
                    if m:
                        cons = m.group(1)
                        vow = m.group(2).lower()
                        if vow == 'i':
                            alts = [cons + 'i', cons + 'e']
                        else:
                            alts = [cons + 'u', cons + 'o']
                        alt_lists.append(alts)
                    else:
                        alt_lists.append([tok])
                # Cartesian product
                if alt_lists:
                    combos = list(itertools.product(*alt_lists))
                    expanded = [''.join(c) for c in combos]
                else:
                    expanded = ['']
                expanded_lines.append(expanded)
            # combine lines back with ' <NL> '
            combined = []
            for prod in itertools.product(*expanded_lines):
                combined.append(' <NL> '.join(prod))
            return combined

        # Prepare expanded permutations for OpenAI
        expanded_permutations = []
        for s in overall_sentences:
            expanded_permutations.extend(expand_vowel_permutations_sentence(s))

        # OpenAI call (optional): supports new openai.OpenAI client and falls back to legacy ChatCompletion
        def call_openai_identify_tagalog(candidates, model_name='gpt-4o-mini', api_key=None):
            if not args.use_openai:
                return 'OPENAI_DISABLED'
            if openai is None:
                return 'OPENAI_PYTHON_NOT_INSTALLED'
            key = api_key if api_key else os.environ.get('OPENAI_API_KEY')
            if not key:
                return 'OPENAI_API_KEY_NOT_SET'

            # Include original best sentence as an extra candidate so concatenations
            # such as 'miss' (from 'mi s s') appear in the prompt explicitly.
            extra_info = ''
            if hasattr(args, 'best_sentence') and args.best_sentence:
                extra_info = '\nOriginalBestSentence:\n' + args.best_sentence + '\n'

            prompt = (
                'You are given a list of candidate word/phrase permutations (possible Filipino/Tagalog words).'
                ' From these candidates, identify which individual tokens are valid Filipino/Tagalog words.'
                ' Return a JSON object with three keys: "tagalog" (array of unique Tagalog words found),'
                ' "others" (array of other candidate words found), and "reconstructed" (a best-effort reconstructed'
                ' full sentence combining tokens). Do not include any extra commentary; output only valid JSON.'
                f"\n\nCandidates:\n" + '\n'.join(candidates[:200]) + extra_info
            )

            messages = [
                {'role': 'system', 'content': 'You are a helpful language assistant who only outputs JSON.'},
                {'role': 'user', 'content': prompt}
            ]

            try:
                # Prefer new 1.x style client if available
                if hasattr(openai, 'OpenAI'):
                    try:
                        client = openai.OpenAI(api_key=key)
                    except TypeError:
                        # some versions accept no arg and use env var
                        client = openai.OpenAI()
                        try:
                            client.api_key = key
                        except Exception:
                            pass
                    try:
                        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.0, max_tokens=600)
                        # Attempt to extract assistant content
                        choice = resp.choices[0]
                        text = None
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            text = choice.message.content
                        elif isinstance(choice, dict) and 'message' in choice and 'content' in choice['message']:
                            text = choice['message']['content']
                        else:
                            text = str(choice)
                        return text.strip() if text else ''
                    except Exception as e:
                        return f'OPENAI_ERROR: {e}'

                # Fallback: legacy ChatCompletion API
                if hasattr(openai, 'ChatCompletion'):
                    try:
                        openai.api_key = key
                        resp = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=0.0, max_tokens=600)
                        return resp['choices'][0]['message']['content'].strip()
                    except Exception as e:
                        return f'OPENAI_ERROR: {e}'

                return 'OPENAI_UNSUPPORTED_CLIENT'
            except Exception as e:
                return f'OPENAI_ERROR: {e}'

        # attach best_sentence to args for the helper to include it in the prompt
        args.best_sentence = best_sentence
        openai_response = call_openai_identify_tagalog(expanded_permutations, model_name=args.openai_model, api_key=args.openai_key)

        # try to parse JSON response to extract reconstructed sentence (if model followed JSON instruction)
        openai_reconstructed = ''
        try:
            parsed = json.loads(openai_response)
            if isinstance(parsed, dict):
                openai_reconstructed = parsed.get('reconstructed') or parsed.get('best_sentence') or parsed.get('sentence') or ''
        except Exception:
            # not JSON or parse failed; leave reconstructed empty
            openai_reconstructed = ''

        with open(txt_path, 'w', encoding='utf8') as fh:
            for x1, cname, score in preds_sorted:
                fh.write(f"{cname}\t{score:.4f}\n")
            fh.write('\n')
            fh.write('BEST_SENTENCE:\n')
            fh.write(best_sentence + '\n')
            fh.write('\n')
            fh.write('OPENAI_RESPONSE:\n')
            fh.write(openai_response + '\n')
            fh.write('\n')
            # list invalid/removed boxes (if any)
            if invalid_boxes:
                fh.write('INVALID_BOXES:\n')
                for ib in invalid_boxes:
                    fh.write(f"  {ib['x1']:.1f},{ib['y1']:.1f},{ib['x2']:.1f},{ib['y2']:.1f}\t{ib['label']}\t{ib['score']:.4f}\t{ib.get('reason','')}\n")
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
        # append to global predictions file / structure (include openai response if available)
        write_global(os.path.basename(p), best_sentence, openai_response, openai_reconstructed)
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

    # finalize global predictions file
    if args.global_format == 'json':
        with open(global_preds_path, 'w', encoding='utf8') as gj:
            json.dump(g_entries, gj, ensure_ascii=False, indent=2)
    elif args.global_format == 'csv':
        gfh.close()
    else:
        # txt: write a summary of unique words at the top, then per-image rows
        # Write a concise summary at the top: image -> reconstructed (fallback to best_sentence)
        with open(global_preds_path, 'w', encoding='utf8') as gfh_out:
            gfh_out.write('SUMMARY (image -> reconstructed):\n')
            for e in g_entries:
                recon = e.get('openai_reconstructed') or e.get('best_sentence') or ''
                gfh_out.write(f"{e['image']} -> {recon}\n")
            gfh_out.write('\n')
            # now write per-image tab-separated rows for more detail
            for e in g_entries:
                gfh_out.write(f"{e['image']}\t{e['best_sentence']}\t{e['openai']}\t{e['openai_reconstructed']}\n")
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
