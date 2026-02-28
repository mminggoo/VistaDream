"""
VistaDream Single-Image-to-3D 평가 스크립트
===========================================
dataset/realmdreamer  (각 씬의 input_rgb/init_img.png 를 입력으로 사용)
dataset/wonderJourney (images/*.png 를 입력으로 사용)

평가 Metric:
  1) LLaVA-IQA  : noise-free, sharp, structure, detail, quality   (ops/eval.py)
  2) CLIP Score  : 입력 이미지와 렌더링 프레임 간 CLIP 유사도
  3) CLIP-IQA    : CLIP 기반 무참조 이미지 품질 평가
  4) IS           : Inception Score (생성 품질 + 다양성)
  5) Depth Pearson: GT depth 가 있는 경우 렌더링 depth 와의 Pearson 상관
  6) DINO         : 입력 이미지와 렌더링 프레임 간 DINO 특징 유사도
  7) Generation Time : 씬 생성 소요 시간(초)
"""

import os
import sys
import json
import time
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import pearsonr

# ── torchvision / torchmetrics ──────────────────────────────────────────────
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

# ── VistaDream 내부 모듈 ─────────────────────────────────────────────────────
from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline

# ═══════════════════════════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════════════════════════

def load_video_frames(video_path: str, max_frames: int = 50) -> list:
    """비디오에서 프레임을 균일하게 샘플링하여 PIL 이미지 리스트로 반환."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if len(frames) == 0:
        return []
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames


def load_depth_video_frames(video_path: str, max_frames: int = 50) -> list:
    """깊이 비디오에서 프레임을 샘플링하여 numpy 배열 리스트로 반환 (그레이스케일)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()
    if len(frames) == 0:
        return []
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
#  LLaVA-IQA  (ops/eval.py 와 동일한 5-metric 방식, 버그 수정 포함)
# ═══════════════════════════════════════════════════════════════════════════════

class LLaVA_IQA:
    """렌더링된 프레임들을 LLaVA VLM 으로 Yes/No 판단하여 점수화."""

    QUESTIONS = {
        'noise-free': 'Is the image free of noise or distortion',
        'sharp':      'Does the image show clear objects and sharp edges',
        'structure':  'Is the overall scene coherent and realistic in terms of layout and proportions in this image',
        'detail':     'Does this image show detailed textures and materials',
        'quality':    'Is this image overall a high quality image with clear objects, sharp edges, nice color, good overall structure, and good visual quality',
    }

    def __init__(self, device='cuda'):
        from ops.llava import Llava
        self.llava = Llava(device=device)

    @torch.no_grad()
    def evaluate(self, frames: list) -> dict:
        results = {k: [] for k in self.QUESTIONS}
        for key, question in self.QUESTIONS.items():
            query = f'<image>\n USER: {question}, just answer with yes or no? \n ASSISTANT: '
            for frame in tqdm(frames, desc=f'  LLaVA-IQA [{key}]', leave=False):
                prompt = self.llava(frame, query)
                split = str.rfind(prompt, 'ASSISTANT: ') + len('ASSISTANT: ')
                answer = prompt[split + 1:]
                results[key].append(1 if answer[:2] == 'Ye' else 0)
        # 평균 점수
        return {k: float(np.mean(v)) for k, v in results.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIP Score  (입력 이미지 ↔ 렌더링 프레임 간 코사인 유사도)
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPScoreEvaluator:
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        import clip as openai_clip
        self.device = device
        self.model, self.preprocess = openai_clip.load(model_name, device=device)

    @torch.no_grad()
    def evaluate(self, input_image: Image.Image, rendered_frames: list) -> float:
        """입력 이미지와 렌더링 프레임들 사이의 평균 CLIP 코사인 유사도."""
        ref = self.preprocess(input_image).unsqueeze(0).to(self.device)
        ref_feat = self.model.encode_image(ref)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)

        sims = []
        for frame in rendered_frames:
            img = self.preprocess(frame).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (ref_feat * feat).sum().item()
            sims.append(sim)
        return float(np.mean(sims))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIP-IQA  (무참조 이미지 품질 — "good photo" vs "bad photo" 유사도 차이)
# ═══════════════════════════════════════════════════════════════════════════════

class CLIPIQAEvaluator:
    """CLIP 기반 무참조 IQA: 'good photo' 와 'bad photo' 텍스트에 대한
       이미지 유사도 차이를 0-1 로 정규화."""

    def __init__(self, model_name='ViT-B/32', device='cuda'):
        import clip as openai_clip
        self.device = device
        self.model, self.preprocess = openai_clip.load(model_name, device=device)
        # 텍스트 프롬프트 인코딩
        good_tok = openai_clip.tokenize(['a good photo']).to(device)
        bad_tok  = openai_clip.tokenize(['a bad photo']).to(device)
        with torch.no_grad():
            self.good_feat = self.model.encode_text(good_tok)
            self.bad_feat  = self.model.encode_text(bad_tok)
            self.good_feat = self.good_feat / self.good_feat.norm(dim=-1, keepdim=True)
            self.bad_feat  = self.bad_feat / self.bad_feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def evaluate(self, rendered_frames: list) -> float:
        scores = []
        for frame in rendered_frames:
            img = self.preprocess(frame).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            good_sim = (feat * self.good_feat).sum().item()
            bad_sim  = (feat * self.bad_feat).sum().item()
            # softmax 정규화 → 0-1 스코어
            score = np.exp(good_sim) / (np.exp(good_sim) + np.exp(bad_sim))
            scores.append(score)
        return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════════
#  Inception Score (IS)
# ═══════════════════════════════════════════════════════════════════════════════

class InceptionScoreEvaluator:
    """Inception-v3 softmax 분포를 이용한 Inception Score."""

    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                  transform_input=False).to(device).eval()
        self.transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def evaluate(self, rendered_frames: list, splits: int = 5) -> tuple:
        """IS 평균 ± 표준편차를 반환."""
        preds = []
        for frame in rendered_frames:
            img = self.transform(frame).unsqueeze(0).to(self.device)
            logits = self.model(img)
            prob = torch.nn.functional.softmax(logits, dim=1)
            preds.append(prob.cpu().numpy())
        preds = np.concatenate(preds, axis=0)

        # split-wise IS
        N = len(preds)
        split_scores = []
        for k in range(splits):
            part = preds[k * N // splits: (k + 1) * N // splits]
            if len(part) == 0:
                continue
            p_y = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
            kl = np.mean(np.sum(kl, axis=1))
            split_scores.append(np.exp(kl))
        return float(np.mean(split_scores)), float(np.std(split_scores))


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth Pearson Correlation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_depth_pearson(gt_depth_dir: str, rendered_depth_frames: list) -> float:
    """GT depth (.npy) 파일과 렌더링된 depth 프레임 간 Pearson 상관 계수 평균.
       GT가 없으면 None 반환."""
    gt_files = sorted(glob(os.path.join(gt_depth_dir, '*.npy')))
    if len(gt_files) == 0:
        return None

    correlations = []
    n_compare = min(len(gt_files), len(rendered_depth_frames))
    for i in range(n_compare):
        gt = np.load(gt_files[i]).astype(np.float32)
        rd = rendered_depth_frames[i]
        # 크기 맞추기
        if gt.shape != rd.shape:
            rd = cv2.resize(rd, (gt.shape[1], gt.shape[0]))
        gt_flat = gt.flatten()
        rd_flat = rd.flatten()
        # 유효한 depth 만 사용 (0 이 아닌)
        valid = (gt_flat > 0) & (rd_flat > 0)
        if valid.sum() < 10:
            continue
        corr, _ = pearsonr(gt_flat[valid], rd_flat[valid])
        if not np.isnan(corr):
            correlations.append(corr)
    if len(correlations) == 0:
        return None
    return float(np.mean(correlations))


# ═══════════════════════════════════════════════════════════════════════════════
#  DINO Score  (입력 이미지 ↔ 렌더링 프레임 간 DINOv2 특징 유사도)
# ═══════════════════════════════════════════════════════════════════════════════

class DINOScoreEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def evaluate(self, input_image: Image.Image, rendered_frames: list) -> float:
        ref = self.transform(input_image).unsqueeze(0).to(self.device)
        ref_feat = self.model(ref)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)

        sims = []
        for frame in rendered_frames:
            img = self.transform(frame).unsqueeze(0).to(self.device)
            feat = self.model(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = (ref_feat * feat).sum().item()
            sims.append(sim)
        return float(np.mean(sims))


# ═══════════════════════════════════════════════════════════════════════════════
#  데이터셋 수집
# ═══════════════════════════════════════════════════════════════════════════════

def collect_dataset_entries(dataset_root: str) -> list:
    """평가 대상 (name, input_image_path, gt_depth_dir_or_None) 리스트 반환."""
    entries = []

    # ── realmdreamer ─────────────────────────────────────────────────────────
    rd_root = os.path.join(dataset_root, 'realmdreamer')
    if os.path.isdir(rd_root):
        for scene in sorted(os.listdir(rd_root)):
            scene_dir = os.path.join(rd_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            rgb_dir = os.path.join(scene_dir, 'input_rgb')
            if not os.path.isdir(rgb_dir):
                continue
            imgs = sorted(glob(os.path.join(rgb_dir, '*.png')) +
                          glob(os.path.join(rgb_dir, '*.jpg')))
            depth_dir = os.path.join(scene_dir, 'depth')
            gt_depth = depth_dir if os.path.isdir(depth_dir) else None
            for img_path in imgs:
                entries.append({
                    'name': f'realmdreamer/{scene}',
                    'input_rgb': img_path,
                    'gt_depth_dir': gt_depth,
                })

    # ── wonderJourney ────────────────────────────────────────────────────────
    wj_root = os.path.join(dataset_root, 'wonderJourney', 'images')
    if os.path.isdir(wj_root):
        imgs = sorted(glob(os.path.join(wj_root, '*.png')) +
                       glob(os.path.join(wj_root, '*.jpg')))
        for img_path in imgs:
            name = os.path.splitext(os.path.basename(img_path))[0]
            entries.append({
                'name': f'wonderJourney/{name}',
                'input_rgb': img_path,
                'gt_depth_dir': None,
            })

    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  VistaDream 실행 + 평가
# ═══════════════════════════════════════════════════════════════════════════════

def run_vistadream(cfg, input_rgb_path: str, output_dir: str):
    """VistaDream 파이프라인을 실행하고, 결과 디렉토리 경로와 소요 시간(초)을 반환."""
    os.makedirs(output_dir, exist_ok=True)
    # 입력 이미지를 output_dir 에 복사 (파이프라인이 in-place 로 동작)
    dst_rgb = os.path.join(output_dir, 'color.png')
    img = Image.open(input_rgb_path).convert('RGB')
    img.save(dst_rgb)

    # cfg 업데이트
    cfg = deepcopy(cfg)
    cfg.scene.input.rgb = dst_rgb

    # OneFormer/Detectron2 가 sys.argv 를 파싱하므로 임시로 비워줌
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]]

    pipeline = Pipeline(cfg)

    start_time = time.time()
    pipeline()
    elapsed = time.time() - start_time

    sys.argv = orig_argv
    return output_dir, elapsed


def evaluate_scene(entry: dict,
                   output_dir: str,
                   elapsed_time: float,
                   llava_iqa_eval,
                   clip_eval,
                   clip_iqa_eval,
                   is_eval,
                   dino_eval,
                   max_frames: int = 50) -> dict:
    """하나의 씬에 대해 모든 metric 을 평가하여 dict 로 반환."""
    result = {'name': entry['name'], 'generation_time_sec': elapsed_time}

    video_rgb_path = os.path.join(output_dir, 'video_rgb.mp4')
    video_dpt_path = os.path.join(output_dir, 'video_dpt.mp4')

    if not os.path.exists(video_rgb_path):
        print(f'  [WARN] {video_rgb_path} 가 존재하지 않습니다. 건너뜁니다.')
        return result

    # 프레임 로드
    rgb_frames = load_video_frames(video_rgb_path, max_frames)
    dpt_frames = load_depth_video_frames(video_dpt_path, max_frames)
    input_image = Image.open(entry['input_rgb']).convert('RGB')

    if len(rgb_frames) == 0:
        print(f'  [WARN] 렌더링된 프레임이 없습니다.')
        return result

    # ── 1) LLaVA-IQA ────────────────────────────────────────────────────────
    print(f'  ▸ LLaVA-IQA 평가 중...')
    try:
        iqa_scores = llava_iqa_eval.evaluate(rgb_frames)
        result.update(iqa_scores)
    except Exception as e:
        print(f'  [ERR] LLaVA-IQA 실패: {e}')

    # ── 2) CLIP Score ────────────────────────────────────────────────────────
    print(f'  ▸ CLIP Score 평가 중...')
    try:
        result['CLIP'] = clip_eval.evaluate(input_image, rgb_frames)
    except Exception as e:
        print(f'  [ERR] CLIP 실패: {e}')

    # ── 3) CLIP-IQA ──────────────────────────────────────────────────────────
    print(f'  ▸ CLIP-IQA 평가 중...')
    try:
        result['CLIP-IQA'] = clip_iqa_eval.evaluate(rgb_frames)
    except Exception as e:
        print(f'  [ERR] CLIP-IQA 실패: {e}')

    # ── 4) Inception Score ───────────────────────────────────────────────────
    print(f'  ▸ Inception Score 평가 중...')
    try:
        is_mean, is_std = is_eval.evaluate(rgb_frames)
        result['IS_mean'] = is_mean
        result['IS_std'] = is_std
    except Exception as e:
        print(f'  [ERR] IS 실패: {e}')

    # ── 5) Depth Pearson ─────────────────────────────────────────────────────
    if entry.get('gt_depth_dir') and len(dpt_frames) > 0:
        print(f'  ▸ Depth Pearson 평가 중...')
        try:
            dp = compute_depth_pearson(entry['gt_depth_dir'], dpt_frames)
            result['Depth_Pearson'] = dp
        except Exception as e:
            print(f'  [ERR] Depth Pearson 실패: {e}')
    else:
        result['Depth_Pearson'] = None

    # ── 6) DINO Score ────────────────────────────────────────────────────────
    print(f'  ▸ DINO Score 평가 중...')
    try:
        result['DINO'] = dino_eval.evaluate(input_image, rgb_frames)
    except Exception as e:
        print(f'  [ERR] DINO 실패: {e}')

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='VistaDream 평가 스크립트')
    parser.add_argument('--cfg', type=str, default='pipe/cfgs/basic.yaml',
                        help='VistaDream 기본 설정 파일 경로')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='평가 데이터셋 루트 디렉토리')
    parser.add_argument('--output', type=str, default='eval_output',
                        help='평가 결과 저장 디렉토리')
    parser.add_argument('--result_json', type=str, default='eval_results.json',
                        help='결과 JSON 파일 경로')
    parser.add_argument('--max_frames', type=int, default=50,
                        help='평가에 사용할 최대 프레임 수')
    parser.add_argument('--skip_generation', action='store_true',
                        help='이미 생성된 결과가 있으면 재생성 건너뛰기')
    args = parser.parse_args()

    # ── 데이터셋 수집 ────────────────────────────────────────────────────────
    entries = collect_dataset_entries(args.dataset)
    print(f'총 {len(entries)} 건의 평가 대상을 발견했습니다.\n')
    if len(entries) == 0:
        print('평가 대상이 없습니다. --dataset 경로를 확인하세요.')
        return

    # ── 평가 모듈 초기화 (GPU 메모리를 위해 순차 로드) ────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('평가 모듈 초기화 중...')
    llava_iqa_eval = LLaVA_IQA(device=device)
    clip_eval      = CLIPScoreEvaluator(device=device)
    clip_iqa_eval  = CLIPIQAEvaluator(device=device)
    is_eval        = InceptionScoreEvaluator(device=device)
    dino_eval      = DINOScoreEvaluator(device=device)
    print('평가 모듈 초기화 완료.\n')

    # ── VistaDream 설정 ──────────────────────────────────────────────────────
    base_cfg = load_cfg(args.cfg)

    all_results = []

    for idx, entry in enumerate(entries):
        print(f'\n{"="*70}')
        print(f'[{idx+1}/{len(entries)}] {entry["name"]}')
        print(f'  입력: {entry["input_rgb"]}')
        print(f'{"="*70}')

        scene_output_dir = os.path.join(args.output, entry['name'].replace('/', os.sep))

        # ── 생성 ─────────────────────────────────────────────────────────
        video_path = os.path.join(scene_output_dir, 'video_rgb.mp4')
        if args.skip_generation and os.path.exists(video_path):
            print(f'  기존 결과 재사용 (--skip_generation)')
            elapsed = 0.0
        else:
            print(f'  VistaDream 파이프라인 실행 중...')
            try:
                _, elapsed = run_vistadream(base_cfg, entry['input_rgb'], scene_output_dir)
                print(f'  생성 완료 — {elapsed:.1f}초 소요')
            except Exception as e:
                print(f'  [ERR] 파이프라인 실행 실패: {e}')
                all_results.append({'name': entry['name'], 'error': str(e)})
                continue

        # ── 평가 ─────────────────────────────────────────────────────────
        result = evaluate_scene(
            entry, scene_output_dir, elapsed,
            llava_iqa_eval, clip_eval, clip_iqa_eval, is_eval, dino_eval,
            max_frames=args.max_frames,
        )
        all_results.append(result)

        # 중간 저장
        with open(args.result_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════════════════
    #  최종 요약
    # ═══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"="*70}')
    print('  최종 평가 결과 요약')
    print(f'{"="*70}')

    metric_keys = ['noise-free', 'sharp', 'structure', 'detail', 'quality',
                   'CLIP', 'CLIP-IQA', 'IS_mean', 'Depth_Pearson', 'DINO',
                   'generation_time_sec']

    # 평균 계산
    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in all_results if key in r and r[key] is not None]
        if vals:
            summary[key] = float(np.mean(vals))
        else:
            summary[key] = None

    for key in metric_keys:
        val = summary[key]
        val_str = f'{val:.4f}' if val is not None else 'N/A'
        print(f'  {key:25s} : {val_str}')

    # JSON에 요약 추가
    final_output = {
        'per_scene': all_results,
        'summary': summary,
    }
    with open(args.result_json, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f'\n결과 저장 완료: {args.result_json}')


if __name__ == '__main__':
    main()

# python evaluate_vistadream.py \
#     --cfg pipe/cfgs/basic.yaml \
#     --dataset dataset \
#     --output eval_output \
#     --result_json eval_results.json
