"""
core.py - llm.py(Streamlit)에서 UI 의존성을 제거해 추출한 서버용 핵심 로직.
 - OCR(PaddleOCR) / 약품·성분 사전 / 식약처 DUR·e약은요 조회 / 경량 EXAONE 추론 / 의료 면책 문구
 - streamlit을 import하지 않으므로 FastAPI 등 어디서든 재사용 가능
"""
import os
import re
import io
import glob
import json
import time
import urllib.parse
from functools import lru_cache
from typing import Optional

import requests
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file


class _Progress:
    """Streamlit 진행바 대체용 (서버에서는 아무 것도 하지 않음)"""
    def progress(self, *args, **kwargs):
        pass

    def empty(self):
        pass


def _progress(*args, **kwargs):
    return _Progress()


@lru_cache(maxsize=None)
def get_ocr_engine():
    """PaddleOCR 3.x 엔진을 한 번 생성해 재사용합니다."""
    from paddleocr import PaddleOCR  # 지연 임포트: OCR 사용 시에만 로드
    # torch와 CUDA/cuDNN DLL이 충돌하지 않도록 OCR은 CPU로 실행 (WinError 127 방지)
    options = dict(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    try:
        return PaddleOCR(device="cpu", **options)
    except TypeError:        # device 옵션을 지원하지 않는 구버전
        return PaddleOCR(**options)


# ==========================================
# 의료 면책 문구 (UI / 프롬프트 / 리포트 말미에 공통 사용)
# ==========================================
MEDICAL_DISCLAIMER = (
    "⚠️ **의료 면책 안내**\n\n"
    "본 결과는 AI가 공공데이터(식약처 DUR·e약은요)를 바탕으로 생성한 **참고용 정보**이며, "
    "의학적 진단·처방·복약 지시를 대체하지 않습니다. AI는 오류나 누락이 있을 수 있습니다.\n\n"
    "**복용 여부, 병용, 용량 변경 등 모든 의학적 판단은 반드시 의사 또는 약사 등 전문가에게 "
    "진단·상담을 받은 후 결정하세요.** "
    "발진, 호흡곤란, 출혈, 심한 복통 등 이상 증상이 있으면 즉시 진료를 받거나 119에 연락하세요."
)


# ==========================================
# 모델 프로파일: 기기 사양에 맞춰 선택 (경량 -> 고성능)
#  - gguf : llama.cpp 4bit 양자화 (CPU/노트북/모바일급, 약 1.6GB)
#  - torch: 원본 bf16 (GPU)
#  - none : LLM 없이 규칙 + 식약처 공공데이터만 사용 (가장 가벼움)
# ==========================================
HF_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

MODEL_PROFILES = {
    "경량 · EXAONE-3.5-2.4B Q4 (약 1.6GB, CPU/노트북 가능)": {
        "backend": "gguf",
        "repo_id": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
        "filename": "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
        "short": "EXAONE-3.5-2.4B (Q4_K_M, 4bit 양자화)",
        "max_new_tokens": 300,
    },
    "균형 · EXAONE-3.5-2.4B bf16 (약 5GB, 일반 GPU)": {
        "backend": "torch",
        "repo_id": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "short": "EXAONE-3.5-2.4B (bf16)",
        "max_new_tokens": 320,
    },
    "고성능 · EXAONE-3.5-7.8B bf16 (약 16GB, RTX 4090급)": {
        "backend": "torch",
        "repo_id": HF_MODEL_ID,
        "short": "EXAONE-3.5-7.8B (bf16)",
        "max_new_tokens": 400,
    },
    "LLM 끄기 · 규칙+공공데이터만 (가장 가벼움)": {
        "backend": "none",
        "short": "규칙 기반 (LLM 미사용)",
        "max_new_tokens": 0,
    },
}
DEFAULT_PROFILE_KEY = list(MODEL_PROFILES.keys())[0]


def _find_cached_snapshot_dir(repo_id: str) -> str:
    """로컬 Hugging Face 캐시에서 해당 모델의 스냅샷 폴더를 찾고, 없으면 다운로드합니다."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snapshots_root = os.path.join(
        hf_home,
        "hub",
        "models--" + repo_id.replace("/", "--"),
        "snapshots"
    )
    if os.path.exists(snapshots_root):
        for snap in os.listdir(snapshots_root):
            snap_path = os.path.join(snapshots_root, snap)
            if os.path.isdir(snap_path):
                if glob.glob(os.path.join(snap_path, "*.safetensors")):
                    return snap_path

    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id)


def _convert_exaone_key_to_llama(key: str) -> str:
    """EXAONE-3.5 가중치 키 이름을 Transformers 내장 LlamaForCausalLM 키 이름으로 1:1 변환합니다."""
    if key == "transformer.wte.weight":
        return "model.embed_tokens.weight"
    if key == "transformer.ln_f.weight":
        return "model.norm.weight"
    if key == "lm_head.weight":
        return "lm_head.weight"

    m = re.match(r"^transformer\.h\.(\d+)\.(.+)$", key)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        mapping = {
            "ln_1.weight": "input_layernorm.weight",
            "ln_2.weight": "post_attention_layernorm.weight",
            "attn.attention.q_proj.weight": "self_attn.q_proj.weight",
            "attn.attention.k_proj.weight": "self_attn.k_proj.weight",
            "attn.attention.v_proj.weight": "self_attn.v_proj.weight",
            "attn.attention.out_proj.weight": "self_attn.o_proj.weight",
            "mlp.c_fc_0.weight": "mlp.gate_proj.weight",
            "mlp.c_fc_1.weight": "mlp.up_proj.weight",
            "mlp.c_proj.weight": "mlp.down_proj.weight",
        }
        if rest in mapping:
            return f"model.layers.{layer_idx}.{mapping[rest]}"
    return key


@lru_cache(maxsize=None)
def get_torch_llm(repo_id: str):
    """
    원격 modeling_exaone.py를 실행하지 않고 meta 디바이스로 골격을 만든 뒤 가중치를 직접 적재합니다.
    repo_id만 바꾸면 2.4B / 7.8B 모두 동일 로더로 동작합니다.
    """
    t0 = time.time()
    progress_bar = _progress(0, text=f"🔍 [1/4] {repo_id} 캐시 경로 확인 중...")
    snap_dir = _find_cached_snapshot_dir(repo_id)

    progress_bar.progress(15, text="📝 [2/4] 토크나이저 및 모델 설정(Config) 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        target_device = "cuda:0"
    else:
        model_dtype = torch.float32
        target_device = "cpu"

    with open(os.path.join(snap_dir, "config.json"), "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    llama_config = LlamaConfig(
        vocab_size=raw_cfg.get("vocab_size", 102400),
        hidden_size=raw_cfg.get("hidden_size", 4096),
        intermediate_size=raw_cfg.get("intermediate_size", 14336),
        num_hidden_layers=raw_cfg.get("num_layers", raw_cfg.get("num_hidden_layers", 32)),
        num_attention_heads=raw_cfg.get("num_attention_heads", 32),
        num_key_value_heads=raw_cfg.get("num_key_value_heads", 8),
        hidden_act=raw_cfg.get("activation_function", "silu"),
        max_position_embeddings=raw_cfg.get("max_position_embeddings", 32768),
        rms_norm_eps=raw_cfg.get("layer_norm_epsilon", 1e-5),
        tie_word_embeddings=raw_cfg.get("tie_word_embeddings", False),  # 2.4B는 True
        bos_token_id=raw_cfg.get("bos_token_id", 1),
        eos_token_id=raw_cfg.get("eos_token_id", 361),
        pad_token_id=raw_cfg.get("pad_token_id", 0),
        rope_theta=float(raw_cfg.get("rope_theta", 1000000.0)),
        attention_bias=False,
        mlp_bias=False,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    )

    progress_bar.progress(25, text="⚡ [3/4] Meta 디바이스 모델 골격 생성 중...")
    with torch.device("meta"):
        model = LlamaForCausalLM(llama_config)

    safetensor_files = sorted(glob.glob(os.path.join(snap_dir, "*.safetensors")))
    total_shards = len(safetensor_files)
    converted_state_dict = {}

    for idx, sf_path in enumerate(safetensor_files, start=1):
        pct = 25 + int((idx / max(total_shards, 1)) * 60)
        progress_bar.progress(
            pct, text=f"📦 [3/4] 가중치 샤드 적재 중 ({idx}/{total_shards}): {os.path.basename(sf_path)}"
        )
        shard = load_file(sf_path, device="cpu")
        for k, v in shard.items():
            converted_state_dict[_convert_exaone_key_to_llama(k)] = v.to(dtype=model_dtype, device=target_device)
        del shard

    progress_bar.progress(90, text="🚀 [4/4] 텐서 바인딩 및 RoPE 버퍼 초기화 중...")
    model.load_state_dict(converted_state_dict, strict=False, assign=True)
    del converted_state_dict

    # 임베딩-출력층 가중치 공유 모델(2.4B)은 lm_head가 meta에 남지 않도록 명시적으로 연결
    if llama_config.tie_word_embeddings:
        model.tie_weights()

    head_dim = llama_config.hidden_size // llama_config.num_attention_heads
    inv_freq = 1.0 / (
        llama_config.rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to(target_device) / head_dim)
    )
    for module in model.modules():
        if hasattr(module, "inv_freq"):
            module.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = inv_freq.clone()

    model.eval()
    progress_bar.progress(100, text=f"✅ 모델 적재 완료! ({time.time() - t0:.1f}초 소요)")
    time.sleep(0.3)
    progress_bar.empty()
    return tokenizer, model


@lru_cache(maxsize=None)
def get_gguf_llm(repo_id: str, filename: str):
    """4bit 양자화 GGUF 모델을 llama.cpp로 적재합니다. (CPU만으로도 동작, GPU가 있으면 자동 활용)"""
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0,
        verbose=False,
    )


def _generate_with_profile(profile: dict, messages: list) -> str:
    """프로파일의 backend에 맞춰 답변을 생성합니다. (항상 greedy: 결과 일관성 확보)"""
    max_new = profile["max_new_tokens"]

    if profile["backend"] == "gguf":
        llm = get_gguf_llm(profile["repo_id"], profile["filename"])
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new,
            temperature=0.0,
            repeat_penalty=1.05,
        )
        return out["choices"][0]["message"]["content"].strip()

    tokenizer, model = get_torch_llm(profile["repo_id"])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_device = next(model.parameters()).device
    model_inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(target_device)

    eos_ids = [tokenizer.eos_token_id]
    endoftext_id = tokenizer.convert_tokens_to_ids("[|endoftext|]")
    if isinstance(endoftext_id, int) and endoftext_id not in eos_ids and endoftext_id >= 0:
        eos_ids.append(endoftext_id)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            use_cache=True,
        )
    generated_ids = output_ids[0][model_inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# 약리 계열 사전 (LLM이 기억으로 추측하지 않도록 근거로 제공)
INGREDIENT_CLASS = {
    "아세트아미노펜": ("해열진통제 (NSAIDs 아님)", "해열진통제(APAP)"),
    "아세틸살리실산": ("NSAIDs(비스테로이드성 소염진통제) · 항혈소판 작용", "NSAIDs"),
    "이부프로펜": ("NSAIDs(비스테로이드성 소염진통제)", "NSAIDs"),
    "나프록센": ("NSAIDs(비스테로이드성 소염진통제)", "NSAIDs"),
    "이소트레티노인": ("레티노이드 (여드름 치료제)", "레티노이드"),
    "세티리진": ("항히스타민제", "항히스타민제"),
    "이소프로필안티피린": ("피린계 해열진통제", "피린계"),
    "카페인": ("중추신경 흥분제", "카페인"),
    "비타민B군": ("비타민", "비타민B"),
}


def build_class_facts(base_ingredients, second_drug="", second_ingredients=None):
    """사전 기준으로 약리 계열과 '같은 계열 중복' 여부를 계산합니다. (LLM이 아니라 코드가 판정)"""
    if second_ingredients is not None:
        second_ingredients = list(second_ingredients)
    else:
        second_ingredients = []
        if second_drug:
            try:
                second_ingredients = list(get_drug_ingredients(second_drug) or [])
            except Exception:
                second_ingredients = []
            for ing in extract_ingredients_from_text(second_drug):
                if ing not in second_ingredients:
                    second_ingredients.append(ing)

    lines = []
    for label, ings in (("기준 약품", base_ingredients or []), ("함께 복용 약품", second_ingredients)):
        for ing in ings:
            cls = INGREDIENT_CLASS.get(ing)
            lines.append(f"- {label} 성분 {ing}: " + (cls[0] if cls else "계열 정보 없음(확인 불가)"))

    base_groups = {}
    for ing in base_ingredients or []:
        if ing in INGREDIENT_CLASS:
            base_groups.setdefault(INGREDIENT_CLASS[ing][1], []).append(ing)
    second_groups = {}
    for ing in second_ingredients:
        if ing in INGREDIENT_CLASS:
            second_groups.setdefault(INGREDIENT_CLASS[ing][1], []).append(ing)

    same = [g for g in base_groups if g in second_groups]
    if not second_drug:
        lines.append("- 같은 계열 중복: 비교 약품 없음")
    elif same:
        for g in same:
            lines.append(
                f"- 같은 계열 중복: 있음 ({g}: {', '.join(base_groups[g])} / {', '.join(second_groups[g])})"
            )
    else:
        lines.append("- 같은 계열 중복: 없음 (사전 기준, 사전에 없는 성분은 확인 불가)")

    return "\n".join(lines) if lines else "- 계열 정보 없음(확인 불가)"


def build_rule_based_report(base_drug: str, second_drug: str, base_ingredients: list, dur_msg: str) -> str:
    """LLM 없이 식약처 DUR 결과와 성분 사전만으로 만드는 최소 리포트 (가장 가벼운 모드 / LLM 실패 시 대체)."""
    ingredients = ", ".join(base_ingredients) if base_ingredients else "사전 미등록"
    return (
        "### 📋 규칙 기반 복약 안내 (LLM 미사용)\n\n"
        f"**1. 기준 약품 / 주성분**: {base_drug} / {ingredients}\n\n"
        f"**2. 함께 복용하려는 약품**: {second_drug if second_drug else '미입력'}\n\n"
        f"**3. 식약처 DUR 조회 결과**: {dur_msg}\n\n"
        "**4. 안내**: 동일 성분 또는 같은 계열(예: 소염진통제)의 약을 함께 복용하면 부작용 위험이 커질 수 있습니다. "
        "정확한 병용 가능 여부는 약사에게 확인하세요."
    )


def _generate_with_auto_fallback(profile, messages):
    """경량(GGUF) 모드에 필요한 llama-cpp-python이 없으면 같은 2.4B bf16 모델로 자동 대체합니다."""
    try:
        return _generate_with_profile(profile, messages)
    except ImportError:
        if profile["backend"] != "gguf":
            raise
        fb = next(
            v for v in MODEL_PROFILES.values()
            if v["backend"] == "torch" and "2.4B" in v["short"]
        )
        text = _generate_with_profile(fb, messages)
        return (
            "ℹ️ 경량(Q4) 실행에 필요한 llama-cpp-python이 설치되어 있지 않아 "
            "같은 2.4B 모델(bf16)로 대신 실행했습니다.\n\n" + text
        )


def analyze_with_local_llm(
    ocr_text: str,
    base_drug: str,
    second_drug: str,
    base_ingredients: list,
    dur_msg: str,
    profile_key: str = DEFAULT_PROFILE_KEY,
    second_ingredients=None
) -> str:
    """OCR 텍스트, 복용 약물, 식약처 DUR 데이터를 종합하여 복약 지도를 생성합니다. 말미에 의료 면책 문구가 항상 붙습니다."""
    profile = MODEL_PROFILES.get(profile_key, MODEL_PROFILES[DEFAULT_PROFILE_KEY])
    class_text = build_class_facts(base_ingredients, second_drug, second_ingredients)
    dur_msg_plain = dur_msg.split(" (병용금기 여부는")[0]

    if profile["backend"] == "none":
        return (
            build_rule_based_report(base_drug, second_drug, base_ingredients, dur_msg_plain)
            + "\n\n**약리 계열 확인 (사전 기준)**\n" + class_text
            + "\n\n---\n" + MEDICAL_DISCLAIMER
        )

    messages = _make_messages(ocr_text, base_drug, second_drug, base_ingredients, dur_msg, class_text)

    t_gen = time.time()
    try:
        report = _generate_with_auto_fallback(profile, messages)
    except Exception as e:
        # 모델 적재/추론 실패 시에도 서비스가 멈추지 않도록 규칙 기반 리포트로 대체
        print(f"[LLM 오류 -> 규칙 기반 대체] {e}", flush=True)
        report = (
            f"⚠️ LLM 실행에 실패하여 규칙 기반 안내로 대체합니다. (원인: {e})\n\n"
            + build_rule_based_report(base_drug, second_drug, base_ingredients, dur_msg_plain)
        )
    print(f"[LLM 추론] 완료 ({time.time() - t_gen:.1f}초, {profile['short']})", flush=True)

    return (
        report
        + "\n\n**약리 계열 확인 (사전 기준 · 자동 검증)**\n" + class_text
        + "\n\n---\n" + MEDICAL_DISCLAIMER
    )


def _make_messages(ocr_text, base_drug, second_drug, base_ingredients, dur_msg, class_text):
    """LLM에 줄 프롬프트(RAG: 사전·DUR 근거를 먼저 넣고 그 근거로만 답하게 함)를 만듭니다."""
    system_prompt = (
        "당신은 복약 정보를 쉽게 풀어 설명하는 보조 AI입니다. 의사나 약사를 대체하지 않습니다. "
        "제공된 [OCR 추출 텍스트], [복용 약물 정보], [식약처 DUR 공식 조회 결과], [약리 계열 근거]만을 근거로 "
        "아래 3가지 항목으로 답변하세요. 각 항목은 2문장 이내, 전체 10줄 이내로 아주 간결하게 쓰세요.\n\n"
        "1. 🔍 [OCR 약품 및 주성분 판독]: OCR 오탈자를 문맥으로 보정하여 식별된 약품명과 주성분을 정리하세요.\n"
        "2. ⚠️ [성분 중복 및 약리학적 계열 충돌 분석]: 성분이 직접 중복되는지, 동일 약리 계열(예: NSAIDs)인지 분석하세요.\n"
        "3. 💡 [환자 맞춤 3줄 복약 지도]: 주의사항을 일상 용어로 3줄 요약하세요.\n\n"
        "반드시 지킬 규칙:\n"
        "- 질병 진단, 처방, 용량 변경 지시는 절대 하지 마세요.\n"
        "- 제공된 근거에 없는 내용은 지어내지 말고 '확인 불가'라고 쓰세요.\n"
        "- 약리 계열(예: NSAIDs 여부)은 [약리 계열 근거]에 적힌 내용만 사용하고, 기억으로 추측하지 마세요.\n"
        "- 확실하지 않으면 안전한 쪽(복용 전 약사 상담)으로 안내하세요.\n"
        "- 마지막 줄에 '정확한 복용 여부는 반드시 의사 또는 약사와 상담하세요.'를 쓰세요."
    )

    user_prompt = (
        f"- [이미지 OCR 추출 텍스트]: {ocr_text if ocr_text else '업로드된 이미지 없음'}\n"
        f"- [기준 약품명]: {base_drug} (등록 성분: {', '.join(base_ingredients) if base_ingredients else '사전 미등록'})\n"
        f"- [함께 복용하려는 약품명]: {second_drug if second_drug else '미입력'}\n"
        f"- [식약처 DUR 병용금기 조회 결과]: {dur_msg}\n"
        f"- [약리 계열 근거(사전 정의)]:\n{class_text}\n\n"
        "위 근거 데이터를 바탕으로 복약 분석 리포트를 작성해주세요."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _ensure_loaded(profile):
    """모델 적재(최초 1회만 오래 걸림)를 분리해서 시간을 따로 잴 수 있게 합니다."""
    if profile["backend"] == "gguf":
        get_gguf_llm(profile["repo_id"], profile["filename"])
    elif profile["backend"] == "torch":
        get_torch_llm(profile["repo_id"])


def _stream_with_profile(profile, messages):
    """생성되는 글자를 조각 단위로 내보냅니다. (항상 greedy)"""
    max_new = profile["max_new_tokens"]

    if profile["backend"] == "gguf":
        llm = get_gguf_llm(profile["repo_id"], profile["filename"])
        for chunk in llm.create_chat_completion(
            messages=messages, max_tokens=max_new, temperature=0.0,
            repeat_penalty=1.05, stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta
        return

    import threading
    from transformers import TextIteratorStreamer

    tokenizer, model = get_torch_llm(profile["repo_id"])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_device = next(model.parameters()).device
    model_inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(target_device)

    eos_ids = [tokenizer.eos_token_id]
    endoftext_id = tokenizer.convert_tokens_to_ids("[|endoftext|]")
    if isinstance(endoftext_id, int) and endoftext_id not in eos_ids and endoftext_id >= 0:
        eos_ids.append(endoftext_id)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    error = {}

    def _run():
        try:
            with torch.inference_mode():
                model.generate(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    max_new_tokens=max_new,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                    use_cache=True,
                    streamer=streamer,
                )
        except Exception as e:           # 스트림이 영원히 대기하지 않도록 종료 신호를 보냄
            error["e"] = e
            streamer.text_queue.put(streamer.stop_signal)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join()
    if "e" in error:
        raise error["e"]


def analyze_with_local_llm_stream(
    ocr_text, base_drug, second_drug, base_ingredients, dur_msg,
    profile_key=DEFAULT_PROFILE_KEY, timing=None, second_ingredients=None
):
    """
    스트리밍 버전: 글자가 만들어지는 대로 내보냅니다. (체감 대기시간 감소)
    timing(dict)에 load(모델 적재), first_token(첫 글자까지), total(전체 생성) 초 단위 기록.
    """
    timing = timing if timing is not None else {}
    profile = MODEL_PROFILES.get(profile_key, MODEL_PROFILES[DEFAULT_PROFILE_KEY])
    class_text = build_class_facts(base_ingredients, second_drug, second_ingredients)
    dur_msg_plain = dur_msg.split(" (병용금기 여부는")[0]
    tail = "\n\n**약리 계열 확인 (사전 기준 · 자동 검증)**\n" + class_text + "\n\n---\n" + MEDICAL_DISCLAIMER

    if profile["backend"] == "none":
        timing.update(load=0.0, first_token=0.0, total=0.0)
        yield (
            build_rule_based_report(base_drug, second_drug, base_ingredients, dur_msg_plain)
            + "\n\n**약리 계열 확인 (사전 기준)**\n" + class_text
            + "\n\n---\n" + MEDICAL_DISCLAIMER
        )
        return

    messages = _make_messages(ocr_text, base_drug, second_drug, base_ingredients, dur_msg, class_text)

    prof = profile
    if profile["backend"] == "gguf":
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            prof = next(
                v for v in MODEL_PROFILES.values()
                if v["backend"] == "torch" and "2.4B" in v["short"]
            )
            yield (
                "ℹ️ 경량(Q4) 실행에 필요한 llama-cpp-python이 설치되어 있지 않아 "
                "같은 2.4B 모델(bf16)로 대신 실행했습니다.\n\n"
            )

    try:
        t0 = time.time()
        _ensure_loaded(prof)
        timing["load"] = time.time() - t0

        t1 = time.time()
        first = True
        for piece in _stream_with_profile(prof, messages):
            if first:
                timing["first_token"] = time.time() - t1
                first = False
            yield piece
        timing["total"] = time.time() - t1
    except Exception as e:
        print(f"[LLM 오류 -> 규칙 기반 대체] {e}", flush=True)
        yield (
            f"\n\n⚠️ LLM 실행에 실패하여 규칙 기반 안내로 대체합니다. (원인: {e})\n\n"
            + build_rule_based_report(base_drug, second_drug, base_ingredients, dur_msg_plain)
        )
    yield tail


# 테스트용 의약품 정보
DRUG_INFO = {
    "타이레놀": {
        "ingredients": ["아세트아미노펜"],
        "description": "해열 및 진통 목적으로 사용되는 의약품입니다.",
        "warning": "아세트아미노펜을 포함한 다른 의약품과 성분 중복 여부를 확인해야 합니다."
    },

    "아스피린": {
        "ingredients": ["아세틸살리실산"],
        "description": "해열·진통 등에 사용되는 의약품입니다.",
        "warning": "다른 의약품과 함께 사용할 경우 상호작용 및 주의사항을 확인해야 합니다."
    },

    "이지엔6": {
        "ingredients": ["제품별 주성분 확인 필요"],
        "description": "제품 종류에 따라 주성분이 다른 진통 관련 의약품입니다.",
        "warning": "정확한 제품명을 확인하여 주성분을 확인해야 합니다."
    },

    "아로나민": {
        "ingredients": ["비타민B군"],
        "description": "비타민 보충을 목적으로 사용하는 제품입니다.",
        "warning": "다른 영양제와 함께 사용하는 경우 동일 영양성분의 중복 여부를 확인해야 합니다."
    },

    "이부프로펜": {
        "ingredients": ["이부프로펜"],
        "description": "해열·진통·소염 목적으로 사용되는 의약품입니다.",
        "warning": "다른 소염진통제와 함께 사용하는 경우 성분 및 상호작용을 확인해야 합니다."
    },

    "나프록센": {
        "ingredients": ["나프록센"],
        "description": "해열·진통·소염 목적으로 사용되는 의약품입니다.",
        "warning": "다른 소염진통제와 함께 사용하는 경우 상호작용 여부를 확인해야 합니다."
    },

    "이소티논": {
        "ingredients": ["이소트레티노인"],
        "description": "여드름 치료에 사용되는 의약품입니다.",
        "warning": "다른 의약품과 함께 사용하는 경우 상호작용 및 주의사항을 확인해야 합니다."
    },

    "지르텍": {
        "ingredients": ["세티리진"],
        "description": "알레르기 증상 완화에 사용하는 항히스타민제입니다.",
        "warning": "다른 항히스타민제와 함께 사용하는 경우 주의사항을 확인해야 합니다."
    },

    "게보린": {
        "ingredients": ["아세트아미노펜", "이소프로필안티피린", "카페인"],
        "description": "해열·진통 목적으로 사용하는 복합 성분 의약품입니다.",
        "warning": "제품의 정확한 성분표를 확인하고 다른 진통제와의 성분 중복 여부를 확인해야 합니다."
    },

    "판콜": {
        "ingredients": ["제품별 복합 성분"],
        "description": "감기 증상 완화를 목적으로 사용하는 복합 성분 의약품입니다.",
        "warning": "제품 종류에 따라 성분이 다를 수 있으므로 정확한 제품명을 확인해야 합니다."
    }
}


DRUG_LIST = list(DRUG_INFO.keys())


# OCR 성분 사전
INGREDIENT_DATABASE = {
    "아세트아미노펜": [
        "아세트아미노펜",
        "acetaminophen",
        "paracetamol"
    ],

    "아세틸살리실산": [
        "아세틸살리실산",
        "아스피린",
        "acetylsalicylic acid",
        "aspirin"
    ],

    "이부프로펜": [
        "이부프로펜",
        "ibuprofen"
    ],

    "나프록센": [
        "나프록센",
        "naproxen"
    ],

    "이소트레티노인": [
        "이소트레티노인",
        "isotretinoin",
        "이소티논"
    ],

    "세티리진": [
        "세티리진",
        "cetirizine"
    ],

    "이소프로필안티피린": [
        "이소프로필안티피린",
        "isopropylantipyrine"
    ],

    "카페인": [
        "카페인",
        "caffeine"
    ],

    "비타민B군": [
        "비타민b",
        "vitamin b",
        "비타민 b"
    ]
}


def is_valid_api_key(api_key):
    if not api_key:
        return False

    if api_key.strip() == "":
        return False

    return True


# PaddleOCR: 이미지 전처리 및 글자 인식
def extract_text_from_image(uploaded_file):
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 작은 라벨 글씨 확대
        height, width = image_cv.shape[:2]
        longest_side = max(height, width)
        if longest_side < 1600:
            scale = min(3.0, 1600 / longest_side)
            image_cv = cv2.resize(
                image_cv, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # 원본과 대비 개선 이미지를 각각 인식
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(gray)
        sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        )
        sharpened = cv2.filter2D(clahe, -1, sharpen_kernel)
        enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        ocr_engine = get_ocr_engine()
        candidates = []

        for processed in (image_cv, enhanced_bgr):
            results = ocr_engine.predict(processed)

            for result in results:
                result_data = getattr(result, "json", None)
                if callable(result_data):
                    result_data = result_data()

                if not isinstance(result_data, dict):
                    continue

                result_data = result_data.get("res", result_data)
                texts = result_data.get("rec_texts", []) or []
                scores = result_data.get("rec_scores", []) or []

                for index, recognized_text in enumerate(texts):
                    recognized_text = str(recognized_text).strip()
                    if not recognized_text:
                        continue

                    if index < len(scores):
                        try:
                            if float(scores[index]) < 0.25:
                                continue
                        except (TypeError, ValueError):
                            pass

                    candidates.append(recognized_text)

        # 중복 문장 제거
        unique_lines = []
        for line in candidates:
            if line not in unique_lines:
                unique_lines.append(line)

        if not unique_lines:
            return (
                "OCR_ERROR: PaddleOCR가 글자를 인식하지 못했습니다. "
                "사진을 밝고 선명하게 촬영한 뒤 다시 시도하세요."
            )

        def candidate_score(candidate):
            try:
                has_drug_name = bool(detect_drug_name_from_text(candidate))
            except Exception:
                has_drug_name = False
            korean_count = len(re.findall(r"[가-힣]", candidate))
            readable_count = len(re.sub(r"\s+", "", candidate))
            return (has_drug_name, korean_count, readable_count)

        # 약품명 사전에 일치하는 줄을 우선 배치
        matching_lines = []
        for line in unique_lines:
            try:
                if detect_drug_name_from_text(line):
                    matching_lines.append(line)
            except Exception:
                pass

        if matching_lines:
            unique_lines.sort(
                key=lambda line: (
                    line in matching_lines,
                    candidate_score(line)
                ),
                reverse=True
            )

        return "\n".join(unique_lines)

    except Exception as e:
        return (
            "OCR_ERROR: PaddleOCR 실행 중 오류가 발생했습니다. "
            "PaddleOCR/PaddlePaddle 설치 및 버전을 확인하세요. "
            f"상세 오류: {e}"
        )


# OCR 결과에서 약품명 찾기
DRUG_ALIASES = {
    "타이레놀": [
        "타이레놀", "타이레놀정", "타이레놀500mg",
        "타이레놀 500mg", "TYLENOL"
    ],
    "아스피린": ["아스피린", "ASPIRIN"],
    "이지엔6": ["이지엔6", "이지엔", "EZN6"],
    "아로나민": ["아로나민", "ARONAMIN"],
    "이부프로펜": ["이부프로펜", "IBUPROFEN"],
    "나프록센": ["나프록센", "NAPROXEN"],
    "이소티논": ["이소티논", "ISOTRETINOIN"],
    "지르텍": ["지르텍", "ZYRTEC"],
    "게보린": ["게보린", "GEVORIN"],
    "판콜": ["판콜", "PANCOLD"]
}


def normalize_drug_text(text):
    # 공백, 구두점, 대소문자 차이를 정리
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def detect_drug_name_from_text(text):

    detected_drugs = []
    normalized_text = normalize_drug_text(text)

    for drug_name in DRUG_LIST:
        aliases = DRUG_ALIASES.get(
            drug_name, [drug_name]
        )

        for alias in aliases:
            normalized_alias = normalize_drug_text(alias)

            if (
                normalized_alias
                and normalized_alias in normalized_text
            ):
                if drug_name not in detected_drugs:
                    detected_drugs.append(drug_name)
                break

    return detected_drugs


# OCR 결과에서 성분 찾기
def extract_ingredients_from_text(text):

    found_ingredients = []

    text_lower = text.lower()

    for ingredient, keywords in INGREDIENT_DATABASE.items():

        for keyword in keywords:

            if keyword.lower() in text_lower:

                if ingredient not in found_ingredients:
                    found_ingredients.append(ingredient)

                break

    return found_ingredients


# 약품명으로 성분 찾기
def get_drug_ingredients(drug_name):

    if drug_name not in DRUG_INFO:
        return []

    return DRUG_INFO[drug_name]["ingredients"]


# 두 약품 성분 비교
def compare_ingredients(drug_a, drug_b):

    ingredients_a = get_drug_ingredients(drug_a)
    ingredients_b = get_drug_ingredients(drug_b)

    if not ingredients_a or not ingredients_b:
        return []

    conflicts = []

    for ingredient in ingredients_a:

        if ingredient in ingredients_b:

            if ingredient not in conflicts:
                conflicts.append(ingredient)

    return conflicts


# OCR 성분과 선택 약품 성분 비교
def compare_ocr_with_drug(
    ocr_ingredients,
    drug_name
):

    drug_ingredients = get_drug_ingredients(
        drug_name
    )

    conflicts = []

    for ingredient in ocr_ingredients:

        if ingredient in drug_ingredients:

            if ingredient not in conflicts:
                conflicts.append(ingredient)

    return conflicts


_EASY_URL = (
    "https://apis.data.go.kr/1471000/"
    "DrbEasyDrugInfoService/getDrbEasyDrugList"
)
_EASY_CACHE = {}         # 성공한 조회만 저장 (오류는 저장하지 않음)


def fetch_drugs_from_api(
    keyword,
    api_key,
    num_rows=4
):
    """
    식약처 e약은요 조회.
    반환: (약품 목록, 오류 메시지). 오류가 없으면 오류 메시지는 None입니다.
    """

    if not is_valid_api_key(api_key):
        return [], "API 인증이 필요합니다. (사이드바 비밀번호 / DUR_API_KEY 환경변수)"

    cache_key = (keyword, num_rows)
    cached = _EASY_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < 3600:
        return cached[1], None

    query = urllib.parse.urlencode(
        {"itemName": keyword, "type": "json", "numOfRows": num_rows},
        quote_via=urllib.parse.quote,
    )

    try:
        response = None
        for key_part in _dur_key_variants(api_key):
            response = requests.get(
                f"{_EASY_URL}?serviceKey={key_part}&{query}", timeout=10
            )
            if response.status_code == 200:
                break
            if "SERVICE_KEY_IS_NOT_REGISTERED" not in (response.text or ""):
                break

        if response.status_code != 200:
            return [], _dur_http_error(response, "e약은요")

        try:
            data = response.json()
        except ValueError:
            return [], (
                "e약은요 응답을 해석하지 못했습니다. "
                "인증키 등록(활용신청 승인) 상태를 확인하세요."
            )

        root = data.get("response", data) if isinstance(data, dict) else {}
        header = root.get("header", {}) or {}
        code = str(header.get("resultCode", "00"))
        if code not in ("00", "0"):
            return [], (
                f"e약은요 API 오류 (resultCode {code}: "
                f"{header.get('resultMsg', '')})"
            )

        items = (root.get("body", {}) or {}).get("items", [])
        if isinstance(items, dict):
            items = items.get("item", items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []

        extracted_drugs = []

        for item in items:
            extracted_drugs.append({
                "itemSeq": item.get("itemSeq", ""),
                "itemName": item.get("itemName", "제품명 미확인"),
                "itemImage": (
                    item.get("itemImage")
                    or "https://via.placeholder.com/200x150?text=No+Image"
                ),
                "efcyQesitm": item.get("efcyQesitm", "효능 정보 없음"),
                "useMethodQesitm": item.get("useMethodQesitm", "용법 정보 없음"),
                "intrcQesitm": item.get("intrcQesitm", "상호작용 정보 없음"),
                "atpnWarnQesitm": item.get("atpnWarnQesitm", ""),
                "entpName": item.get("entpName", "제조사 미확인"),
            })

        if extracted_drugs:
            _EASY_CACHE[cache_key] = (time.time(), extracted_drugs)
            return extracted_drugs, None

        return [], None          # 오류는 아니지만 검색 결과 없음

    except requests.exceptions.Timeout:
        return [], "e약은요 서버 응답 시간이 초과되었습니다. 인터넷 연결을 확인하세요."
    except Exception as e:
        return [], f"e약은요 조회 중 오류가 발생했습니다: {e}"


_DUR_URL = (
    "https://apis.data.go.kr/1471000/"
    "DURPrdlstInfoService03/"
    "getUsjntTabooInfoList03"
)
_DUR_CACHE = {}          # 성공한 조회만 저장 (오류는 저장하지 않음)
_DUR_CACHE_TTL = 3600


def _dur_key_variants(api_key):
    """인증키 표기(Encoding/Decoding)가 달라도 시도할 수 있도록 후보를 만듭니다."""
    k = api_key.strip()
    raw = urllib.parse.unquote(k)
    variants = []
    for cand in (urllib.parse.quote(raw, safe=""), k):
        if cand and cand not in variants:
            variants.append(cand)
    return variants


def _dur_get(api_key, item_name, page, rows):
    query = urllib.parse.urlencode(
        {
            "typeName": "병용금기",
            "itemName": item_name,
            "type": "json",
            "numOfRows": rows,
            "pageNo": page,
        },
        quote_via=urllib.parse.quote,
    )
    last = None
    for key_part in _dur_key_variants(api_key):
        last = requests.get(f"{_DUR_URL}?serviceKey={key_part}&{query}", timeout=10)
        if last.status_code == 200:
            return last
        if "SERVICE_KEY_IS_NOT_REGISTERED" not in (last.text or ""):
            return last      # 키 문제가 아니면 다른 표기로 재시도할 필요 없음
    return last


def _dur_http_error(response, label="DUR"):
    detail = ""
    try:
        head = response.json().get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {})
        parts = [str(head.get("errMsg", "")), str(head.get("returnAuthMsg", ""))]
        detail = " / ".join(p for p in parts if p)
    except Exception:
        detail = (response.text or "").strip().replace("\n", " ")[:120]
    hint = ""
    if "NOT_REGISTERED" in (response.text or ""):
        hint = " → data.go.kr에 등록(활용신청 승인)된 인증키인지, 키 값이 정확한지 확인하세요."
    if "ACCESS_DENIED" in (response.text or "") or "ACCESS DENIED" in (response.text or ""):
        hint = " → 이 서비스의 활용신청이 승인되었는지 data.go.kr에서 확인하세요."
    return f"{label} 서버 응답 실패 (HTTP {response.status_code})" + (f" - {detail}" if detail else "") + hint


def _dur_fetch_all(api_key, item_name, max_pages=10, rows=100):
    """
    DUR 병용금기 API를 페이지 끝까지 조회합니다.
    반환: (items, error_message). 오류 시 items=None.
    - HTTP 200이어도 header.resultCode가 00이 아니면 오류로 처리합니다. (인증키 오류가 '정보 없음'으로 보이는 문제 방지)
    - 한 약물에 병용금기 항목이 100건을 넘어도 모두 조회합니다.
    """
    items_all = []

    for page in range(1, max_pages + 1):
        response = _dur_get(api_key, item_name, page, rows)

        if response.status_code != 200:
            return None, _dur_http_error(response)

        try:
            data = response.json()
        except ValueError:
            return None, (
                "DUR 응답을 해석하지 못했습니다. "
                "인증키 등록(활용신청 승인) 상태를 확인하세요."
            )

        root = data.get("response", data) if isinstance(data, dict) else {}
        header = root.get("header", {}) or {}
        code = str(header.get("resultCode", "00"))

        if code not in ("00", "0"):
            return None, (
                f"DUR API 오류 (resultCode {code}: "
                f"{header.get('resultMsg', '')})"
            )

        body = root.get("body", {}) or {}
        items = body.get("items", [])

        if isinstance(items, dict):
            items = items.get("item", items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []

        items_all.extend(items)

        try:
            total = int(body.get("totalCount", 0) or 0)
        except (TypeError, ValueError):
            total = 0

        if not items or len(items_all) >= total:
            break

    return items_all, None


def _dur_name_variants(drug_name):
    """제품명 + 등록 성분명(예: 타이레놀 -> 아세트아미노펜)을 매칭 후보로 만듭니다."""
    names = {drug_name.strip().replace(" ", "")}
    try:
        for ing in get_drug_ingredients(drug_name):
            names.add(str(ing).strip().replace(" ", ""))
    except Exception:
        pass
    return {n for n in names if len(n) >= 2}


def _dur_find_match(items, target_names):
    """items 중 '병용금기 상대 약물'이 target_names와 겹치는 항목을 찾습니다."""
    for item in items:
        candidates = [
            str(item.get("MIXTURE_ITEM_NAME", "") or ""),
            str(item.get("MIXTURE_INGR_KOR_NAME", "") or ""),
        ]
        for cand in candidates:
            cand_clean = cand.strip().replace(" ", "")
            if len(cand_clean) < 2:      # 빈 문자열이 항상 '포함'으로 판정되던 오탐 방지
                continue
            for target in target_names:
                if target in cand_clean or cand_clean in target:
                    return item
    return None


def check_dur_contraindication(
    drug_name_a,
    drug_name_b,
    api_key
):
    """
    식약처 DUR 병용금기를 양방향(A->B, B->A)으로 조회합니다.
    반환 dict: has_conflict, risk_level, message, status("ok" | "error" | "no_key" | "no_input")
    """

    if not is_valid_api_key(api_key):
        return {
            "has_conflict": False,
            "risk_level": "인증 필요",
            "status": "no_key",
            "message": (
                "DUR API 인증키가 없어 병용금기 조회를 하지 못했습니다. "
                "(Streamlit: 사이드바 비밀번호 입력 / 서버: DUR_API_KEY 환경변수 설정)"
            )
        }

    if not drug_name_a or not drug_name_b:
        return {
            "has_conflict": False,
            "risk_level": "미입력",
            "status": "no_input",
            "message": "비교할 두 약물의 이름을 모두 입력해주세요."
        }

    cache_key = (drug_name_a.strip(), drug_name_b.strip())
    cached = _DUR_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _DUR_CACHE_TTL:
        return cached[1]

    names_a = _dur_name_variants(drug_name_a)
    names_b = _dur_name_variants(drug_name_b)

    try:
        for query_name, target_names in (
            (drug_name_a, names_b),
            (drug_name_b, names_a),
        ):
            items, err = _dur_fetch_all(api_key, query_name)

            if err:
                return {
                    "has_conflict": False,
                    "risk_level": "조회 실패",
                    "status": "error",
                    "message": err
                }

            hit = _dur_find_match(items, target_names)

            if hit:
                reason = hit.get("PROHBT_CONTENT", "병용 금기 약물입니다.")
                result = {
                    "has_conflict": True,
                    "risk_level": "위험 (병용금기)",
                    "status": "ok",
                    "message": (
                        f"[{drug_name_a}]와(과) "
                        f"[{drug_name_b}]는 식약처 DUR "
                        f"병용금기 데이터에 등록되어 있습니다.\n\n"
                        f"사유: {reason}"
                    )
                }
                _DUR_CACHE[cache_key] = (time.time(), result)
                return result

        result = {
            "has_conflict": False,
            "risk_level": "등록 정보 없음",
            "status": "ok",
            "message": (
                f"식약처 DUR에서 "
                f"[{drug_name_a}]와(과) "
                f"[{drug_name_b}] 간 등록된 "
                f"병용금기 정보를 찾지 못했습니다. "
                f"(양방향 조회, 등록 정보 없음이 안전을 의미하지는 않습니다)"
            )
        }
        _DUR_CACHE[cache_key] = (time.time(), result)
        return result

    except requests.exceptions.Timeout:
        return {
            "has_conflict": False,
            "risk_level": "조회 시간 초과",
            "status": "error",
            "message": "DUR 서버 응답 시간이 초과되었습니다."
        }

    except Exception as e:
        return {
            "has_conflict": False,
            "risk_level": "오류",
            "status": "error",
            "message": f"DUR 점검 중 오류가 발생했습니다: {e}"
        }


def dur_debug(item_name, api_key):
    """진단용: DUR API가 실제로 무엇을 돌려주는지 요약합니다."""
    if not is_valid_api_key(api_key):
        return {"error": "DUR 인증키가 없습니다."}
    items, err = _dur_fetch_all(api_key, item_name, max_pages=1)
    if err:
        return {"error": err}
    return {
        "fetched": len(items),
        "sample_keys": sorted(items[0].keys()) if items else [],
        "sample_mixtures": [
            f"{i.get('MIXTURE_ITEM_NAME', '')} / {i.get('MIXTURE_INGR_KOR_NAME', '')}"
            for i in items[:5]
        ],
    }


def predict_drug_and_conflict(drug_name):

    info = DRUG_INFO.get(drug_name)

    if info is None:

        return drug_name, {
            "has_conflict": False,
            "risk_level": "정보 없음",
            "message": "등록된 의약품 정보가 없습니다.",
            "synergy_message": "공공 DB에서 정보를 확인해주세요."
        }

    return drug_name, {
        "has_conflict": False,
        "risk_level": "확인 필요",
        "message": (
            f"주성분: {', '.join(info['ingredients'])}\n\n"
            f"{info['warning']}"
        ),
        "synergy_message": info["description"]
    }


