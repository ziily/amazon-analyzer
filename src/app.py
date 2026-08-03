"""
Amazon 产品竞争力分析 v3.1 (增强版)
===================================
核心改进：
1. 双语检测：标题/五点同时识别英德关键词
2. 否定词处理：避免误报"缺少"实际为"无需"的属性
3. 高频词参考：从数据集优秀产品中提取常用词，指导用户补充
4. 具体改写模板：针对五点描述给出可复制的改写示例
5. 缓存优化：高频词统计只计算一次
6. 新增：从标题或 attributes 提取套装数量，计算单价并显示在前端
7. 价格评分、竞品匹配、价格建议均改用「单价」而非「总价」
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
import time
import re
import hashlib
import gc
from io import BytesIO
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
import torch
import numpy as np

# ---------- 可选依赖：rembg（如果未安装，降级为无外观模式） ----------
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    remove = None
    HAS_REMBG = False
    print("⚠️ rembg 未安装，外观匹配功能将禁用。如需启用请 pip install rembg")

# ==================== 页面配置 ====================
st.set_page_config(page_title="Amazon 产品竞争力分析 v3.1", layout="wide", page_icon="📊")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stProgress > div > div {background-color: #FF9900;}
</style>
""", unsafe_allow_html=True)

# ==================== 配置 ====================
CONFIG = {
    "max_images_per_product": 3,
    "image_download_workers": 6,
    "analysis_cache_dir": "data/analysis_cache",
}


def dataset_hash(data):
    try:
        s = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:16]
    except Exception:
        return str(int(time.time()))


# ==================== 分类映射 ====================
CLASSIFICATION = {
    "搜索页信息": ["title", "url", "asin", "thumbnailImage", "price", "listPrice",
                   "stars", "reviewsCount", "delivery", "fastestDelivery",
                   "isAmazonChoice", "amazonChoiceText", "categoryPageData",
                   "unNormalizedProductUrl", "input"],
    "listing信息": ["originalAsin", "inStock", "inStockText", "brand", "author",
                    "breadCrumbs", "videosCount", "visitStoreLink", "galleryThumbnails",
                    "highResolutionImages", "importantInformation", "sustainabilityFeatures",
                    "description", "features", "attributes", "productOverview",
                    "variantAsins", "variantDetails", "variantAttributes",
                    "manufacturerAttributes", "condition", "returnPolicy", "support",
                    "aPlusContent", "brandStory", "bookDescription", "locationText",
                    "loadedCountryCode"],
    "评论信息": ["reviewsLink", "starsBreakdown", "aiReviewsSummary",
                 "productPageReviews", "productPageReviewsFromOtherCountries",
                 "reviewImages", "totalRatings", "customerId"]
}


def classify_raw_data(products):
    classified = {"搜索页信息": [], "listing信息": [], "评论信息": []}
    for prod in products:
        search_item = {k: prod.get(k) for k in CLASSIFICATION["搜索页信息"] if k in prod}
        listing_item = {k: prod.get(k) for k in CLASSIFICATION["listing信息"] if k in prod}
        review_item = {k: prod.get(k) for k in CLASSIFICATION["评论信息"] if k in prod}
        for k, v in prod.items():
            if k not in search_item and k not in listing_item and k not in review_item:
                listing_item[k] = v
        classified["搜索页信息"].append(search_item)
        classified["listing信息"].append(listing_item)
        classified["评论信息"].append(review_item)
    return classified


# ==================== 关键词评分系统（扩充词库）====================
PSYCH_KEYWORDS = {
    "quality": ["premium", "hochwertig", "qualität", "profi", "erstklassig", "top quality", "geprüft", "exzellent",
                "spitzenqualität"],
    "convenience": ["einfach", "mühelos", "schnell", "bequem", "praktisch", "easy", "handlich", "komfortabel"],
    "cost_saving": ["sparen", "günstig", "preiswert", "bestes preis", "value", "save", "discount", "kostengünstig"],
    "safety": ["sicher", "stabil", "robust", "rutschfest", "safety", "kindersicher", "bruchsicher"],
    "social_status": ["luxus", "exklusiv", "designer", "elegant", "premium design", "lifestyle", "edel"],
    "health": ["gesund", "ergonomisch", "atmungsaktiv", "schadstofffrei", "natural", "bio", "öko",
               "allergikerfreundlich"],
    "durability": ["langlebig", "dauerhaft", "robust", "widerstandsfähig", "garantie", "rostfrei", "strapazierfähig"],
    "aesthetics": ["schön", "elegant", "modern", "stilvoll", "design", "ästhetisch", "chic", "zeitlos"],
    "innovation": ["innovativ", "neuartig", "einzigartig", "smart", "intelligent", "revolutionär", "fortschrittlich"],
    "trust": ["vertrauen", "garantie", "zertifiziert", "geprüft", "trusted", "official", "geprüfte sicherheit"],
}


def keyword_psych_score(text):
    if not text or len(text.strip()) < 3:
        return 0.0
    text_lower = text.lower()
    hit_dims = 0
    for dim, keywords in PSYCH_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            hit_dims += 1
    diversity_score = (hit_dims / len(PSYCH_KEYWORDS)) * 60
    length_score = min(len(text) / 100, 1.0) * 40
    return round(diversity_score + length_score, 2)


def batch_psych_scores_keyword(texts):
    return {t: keyword_psych_score(t) for t in texts if t and len(t.strip()) >= 3}


# ==================== 信息点检测（双语合并 + 否定词处理）====================
INFO_PATTERNS = {
    "material": {
        "en": r"leather|fabric|wood|metal|plastic|steel|oak|velvet|linen|pu|abs|cotton|polyester|faux leather|vegan leather|solid wood|engineered wood",
        "de": r"leder|stoff|holz|metall|kunststoff|stahl|eiche|samt|leinen|pu|abs|baumwolle|polyester|kunstleder|massivholz|werkstoff"
    },
    "color": {
        "en": r"black|white|grey|gray|brown|blue|red|green|beige|navy|charcoal|cream|gold|silver|burgundy|teal",
        "de": r"schwarz|weiß|grau|braun|blau|rot|grün|beige|marine|anthrazit|creme|gold|silber|burgunder|türkis"
    },
    "size": {
        "en": r"\d+[\'\"]|\d+\s?(inch|cm|mm|ft)|\d+\s?(lb|kg|g)|\d+\s?set of \d+|\d+\s?pack",
        "de": r"\d+\s?(cm|mm|m|kg|g|l|ml)|\d+\s?stück|\d+\s?set|\d+\s?pack"
    },
    "function": {
        "en": r"foldable|stackable|swivel|adjustable|ergonomic|reclining|portable|waterproof|breathable|space-saving|easy-clean",
        "de": r"faltbar|stapelbar|drehbar|verstellbar|ergonomisch|neigbar|tragbar|wasserdicht|atmungsaktiv|platzsparend|leicht zu reinigen"
    },
    "scenario": {
        "en": r"kitchen|dining|office|living room|bedroom|outdoor|bar|home|work|study|garage|garden",
        "de": r"küche|esszimmer|büro|wohnzimmer|schlafzimmer|draußen|bar|zuhause|arbeit|studium|garage|garten"
    },
}

NEGATION_WORDS = ['no', 'not', 'without', 'frei', 'ohne', 'kein', 'keine', 'nicht', 'never', 'niemals']


def has_negation(text, pos=None, window=5):
    if not text:
        return False
    text_lower = text.lower()
    if pos is None:
        return any(neg in text_lower for neg in NEGATION_WORDS)
    words = text_lower.split()
    start = max(0, pos - window)
    end = min(len(words), pos + window + 1)
    return any(neg in ' '.join(words[start:end]) for neg in NEGATION_WORDS)


def check_info_points(text, language="de"):
    if not text:
        return {}
    text_lower = text.lower()
    hit = {}
    for dim, lang_dict in INFO_PATTERNS.items():
        en_pattern = lang_dict.get("en", "")
        de_pattern = lang_dict.get("de", "")
        if en_pattern and de_pattern:
            combined_pattern = f"({en_pattern})|({de_pattern})"
        else:
            combined_pattern = en_pattern or de_pattern
        match = re.search(combined_pattern, text_lower, re.IGNORECASE)
        if match:
            if has_negation(text_lower, pos=match.start()):
                hit[dim] = 0
            else:
                hit[dim] = 1
        else:
            hit[dim] = 0
    return hit


# ==================== 覆盖度检测 ====================
COVERAGE_KEYWORDS = {
    'de': {'size': ['größe', 'abmessungen', 'länge', 'breite', 'höhe', 'cm', 'mm', 'kg', 'g', 'ml', 'l', 'dimension'],
           'material': ['material', 'stoff', 'leder', 'kunststoff', 'holz', 'baumwolle', 'polyester', 'metall', 'samt',
                        'werkstoff'],
           'warranty': ['garantie', 'gewährleistung', '2-jährig', '1-jährig', 'lebenslang', 'rückgabe', 'zurückgeben'],
           'usage': ['verwendung', 'geeignet für', 'perfekt für', 'ideal für', 'szenario', 'einsatz', 'anwendbar'],
           'differentiation': ['einzigartig', 'exklusiv', 'nur', 'beste', 'anders als', 'vergleichen', 'überlegen',
                               'unique', 'exclusive'],
           'user_oriented': ['sie', 'ihnen', 'ihr', 'benutzer', 'kunde', 'komfort', 'genießen', 'erleben', 'ihre',
                             'ihnen']},
    'en': {'size': ['size', 'dimension', 'length', 'width', 'height', 'cm', 'mm', 'kg', 'g', 'ml', 'l', 'dimensions'],
           'material': ['material', 'fabric', 'leather', 'plastic', 'metal', 'wood', 'cotton', 'polyester', 'steel',
                        'velvet'],
           'warranty': ['warranty', 'guarantee', '2-year', '1-year', 'lifetime', 'return', 'money-back'],
           'usage': ['use', 'applicable', 'suitable for', 'perfect for', 'ideal for', 'scenario', 'application'],
           'differentiation': ['unique', 'exclusive', 'only', 'best', 'unlike', 'compare', 'superior', 'exceptional'],
           'user_oriented': ['you', 'your', 'user', 'customer', 'comfort', 'enjoy', 'experience', 'yourself']}
}
COVERAGE_WEIGHTS = {'size': 0.2, 'material': 0.2, 'warranty': 0.1, 'usage': 0.15,
                    'differentiation': 0.2, 'user_oriented': 0.15}


def check_coverage(text, language='de'):
    if not text or len(text.strip()) < 5:
        return {'coverage': {}, 'total_score': 0}
    text_lower = text.lower()
    coverage = {}
    for dim in COVERAGE_KEYWORDS['de'].keys():
        de_words = COVERAGE_KEYWORDS['de'].get(dim, [])
        en_words = COVERAGE_KEYWORDS['en'].get(dim, [])
        combined = list(dict.fromkeys(de_words + en_words))
        coverage[dim] = 1 if any(w in text_lower for w in combined) else 0
    total = sum(coverage[dim] * COVERAGE_WEIGHTS[dim] for dim in COVERAGE_WEIGHTS) * 100
    return {'coverage': coverage, 'total_score': round(total, 2)}


# ==================== CLIP 模型 ====================
@st.cache_resource
def load_clip():
    print("🖼️ 加载 CLIP 模型...")
    import torch
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.eval()
    print("✅ CLIP 加载完成")
    return model, processor

@st.cache_resource
def load_dinov2():
    print("📦 加载 DINOv2 外观模型...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()
    print("✅ DINOv2 加载完成")
    return processor, model

LABELS = {
    "attention": "a visually striking product photo that stands out in Amazon search results",
    "product_understanding": "a clear ecommerce product photo showing the product clearly",
    "value_perception": "a product image showing important benefits and product advantages",
    "usage_imagination": "a lifestyle product photo showing people using the product",
    "trust_signal": "a professional ecommerce product image that looks trustworthy",
    "risk_reduction": "a product image showing details features dimensions and information clearly",
    "quality_perception": "a premium high quality product photography",
    "differentiation": "a product image showing unique features compared with competitors",
    "purchase_intent": "an attractive product image that makes customers want to buy",
    "negative_baseline": "a blurry low quality product photo with bad lighting"
}
FEATURE_NAMES = list(LABELS.keys())[:-1]
TEXT_PROMPTS = list(LABELS.values())


def extract_product_appearance_embedding(image_bytes):
    """
    真正的“产品外观”向量：先抠掉背景，再用 DINOv2 提取形状/纹理特征
    如果 rembg 不可用或失败，返回 None
    """
    if not HAS_REMBG or remove is None:
        print("⚠️ rembg 未安装，跳过外观提取")
        return None
    try:
        # 1. 加载原图
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 2. 【核心】去除背景，只留产品本身
        img_no_bg = remove(img)  # rembg 会自动抠图

        # 3. 转为 RGB（rembg 输出可能带 alpha 通道，需要转）
        if img_no_bg.mode == 'RGBA':
            # 新建白底，把产品贴上去（保证模型输入统一）
            background = Image.new('RGB', img_no_bg.size, (255, 255, 255))
            background.paste(img_no_bg, mask=img_no_bg.split()[3])
            img_no_bg = background

        # 4. 加载 DINOv2 提取特征
        processor, model = load_dinov2()
        inputs = processor(images=img_no_bg, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            # 取 [CLS] token 或池化层
            embedding = outputs.pooler_output

        # 5. L2 归一化
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze().cpu().numpy()

    except Exception as e:
        print(f"⚠️ 外观向量提取失败: {e}")
        return None


def calculate_consumer_score(features, image_type):
    if image_type == "thumbnail":
        score = (features["attention"] * 0.35 + features["product_understanding"] * 0.30 +
                 features["quality_perception"] * 0.20 + features["differentiation"] * 0.15)
    elif image_type == "high_resolution":
        score = (features["value_perception"] * 0.30 + features["usage_imagination"] * 0.25 +
                 features["risk_reduction"] * 0.25 + features["trust_signal"] * 0.20)
    elif image_type == "a_plus":
        score = (features["trust_signal"] * 0.35 + features["quality_perception"] * 0.25 +
                 features["differentiation"] * 0.20 + features["value_perception"] * 0.20)
    else:
        score = 0
    return round(score * 100, 2)


def _load_image_from_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def analyze_uploaded_images_with_clip(uploaded_image_bytes_list, image_types_list):
    import torch
    if not uploaded_image_bytes_list:
        return {}
    model, processor = load_clip()
    per_image_results = []
    main_image_embedding = None
    main_image_index = None
    if 'main' in image_types_list:
        main_image_index = list(image_types_list).index('main')
    else:
        main_image_index = 0
    print(f"🖼️ 新品图片分析：共 {len(uploaded_image_bytes_list)} 张，主图索引 = {main_image_index}")
    for idx, (img_bytes, img_type) in enumerate(zip(uploaded_image_bytes_list, image_types_list)):
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"⚠️ 上传图片打开失败: {e}")
            continue
        try:
            inputs = processor(text=TEXT_PROMPTS, images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            positive = logits[:-1]
            baseline = logits[-1]
            scores_arr = torch.sigmoid(positive - baseline)
            features = {name: float(scores_arr[i]) for i, name in enumerate(FEATURE_NAMES)}
            consumer_type = {'main': 'thumbnail', 'detail': 'high_resolution',
                             'lifestyle': 'high_resolution', 'aplus': 'a_plus'}.get(img_type, 'high_resolution')
            consumer = calculate_consumer_score(features, consumer_type)
            per_image_results.append({'image_type': img_type, 'consumer_score': consumer, 'features': features})
            if idx == main_image_index:
                img_emb_tensor = outputs.image_embeds
                if img_emb_tensor is not None and hasattr(img_emb_tensor, 'shape'):
                    img_emb_tensor = img_emb_tensor / img_emb_tensor.norm(dim=-1, keepdim=True)
                    main_image_embedding = img_emb_tensor[0].cpu().numpy()
                    print(f"✅ 主图 embedding 提取成功，维度 = {main_image_embedding.shape}")
        except Exception as e:
            import traceback
            print(f"⚠️ CLIP 推理失败 (图 {idx}): {e}")
            traceback.print_exc()
            continue
    if not per_image_results:
        return {}
    type_weights = {'main': 0.40, 'detail': 0.30, 'lifestyle': 0.20, 'aplus': 0.10}
    total_weight = sum(type_weights.get(r['image_type'], 0.1) for r in per_image_results)
    weighted_score = sum(r['consumer_score'] * type_weights.get(r['image_type'], 0.1)
                         for r in per_image_results) / total_weight if total_weight > 0 else 0
    dim_avgs = {}
    for dim in FEATURE_NAMES:
        vals = [r['features'].get(dim, 0) * 100 for r in per_image_results]
        dim_avgs[dim] = sum(vals) / len(vals) if vals else 0
    type_scores = {}
    for t in ['main', 'detail', 'lifestyle', 'aplus']:
        sub = [r for r in per_image_results if r['image_type'] == t]
        type_scores[t] = {'score': sum(r['consumer_score'] for r in sub) / len(sub) if sub else 0, 'count': len(sub)}
    return {
        'overall_score': round(weighted_score, 2),
        'per_image': per_image_results,
        'dim_avgs': dim_avgs,
        'type_scores': type_scores,
        'image_count': len(per_image_results),
        'main_image_embedding': main_image_embedding,
        'main_image_index': main_image_index,
    }


@st.cache_data(show_spinner="🖼️ 计算数据集图片向量中（首次约 1-3 分钟，之后秒级）", ttl=3600 * 24 * 7)
def compute_dataset_image_embeddings(thumbnail_urls_tuple):
    import torch
    if not thumbnail_urls_tuple:
        return {}
    model, processor = load_clip()
    embeddings = {}

    def _download(url):
        try:
            resp = requests.get(url, timeout=10)
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None

    urls = [u for _, u in thumbnail_urls_tuple]
    with ThreadPoolExecutor(max_workers=CONFIG["image_download_workers"]) as executor:
        images = list(executor.map(_download, urls))
    batch_size = 8
    valid_records = [(asin, img) for (asin, _), img in zip(thumbnail_urls_tuple, images) if img is not None]
    print(f"🖼️ 数据集图片：{len(thumbnail_urls_tuple)} 个 ASIN，{len(valid_records)} 张下载成功")
    for i in range(0, len(valid_records), batch_size):
        batch = valid_records[i:i + batch_size]
        batch_imgs = [img for _, img in batch]
        batch_asins = [asin for asin, _ in batch]
        try:
            inputs = processor(text=["a product photo"], images=batch_imgs, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embs = outputs.image_embeds
            embs = embs / embs.norm(dim=-1, keepdim=True)
            for asin, emb in zip(batch_asins, embs):
                embeddings[asin] = emb.cpu().numpy()
        except Exception as e:
            import traceback
            print(f"⚠️ 批量 embedding 失败: {e}")
            traceback.print_exc()
            continue
        if (i + batch_size) % 32 == 0 or i + len(batch) >= len(valid_records):
            print(f"  embedding 进度 {min(i + batch_size, len(valid_records))}/{len(valid_records)}")
    print(f"✅ 数据集图片 embedding 完成，{len(embeddings)}/{len(thumbnail_urls_tuple)}")
    return embeddings


@st.cache_data(show_spinner="🖼️ 批量 CLIP 图片分析中（逐产品处理，内存优化）", ttl=3600 * 24 * 7)
def batch_analyze_images_with_clip(image_records_tuple):
    import torch
    if not image_records_tuple:
        return {}
    model, processor = load_clip()
    results = {}
    total_records = len(image_records_tuple)

    asin_to_urls = {}
    for asin, img_type, url in image_records_tuple:
        asin_to_urls.setdefault(asin, []).append((img_type, url))
    total_products = len(asin_to_urls)
    print(f"🖼️ 批量 CLIP 分析开始：{total_records} 张图片，{total_products} 个产品")

    processed = 0
    for asin, url_list in asin_to_urls.items():
        imgs = []
        types = []
        for img_type, url in url_list:
            img = _load_image_from_url(url)
            if img is not None:
                imgs.append(img)
                types.append(img_type)
        if not imgs:
            print(f"⚠️ 产品 {asin} 图片下载失败，跳过")
            continue

        try:
            inputs = processor(text=TEXT_PROMPTS, images=imgs, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits_per_image
            img_embeds = outputs.image_embeds
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            per_type_scores = {'thumbnail': [], 'high_resolution': [], 'a_plus': []}
            per_type_features = {'thumbnail': [], 'high_resolution': [], 'a_plus': []}

            for i, itype in enumerate(types):
                pos = logits[i][:-1]
                base = logits[i][-1]
                scores_arr = torch.sigmoid(pos - base)
                features = {name: float(scores_arr[j]) for j, name in enumerate(FEATURE_NAMES)}
                consumer = calculate_consumer_score(features, itype)
                per_type_scores[itype].append(consumer)
                per_type_features[itype].append(features)

            summary = {'asin': asin}
            if per_type_scores['thumbnail']:
                summary['thumbnail_score'] = float(np.mean(per_type_scores['thumbnail']))
                thumb_feats = per_type_features['thumbnail'][0]
                summary['thumbnail_attention'] = thumb_feats['attention'] * 100
                summary['thumbnail_purchase'] = thumb_feats['purchase_intent'] * 100
                summary['thumbnail_quality'] = thumb_feats['quality_perception'] * 100
            else:
                summary['thumbnail_score'] = 0
                summary['thumbnail_attention'] = summary['thumbnail_purchase'] = summary['thumbnail_quality'] = 0

            if per_type_scores['high_resolution']:
                summary['detail_score'] = float(np.mean(per_type_scores['high_resolution']))
                summary['detail_image_count'] = len(per_type_scores['high_resolution'])
                detail_feats_avg = {dim: float(np.mean([f[dim] for f in per_type_features['high_resolution']])) * 100
                                    for dim in FEATURE_NAMES}
                summary['detail_trust'] = detail_feats_avg['trust_signal']
                summary['detail_value'] = detail_feats_avg['value_perception']
                summary['detail_usage'] = detail_feats_avg['usage_imagination']
                summary['detail_risk'] = detail_feats_avg['risk_reduction']
            else:
                summary['detail_score'] = 0
                summary['detail_image_count'] = 0
                summary['detail_trust'] = summary['detail_value'] = summary['detail_usage'] = summary['detail_risk'] = 0

            if per_type_scores['a_plus']:
                summary['aplus_score'] = float(np.mean(per_type_scores['a_plus']))
                summary['aplus_count'] = len(per_type_scores['a_plus'])
                aplus_feats_avg = {dim: float(np.mean([f[dim] for f in per_type_features['a_plus']])) * 100
                                   for dim in FEATURE_NAMES}
                summary['aplus_trust'] = aplus_feats_avg['trust_signal']
                summary['aplus_quality'] = aplus_feats_avg['quality_perception']
                summary['aplus_value'] = aplus_feats_avg['value_perception']
                summary['aplus_brand'] = aplus_feats_avg['differentiation']
            else:
                summary['aplus_score'] = 0
                summary['aplus_count'] = 0
                summary['aplus_trust'] = summary['aplus_quality'] = summary['aplus_value'] = summary['aplus_brand'] = 0

            all_feats = []
            if per_type_features['thumbnail']: all_feats.extend(per_type_features['thumbnail'])
            if per_type_features['high_resolution']: all_feats.extend(per_type_features['high_resolution'])
            if per_type_features['a_plus']: all_feats.extend(per_type_features['a_plus'])
            dim_avgs = {}
            for dim in FEATURE_NAMES:
                dim_avgs[dim] = float(np.mean([f[dim] * 100 for f in all_feats])) if all_feats else 0
            summary['dim_avgs'] = dim_avgs
            summary['image_count'] = len(imgs)

            if 'thumbnail' in types:
                thumb_idx = types.index('thumbnail')
                summary['main_embedding'] = img_embeds[thumb_idx].cpu().numpy()
            elif len(img_embeds) > 0:
                summary['main_embedding'] = img_embeds[0].cpu().numpy()
            else:
                summary['main_embedding'] = None

            results[asin] = summary

        except Exception as e:
            print(f"⚠️ 产品 {asin} CLIP 分析失败: {e}")
            continue
        finally:
            del imgs, types, inputs, outputs, logits, img_embeds
            if 'summary' in locals():
                del summary

        processed += 1
        if processed % 10 == 0 or processed == total_products:
            print(f"  CLIP 进度 {processed}/{total_products}")

    print(f"✅ 批量 CLIP 分析完成，{len(results)}/{total_products} 个产品成功")
    return results


# ==================== 评分函数 ====================
def compute_title_score(title, text2score=None):
    if not title:
        return 0, {"length": 0, "psych": 0, "info": 0, "has_num": False, "has_unit": False, "hit": {}}
    length = len(title)
    if 60 <= length <= 180:
        len_score = 100
    elif 30 <= length < 60:
        len_score = 70
    elif 180 < length <= 250:
        len_score = 80
    elif length > 250:
        len_score = 50
    else:
        len_score = 40
    if text2score and title in text2score:
        psych_score = text2score[title]
    else:
        psych_score = keyword_psych_score(title)
    info_hit = check_info_points(title, language='de')
    hit_count = sum(info_hit.values())
    info_score = (hit_count / len(info_hit)) * 100
    total = len_score * 0.25 + psych_score * 0.50 + info_score * 0.25
    has_num = any(c.isdigit() for c in title)
    has_unit = any(u in title.lower() for u in ['cm', 'mm', 'kg', 'g', 'ml', 'l', 'w', 'h', 'stück'])
    total += (5 if has_num else 0) + (5 if has_unit else 0)
    total = round(min(total, 100), 2)
    details = {"length": len_score, "psych": round(psych_score, 2), "info": round(info_score, 2),
               "has_num": has_num, "has_unit": has_unit, "hit": info_hit}
    return total, details


def score_features_batch(features, text2score):
    if not features: return 0.0
    count = len(features)
    count_score = min(count / 5, 1.0) * 100
    avg_len = sum(len(f) for f in features) / count
    len_score = min(avg_len / 80, 1.0) * 100
    psycho_scores = [text2score.get(f, 0.0) for f in features]
    avg_psycho = sum(psycho_scores) / len(psycho_scores)
    max_psycho = max(psycho_scores)
    psycho_score = avg_psycho * 0.7 + max_psycho * 0.3
    total = count_score * 0.3 + len_score * 0.3 + psycho_score * 0.4
    return round(min(total, 100), 2)


def score_attributes(attributes):
    if not attributes: return 0.0
    count = len(attributes)
    count_score = min(count / 15, 1.0) * 100
    avg_len = sum(len(str(a.get("value", ""))) for a in attributes) / count
    len_score = min(avg_len / 20, 1.0) * 100
    has_number = any(any(c.isdigit() for c in str(a.get("value", ""))) for a in attributes)
    numeric_bonus = 10 if has_number else 0
    total = count_score * 0.5 + len_score * 0.4 + numeric_bonus
    return round(min(total, 100), 2)


def score_important(info):
    return 100.0 if info and info.get("items") else 0.0


def score_aplus(aplus):
    if not aplus: return 0.0
    modules = aplus.get("modules", []) if isinstance(aplus, dict) else []
    mod_score = min(len(modules) / 5, 1.0) * 60
    video_bonus = 0
    img_count = 0
    if isinstance(aplus, dict):
        video_bonus = min(len(aplus.get("rawVideos", [])), 2) * 15
        img_count = sum(1 for img in aplus.get("rawImages", []) if img.get("url"))
    img_bonus = min(img_count, 3) * 3
    return round(min(mod_score + video_bonus + img_bonus, 100), 2)


def score_video(count):
    if count is None or count == 0: return 0.0
    return 100.0 if count >= 3 else (90.0 if count == 2 else 70.0)


def score_brandstory(story):
    return 100.0 if story and story.get("items") else 0.0


def score_images_with_clip(asin, clip_result):
    if not clip_result:
        return 20.0
    thumb = clip_result.get('thumbnail_score', 0)
    thumb_purchase = clip_result.get('thumbnail_purchase', 0)
    main_score = thumb * 0.6 + thumb_purchase * 0.4
    detail_visual = clip_result.get('detail_score', 0)
    detail_risk = clip_result.get('detail_risk', 0)
    detail_score = detail_visual * 0.6 + detail_risk * 0.4
    aplus_mean = clip_result.get('aplus_score', 0)
    aplus_trust = clip_result.get('aplus_trust', 0)
    aplus_score = aplus_mean * 0.5 + aplus_trust * 0.5
    total = main_score * 0.40 + detail_score * 0.40 + aplus_score * 0.20
    return round(total, 2)


def score_images_simple(asin, listing):
    high_imgs = listing.get("highResolutionImages", []) or []
    gallery = listing.get("galleryThumbnails", []) or []
    aplus = listing.get("aPlusContent")
    aplus_count = 0
    if aplus:
        def _count_urls(obj):
            n = 0
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "url" and isinstance(v, str):
                        n += 1
                    else:
                        n += _count_urls(v)
            elif isinstance(obj, list):
                for item in obj: n += _count_urls(item)
            return n

        aplus_count = _count_urls(aplus)
    high_count = len(high_imgs)
    gallery_count = len(gallery)
    total_img = high_count + gallery_count + aplus_count
    count_score = min(40 + total_img * 7.5, 90) if total_img > 0 else 20
    diversity_bonus = 0
    if high_count > 0: diversity_bonus += 5
    if aplus_count > 0: diversity_bonus += 5
    return round(min(count_score + diversity_bonus, 100), 2)


# ==================== 评论分析 ====================
def analyze_reviews_fast(review_data):
    results = []
    for idx, rev in enumerate(review_data):
        reviews_link = rev.get("reviewsLink", "")
        asin = reviews_link.split("/")[-1].split("?")[0] if reviews_link else None
        if not asin: asin = f"Unknown_{idx + 1}"
        stars_br = rev.get("starsBreakdown") or {}
        avg_stars = (stars_br.get("5star", 0) * 5 + stars_br.get("4star", 0) * 4 +
                     stars_br.get("3star", 0) * 3 + stars_br.get("2star", 0) * 2 +
                     stars_br.get("1star", 0) * 1)
        ai_summary = rev.get("aiReviewsSummary") or {}
        keywords = ai_summary.get("keywords", [])
        pos_mentions = sum(kw.get("customersMentionedCount", {}).get("total", 0)
                           for kw in keywords if kw.get("sentiment") == "positive")
        neg_mentions = sum(kw.get("customersMentionedCount", {}).get("total", 0)
                           for kw in keywords if kw.get("sentiment") == "negative")
        total_mentions = pos_mentions + neg_mentions
        ai_pos_ratio = pos_mentions / total_mentions if total_mentions > 0 else 0.5
        all_reviews = []
        for r in rev.get("productPageReviews", []):
            desc = r.get("reviewDescription", "")
            rating = r.get("ratingScore")
            if desc and rating: all_reviews.append((desc, rating))
        for r in rev.get("productPageReviewsFromOtherCountries", []):
            desc = r.get("reviewDescription", "")
            rating = r.get("ratingScore")
            if desc and rating: all_reviews.append((desc, rating))
        total_comments = len(all_reviews)
        stars_norm = avg_stars / 5 if avg_stars else 0.5
        pos_norm = ai_pos_ratio if ai_pos_ratio is not None else 0.5
        count_norm = min(math.log(total_comments + 1) / math.log(1001), 1.0) if total_comments else 0.0
        five_star_ratio = stars_br.get("5star", 0)
        one_star_ratio = stars_br.get("1star", 0)
        sent_proxy = five_star_ratio - one_star_ratio
        sent_norm = (sent_proxy + 1) / 2
        w = [0.30, 0.25, 0.20, 0.15, 0.10]
        final_score = (stars_norm * w[0] + sent_norm * w[1] + pos_norm * w[2] +
                       count_norm * w[3] + ai_pos_ratio * w[4]) * 100
        results.append({"ASIN": asin, "Avg_Stars": round(avg_stars, 2),
                        "Total_Reviews_Count": total_comments,
                        "5star_ratio": round(stars_br.get("5star", 0) * 100, 1),
                        "1star_ratio": round(stars_br.get("1star", 0) * 100, 1),
                        "Positive%": round(ai_pos_ratio * 100, 1),
                        "Conversion_Score": round(final_score, 2)})
    return pd.DataFrame(results)


# ==================== 新增：从标题和 attributes 提取数量及单价 ====================
def extract_quantity_from_title(title):
    if not title:
        return None
    title_lower = title.lower()
    patterns = [
        r'(\d+)\s*er\s*set',
        r'(\d+)\s*stück',
        r'(\d+)\s*-pack',
        r'pack\s*of\s*(\d+)',
        r'(\d+)\s*pack',
        r'(\d+)\s*teilig',
        r'(\d+)\s*in\s*1',
        r'(\d+)\s*set',
        r'(\d+)\s*pc',
        r'(\d+)\s*piece',
        r'(\d+)\s*pieces',
    ]
    for pat in patterns:
        m = re.search(pat, title_lower)
        if m:
            return int(m.group(1))
    return None


def extract_quantity_from_attributes(attributes):
    if not attributes:
        return None
    quantity_keys = ['anzahl von einheiten', 'anzahl der teile', 'set-größe',
                     'anzahl', 'einheiten', 'stück', 'menge', 'number of units',
                     'pack quantity', 'count']
    for attr in attributes:
        key = attr.get('key', '').lower()
        if any(k in key for k in quantity_keys):
            value = attr.get('value', '')
            nums = re.findall(r'(\d+\.?\d*)', value)
            if nums:
                return float(nums[0])
    return None


def get_quantity_and_unit_price(price_value, title, attributes):
    quantity = extract_quantity_from_title(title)
    if quantity is None:
        quantity = extract_quantity_from_attributes(attributes)
    if quantity is None:
        quantity = 1
    else:
        quantity = int(quantity)
    if price_value and quantity > 0:
        unit_price = price_value / quantity
    else:
        unit_price = None
    return quantity, unit_price


# ==================== 主分析流水线 ====================
def run_fast_analysis(classified_data, limit=0, progress_callback=None, enable_clip=False):
    def _update(p, msg):
        if progress_callback: progress_callback(p, msg)

    total_start = time.time()
    if limit > 0:
        classified_data = {k: v[:limit] for k, v in classified_data.items()}
    data = classified_data
    search_list = data.get("搜索页信息", [])
    listing_list = data.get("listing信息", [])
    review_list = data.get("评论信息", [])
    print(f"📊 数据加载：搜索页 {len(search_list)}，详情页 {len(listing_list)}，评论 {len(review_list)}")
    _update(5, f"分类数据加载完成：{len(search_list)} 个产品")

    asin_to_attributes = {}
    for listing in listing_list:
        asin = listing.get("originalAsin")
        if asin:
            asin_to_attributes[asin] = listing.get("attributes", [])

    t_psych = time.time()
    all_features = []
    for listing in listing_list:
        feats = listing.get("features", []) or []
        all_features.extend(feats)
    for item in search_list:
        title = item.get("title")
        if title and len(title.strip()) >= 3: all_features.append(title)
    text2score = batch_psych_scores_keyword(all_features)
    print(f"⏱️ 心理评分耗时: {time.time() - t_psych:.2f} 秒")
    _update(15 if not enable_clip else 10, "心理评分完成")

    clip_results = {}
    if enable_clip:
        print("🖼️ 启用批量 CLIP 图片分析...")
        t_clip = time.time()
        image_records = []
        for product in listing_list:
            asin = product.get("originalAsin")
            high = (product.get("highResolutionImages") or [])[:2]
            for url in high:
                image_records.append({'asin': asin, 'image_type': 'high_resolution', 'image_url': url})
            aplus = product.get("aPlusContent")
            if aplus:
                def _extract_urls(obj):
                    urls = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k == "url" and isinstance(v, str):
                                urls.append(v)
                            else:
                                urls.extend(_extract_urls(v))
                    elif isinstance(obj, list):
                        for item in obj: urls.extend(_extract_urls(item))
                    return urls

                aplus_urls = _extract_urls(aplus)[:2]
                for url in aplus_urls:
                    image_records.append({'asin': asin, 'image_type': 'a_plus', 'image_url': url})
        for product in search_list:
            asin = product.get("asin")
            thumb = product.get("thumbnailImage")
            if thumb:
                image_records.append({'asin': asin, 'image_type': 'thumbnail', 'image_url': thumb})
        print(f"🖼️ 共提取 {len(image_records)} 张图片")
        _update(20, f"提取图片 {len(image_records)} 张，开始 CLIP 分析")
        try:
            clip_results = batch_analyze_images_with_clip(
                tuple([(r['asin'], r['image_type'], r['image_url']) for r in image_records]))
        except Exception as e:
            print(f"⚠️ 批量 CLIP 失败: {e}")
            clip_results = {}
        print(f"⏱️ 批量 CLIP 耗时: {time.time() - t_clip:.2f} 秒")
        _update(50, "CLIP 图片分析完成")
    else:
        _update(20, "图片评分（快速层）")

    t_search = time.time()

    # 构建有效单价列表（用于价格评分）
    valid_unit_prices = []
    valid_prices = []
    for item in search_list:
        asin = item.get("asin")
        title = item.get("title")
        price_data = item.get("price")
        if isinstance(price_data, dict):
            price = price_data.get("value")
        else:
            price = price_data
        if price is not None:
            valid_prices.append(price)
            attributes = asin_to_attributes.get(asin, [])
            qty, unit_p = get_quantity_and_unit_price(price, title, attributes)
            if unit_p is not None:
                valid_unit_prices.append(unit_p)
            else:
                valid_unit_prices.append(price)

    search_results = []
    for idx, item in enumerate(search_list):
        asin = item.get("asin")
        title = item.get("title")
        price_data = item.get("price")
        if isinstance(price_data, dict):
            price = price_data.get("value")
        else:
            price = price_data
        stars = item.get("stars") or 0
        reviews = item.get("reviewsCount") or 0
        position = (item.get("categoryPageData") or {}).get("productPosition")

        title_score_val, title_details = compute_title_score(title, text2score)

        if asin in clip_results:
            thumb_score = clip_results[asin].get('thumbnail_score', 50)
        else:
            thumb_score = 50.0

        # 提取数量和单价
        attributes = asin_to_attributes.get(asin, [])
        quantity, unit_price = get_quantity_and_unit_price(price, title, attributes)

        # 价格评分：基于单价
        if unit_price is not None and valid_unit_prices:
            rank = sum(x > unit_price for x in valid_unit_prices)
            price_score = round(rank / len(valid_unit_prices) * 100, 2)
        elif price is not None and valid_prices:
            rank = sum(x > price for x in valid_prices)
            price_score = round(rank / len(valid_prices) * 100, 2)
        else:
            price_score = 50

        rating = stars / 5 if stars else 0
        review_norm = min(math.log(reviews + 1) / math.log(100000), 1) if reviews > 0 else 0
        trust_score_val = round((rating * 0.6 + review_norm * 0.4) * 100, 2)
        pos_score = round(1 / math.log(position + 2) * 100, 2) if position and position > 0 else 0

        search_score = (title_score_val * 0.25 + thumb_score * 0.30 + price_score * 0.15 +
                        trust_score_val * 0.20 + pos_score * 0.10)

        search_results.append({
            "asin": asin,
            "title": (title or "")[:80],
            "title_score": title_score_val,
            "thumbnail_score": thumb_score,
            "price_score": price_score,
            "price_value": price,
            "unit_price": unit_price,
            "quantity": quantity,
            "trust_score": trust_score_val,
            "position_score": pos_score,
            "search_score": round(search_score, 2),
            "stars": stars,
            "reviews": reviews
        })
    search_df = pd.DataFrame(search_results)
    if not search_df.empty:
        search_df["search_rank"] = search_df["search_score"].rank(ascending=False, method="min")
    print(f"⏱️ 搜索页评分耗时: {time.time() - t_search:.2f} 秒")
    _update(70, "搜索页评分完成")

    t_list = time.time()
    listing_results = []
    for idx, listing in enumerate(listing_list):
        asin = listing.get("originalAsin")
        if not asin: continue
        features = listing.get("features", []) or []
        attributes = listing.get("attributes", []) or []
        important = listing.get("importantInformation")
        aplus = listing.get("aPlusContent")
        videos = listing.get("videosCount", 0)
        brand = listing.get("brandStory")
        feat_s = score_features_batch(features, text2score)
        attr_s = score_attributes(attributes)
        imp_s = score_important(important)
        aplus_s = score_aplus(aplus)
        vid_s = score_video(videos)
        if asin in clip_results:
            img_s = score_images_with_clip(asin, clip_results[asin])
        else:
            img_s = score_images_simple(asin, listing)
        brand_s = score_brandstory(brand)
        weights = {"features": 0.30, "attributes": 0.25, "important": 0.05,
                   "aplus": 0.10, "video": 0.05, "image": 0.25}
        base = (feat_s * weights["features"] + attr_s * weights["attributes"] +
                imp_s * weights["important"] + aplus_s * weights["aplus"] +
                vid_s * weights["video"] + img_s * weights["image"])
        brand_bonus = 10 if brand_s > 0 else 0
        search_row = search_df[search_df["asin"] == asin]
        if not search_row.empty:
            stars = search_row.iloc[0].get("stars", 0)
            reviews = search_row.iloc[0].get("reviews", 0)
        else:
            stars = reviews = 0
        trust_bonus = 0
        if stars >= 4.5:
            trust_bonus += 4
        elif stars >= 4.2:
            trust_bonus += 2
        if reviews > 1000:
            trust_bonus += 3
        elif reviews > 500:
            trust_bonus += 1.5
        elif reviews > 100:
            trust_bonus += 0.5
        final = min(base + brand_bonus + trust_bonus, 100)
        cov_scores = []
        for feat in features:
            cov = check_coverage(feat, language='de')
            cov_scores.append(cov['total_score'])
        avg_cov = np.mean(cov_scores) if cov_scores else 0
        listing_results.append({"asin": asin, "brand": listing.get("brand", ""),
                                "bullet_score": feat_s, "attributes_score": attr_s,
                                "important_score": imp_s, "aplus_score": aplus_s,
                                "video_score": vid_s, "image_score": img_s,
                                "brand_bonus": brand_bonus, "trust_bonus": round(trust_bonus, 2),
                                "listing_score": round(final, 2), "coverage_score": round(avg_cov, 2)})
    listing_df = pd.DataFrame(listing_results)
    if not listing_df.empty:
        listing_df["listing_rank"] = listing_df["listing_score"].rank(ascending=False, method="min")
    print(f"⏱️ Listing 评分耗时: {time.time() - t_list:.2f} 秒")
    _update(85, "详情页评分完成")

    t_rev = time.time()
    review_df = analyze_reviews_fast(review_list)
    print(f"⏱️ 评论分析耗时: {time.time() - t_rev:.2f} 秒")
    _update(95, "评论分析完成")

    merged = pd.merge(search_df, listing_df, on="asin", how="inner")
    merged = pd.merge(merged, review_df, left_on="asin", right_on="ASIN", how="inner")
    if 'ASIN' in merged.columns: merged.drop(columns=['ASIN'], inplace=True)
    merged["Detail_Conversion"] = 0.6 * merged["listing_score"] + 0.4 * merged["Conversion_Score"]
    merged["Total_Score"] = 0.5 * merged["search_score"] + 0.5 * merged["Detail_Conversion"]
    merged = merged.sort_values("Total_Score", ascending=False)
    final_cols = ["asin", "title", "brand", "search_score", "Detail_Conversion", "Total_Score",
                  "listing_score", "Conversion_Score", "search_rank", "price_value", "unit_price",
                  "quantity", "stars", "reviews"]
    final_df = merged[[c for c in final_cols if c in merged.columns]]
    _update(100, "分析完成！")
    print(f"✅ 全部分析完成！总耗时 {time.time() - total_start:.2f} 秒")
    return final_df, merged, clip_results


# ==================== 智能新品分析 ====================
def compute_quantiles(series, qs=(0.25, 0.5, 0.75)):
    s = pd.Series(series).dropna()
    if len(s) == 0: return {q: 50 for q in qs}
    return {q: float(s.quantile(q)) for q in qs}


def find_top_competitors(full_df, new_unit_price, new_title, top_n=3,
                         new_appearance_emb=None, dataset_appearance_embeddings=None,
                         title_weight=0.40, appearance_weight=0.40, price_weight=0.20):
    if full_df.empty:
        return full_df.head(top_n), pd.DataFrame()
    df = full_df.copy()

    # 1. 标题相似度（Jaccard）
    if new_title:
        new_words = set(re.findall(r'\w+', new_title.lower()))
        def jaccard(title):
            if not title or not isinstance(title, str):
                return 0
            words = set(re.findall(r'\w+', title.lower()))
            if not words:
                return 0
            return len(new_words & words) / len(new_words | words)
        df['title_sim'] = df['title'].apply(jaccard) if 'title' in df.columns else 0
    else:
        df['title_sim'] = 0

    # 2. 价格相似度（基于单价）
    if new_unit_price is not None:
        if 'unit_price' in df.columns and df['unit_price'].notna().sum() > 1:
            price_col = 'unit_price'
            prices = df['unit_price'].dropna()
        else:
            price_col = 'price_value'
            prices = df['price_value'].dropna()
        if len(prices) > 1:
            price_range = max(prices.max() - prices.min(), 1.0)
            df['price_sim'] = 1 - (df[price_col].fillna(df['price_value']).fillna(new_unit_price) - new_unit_price).abs() / price_range
        else:
            df['price_sim'] = 0.5
    else:
        df['price_sim'] = 0.5

    # 3. 外观相似度（如果提供了向量）
    if new_appearance_emb is not None and dataset_appearance_embeddings:
        def get_appearance_sim(asin):
            emb = dataset_appearance_embeddings.get(asin)
            if emb is None:
                return 0.0
            return float(np.dot(new_appearance_emb, emb))
        df['appearance_sim'] = df['asin'].apply(get_appearance_sim)
        # 填充缺失值
        if df['appearance_sim'].notna().any():
            median_sim = df['appearance_sim'].median()
            df['appearance_sim'] = df['appearance_sim'].fillna(median_sim)
        else:
            df['appearance_sim'] = 0.0
    else:
        df['appearance_sim'] = 0.5  # 无外观时中性值

    # 4. 综合相似度
    total = title_weight + appearance_weight + price_weight
    df['sim'] = (df['title_sim'] * title_weight + df['appearance_sim'] * appearance_weight + df['price_sim'] * price_weight) / total
    top = df.nlargest(top_n, 'sim')
    return top, df


def _quantile_position(val, q_dict):
    p25, p50, p75 = q_dict[0.25], q_dict[0.5], q_dict[0.75]
    if val >= p75:
        return "🟢 前 25%"
    elif val >= p50:
        return "🟡 中上"
    elif val >= p25:
        return "🟠 中下"
    else:
        return "🔴 后 25%"


@st.cache_data(ttl=3600 * 24)
def get_top_keywords(df, column='title', top_n=10):
    if df.empty or column not in df.columns:
        return []
    stopwords = {'the', 'a', 'an', 'to', 'for', 'of', 'with', 'in', 'on', 'at', 'by', 'and', 'or', 'for', 'is', 'it',
                 'this', 'that', 'from',
                 'und', 'der', 'die', 'das', 'für', 'mit', 'von', 'zu', 'auf', 'bei', 'aus', 'ein', 'eine', 'einen',
                 'dem', 'den', 'des',
                 'als', 'auch', 'sich', 'nicht', 'ich', 'es', 'sie', 'wir', 'euch', 'ihr', 'ihnen', 'euch', 'mein',
                 'dein', 'ihr', 'unser',
                 'euer', 'meine', 'deine', 'ihre', 'unsere', 'eure', 'ist', 'sind', 'war', 'waren', 'hat', 'haben',
                 'wurde', 'wurden',
                 'your', 'our', 'their', 'my', 'your', 'her', 'his', 'its', 'our', 'their'}
    texts = df[column].dropna().str.lower()
    all_words = ' '.join(texts).split()
    words = [w for w in all_words if re.match(r'^[a-zäöüß]+$', w) and len(w) > 2 and w not in stopwords]
    freq = Counter(words)
    return [w for w, c in freq.most_common(top_n)]


def _generate_smart_advice(new_title, new_features, features_list, feat_details, title_details,
                           title_score_val, price_score, trust_score_val, feat_avg, vid_s, aplus_score,
                           has_aplus, has_brandstory, video_count, new_stars, new_reviews,
                           benchmarks, competitors, img_avg=None, img_source='estimated',
                           image_analysis_result=None, full_df=None, new_unit_price=None):
    advice = []

    # ========== 标题建议 ==========
    title_q = benchmarks['title_score']
    if not new_title:
        advice.append(("❌", "标题", "标题为空，请填写完整标题（建议 60-180 字符，含品牌+核心关键词+规格）"))
    else:
        title_lower = new_title.lower()
        words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{3,}\b', new_title)
        has_brand = any(word.istitle() or word.isupper() for word in new_title.split()) and len(new_title.split()) > 2
        has_material = title_details['hit'].get('material', 0)
        has_color = title_details['hit'].get('color', 0)
        has_size = title_details['hit'].get('size', 0)
        has_function = title_details['hit'].get('function', 0)
        has_scenario = title_details['hit'].get('scenario', 0)
        has_emotion = any(kw in title_lower for dim, kws in PSYCH_KEYWORDS.items() for kw in kws)
        has_num = title_details['has_num']
        has_unit = title_details['has_unit']

        issues = []
        suggestions = []

        if len(new_title) < 60:
            issues.append(f"标题过短（{len(new_title)} 字符），建议 60-180")
            suggestions.append("扩展描述：加入核心属性（材质/尺寸/功能）和使用场景")
        elif len(new_title) > 180:
            issues.append(f"标题过长（{len(new_title)}），可能被截断")
            suggestions.append("删除冗余修饰词，保留最核心的 3-4 个卖点")

        if not has_brand:
            issues.append("未检测到品牌名")
            suggestions.append("在标题开头加入品牌名（如 'BrandName Product'）")

        missing_info = []
        if not has_material: missing_info.append("材质")
        if not has_color: missing_info.append("颜色")
        if not has_size: missing_info.append("尺寸/容量")
        if not has_function: missing_info.append("功能")
        if not has_scenario: missing_info.append("使用场景")
        if missing_info:
            issues.append(f"缺少信息点：{', '.join(missing_info)}")
            suggestions.append(f"在标题中加入 {'、'.join(missing_info)} 相关关键词")

        if not has_emotion and len(new_title) > 30:
            issues.append("缺少情感触发词（如 premium, elegant）")
            suggestions.append("加入 1-2 个情感词，例如 'Premium Quality'")

        if not has_num and not has_unit:
            issues.append("缺少数字或单位（如 4er Set, 5kg）")
            suggestions.append("添加具体规格数字，增强说服力")

        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        duplicates = [w for w, c in word_freq.items() if c > 1 and len(w) > 3]
        if duplicates:
            issues.append(f"存在重复关键词：{', '.join(duplicates[:2])}")
            suggestions.append("精简重复词，使用同义词或属性词替代")

        if title_score_val < title_q[0.25]:
            if issues:
                advice.append(("🔴", "标题",
                               f"标题得分 {title_score_val}，处于后 25%。主要问题：{'；'.join(issues)}。优化建议：{'；'.join(suggestions[:3])}"))
            else:
                advice.append(("🔴", "标题",
                               f"标题得分 {title_score_val}，处于后 25%。建议参考优秀竞品标题结构：品牌 + 核心关键词 + 属性 + 场景 + 情感词"))
        elif title_score_val < title_q[0.5]:
            if issues:
                advice.append(("🟠", "标题",
                               f"标题得分 {title_score_val}，低于中位（{title_q[0.5]:.1f}）。建议：{'；'.join(suggestions[:2])}"))
            else:
                advice.append(("🟠", "标题", f"标题得分 {title_score_val}，可继续优化关键词覆盖和情感表达"))
        else:
            advice.append(
                ("🟢", "标题", f"标题得分 {title_score_val}，已超过中位水平。保持现有结构，可微调情感词增强吸引力"))

        if full_df is not None and not full_df.empty:
            top_words = get_top_keywords(full_df, 'title', 8)
            if top_words:
                advice.append(("💡", "优秀标题常用词参考", f"{', '.join(top_words)}"))

        if missing_info or not has_emotion:
            example_parts = []
            core = ' '.join([w for w in words if len(w) > 3][:3]) if words else "[产品核心名称]"
            if not has_brand:
                example_parts.append("[品牌名]")
            example_parts.append(core)
            if has_size == 0:
                example_parts.append("[尺寸/容量]")
            if has_material == 0:
                example_parts.append("[材质]")
            if has_color == 0:
                example_parts.append("[颜色]")
            if has_function == 0:
                example_parts.append("[功能]")
            if has_scenario == 0:
                example_parts.append("[适用场景]")
            if not has_emotion:
                example_parts.append("[情感词]")
            if example_parts:
                example_title = ' '.join(example_parts)
                if len(example_title) > 150:
                    example_title = example_title[:150] + '...'
                advice.append(("💡", "标题示例", f"可参考结构：{example_title}"))

    # ========== 五点描述建议 ==========
    bullet_q = benchmarks['bullet_score']
    if not features_list:
        advice.append(("❌", "五点描述",
                       "未填写五点描述，请至少提供 3-5 条卖点。建议每条卖点遵循 FAB 结构：Feature（特征）→ Advantage（优势）→ Benefit（利益）"))
    else:
        feature_keywords = ['material', 'size', 'weight', 'dimension', 'capacity', 'power', 'voltage', '材质', '尺寸',
                            '重量', '容量']
        advantage_keywords = ['easy', 'simple', 'fast', 'quick', 'effortless', '节省', '简化', '快速', '轻松']
        benefit_keywords = ['enjoy', 'experience', 'feel', 'relax', 'solve', 'avoid', 'prevent', '享受', '体验', '解决',
                            '避免']
        fab_scores = []
        for d in feat_details:
            text = d['text'].lower()
            has_f = any(kw in text for kw in feature_keywords)
            has_a = any(kw in text for kw in advantage_keywords)
            has_b = any(kw in text for kw in benefit_keywords)
            fab_scores.append(sum([has_f, has_a, has_b]))
        avg_fab = sum(fab_scores) / len(fab_scores) if fab_scores else 0

        if feat_avg < bullet_q[0.25]:
            advice.append(
                ("🔴", "五点描述", f"五点描述均分 {feat_avg:.1f}，处于后 25%（中位 {bullet_q[0.5]:.1f}）。逐条诊断："))
            for i, d in enumerate(feat_details):
                sub_issues = []
                sub_suggestions = []
                text = d['text']
                if len(text) < 50:
                    sub_issues.append(f"过短（{len(text)} 字符）")
                    sub_suggestions.append("扩展内容，加入具体数值或用户获益点")
                if avg_fab < 2:
                    if not any(kw in text.lower() for kw in feature_keywords):
                        sub_issues.append("缺少 Feature（特征参数）")
                        sub_suggestions.append("加入具体材质、尺寸或技术指标")
                    if not any(kw in text.lower() for kw in advantage_keywords):
                        sub_issues.append("缺少 Advantage（优势描述）")
                        sub_suggestions.append("说明此特征带来的便利，如 '易于清洁'")
                    if not any(kw in text.lower() for kw in benefit_keywords):
                        sub_issues.append("缺少 Benefit（用户利益）")
                        sub_suggestions.append("描述用户能获得什么好处，如 '享受舒适坐感'")
                if not d['has_user_mention']:
                    sub_issues.append("缺少用户导向词（Sie/Ihnen）")
                    sub_suggestions.append("使用第二人称，让顾客感觉被关注")
                if not d['has_differentiation']:
                    sub_issues.append("缺少差异化词（einzigartig/exklusiv）")
                    sub_suggestions.append("强调独特卖点，如 '独家设计'")
                missing_cov = [k for k, v in d['coverage'].items() if v == 0]
                if missing_cov:
                    cov_map = {'size': '尺寸', 'material': '材质', 'warranty': '保修', 'usage': '用途',
                               'differentiation': '差异化', 'user_oriented': '用户导向'}
                    missing_cn = '、'.join(cov_map.get(m, m) for m in missing_cov[:3])
                    sub_issues.append(f"缺信息维度：{missing_cn}")
                    sub_suggestions.append(f"补充 {missing_cn} 相关信息")
                if sub_issues:
                    advice.append(("📌", f"  第{i + 1}条",
                                   f"{d['text'][:60]}... → 问题：{'；'.join(sub_issues)}。建议：{'；'.join(sub_suggestions[:2])}"))
                    orig = d['text']
                    new_sentence = orig
                    if '材质' in missing_cn or not any(
                            kw in orig.lower() for kw in ['material', 'leder', 'holz', 'stahl', 'stoff']):
                        new_sentence += " Hergestellt aus hochwertigem [Material]."
                    if '尺寸' in missing_cn:
                        new_sentence += " Mit idealen Maßen von [Größe]."
                    if not d['has_user_mention']:
                        new_sentence += " So können Sie [Vorteil] genießen."
                    if not d['has_differentiation']:
                        new_sentence += " Einzigartig im Vergleich zu herkömmlichen Produkten."
                    advice.append(("✏️", f"  改写示例", f"原句：'{orig}'\n建议改写：'{new_sentence}'"))
                else:
                    advice.append(("✅", f"  第{i + 1}条", f"{d['text'][:60]}... → 结构良好，可保持"))
        elif feat_avg < bullet_q[0.5]:
            advice.append(("🟠", "五点描述", f"五点描述均分 {feat_avg:.1f}，低于中位（{bullet_q[0.5]:.1f}）。整体建议："))
            if avg_fab < 2:
                advice.append(("📌", "  结构", "多数卖点缺少完整的 FAB 结构，建议每条卖点按「特征→优势→利益」展开"))
            if not any(d['has_differentiation'] for d in feat_details):
                advice.append(("📌", "  差异化", "缺乏独特卖点，可加入竞品对比或独家功能描述"))
            if not any(d['has_user_mention'] for d in feat_details):
                advice.append(("📌", "  用户导向", "使用第二人称（Sie/Ihnen）拉近距离，增强说服力"))
        else:
            advice.append(("🟢", "五点描述", f"五点描述均分 {feat_avg:.1f}，超过中位水平。可进一步优化："))
            if avg_fab < 2:
                advice.append(("📌", "  细节", "部分卖点可强化 Benefit 描述，让顾客明确使用价值"))

    # ========== 价格建议：基于单价 ==========
    price_q = benchmarks['price_score']

    comp_unit_prices = []
    if not competitors.empty and 'unit_price' in competitors.columns:
        comp_unit_prices = competitors['unit_price'].dropna().tolist()
    if not comp_unit_prices and not competitors.empty and 'price_value' in competitors.columns:
        for _, c in competitors.iterrows():
            if c.get('price_value') and c.get('quantity'):
                comp_unit_prices.append(c['price_value'] / c['quantity'])
        if not comp_unit_prices:
            comp_unit_prices = competitors['price_value'].dropna().tolist()

    if price_score < price_q[0.25]:
        if comp_unit_prices:
            comp_avg = sum(comp_unit_prices) / len(comp_unit_prices)
            display_price = new_unit_price if new_unit_price is not None else new_title
            advice.append(("🔴", "价格",
                           f"单价 {display_price:.2f}€ 处于后 25%（百分位 {price_score}），最相似竞品均价 {comp_avg:.2f}€/件，建议降价至 {comp_avg * 0.95:.2f}€/件 以下"))
        else:
            advice.append(("🔴", "价格", f"价格竞争力处于后 25%（百分位 {price_score}），建议调价"))
    elif price_score > price_q[0.75]:
        advice.append(("🟢", "价格", f"价格优势明显（百分位 {price_score}，前 25%）"))
    else:
        advice.append(("🟡", "价格", f"价格处于中位水平（百分位 {price_score}）"))

    # ========== 信任 ==========
    trust_q = benchmarks['trust_score']
    if trust_score_val < trust_q[0.25]:
        advice.append(
            ("🔴", "信任", f"信任分 {trust_score_val}，处于后 25%。建议：1) Vine 计划；2) 30 天退货；3) 突出售后保障"))
    elif trust_score_val < trust_q[0.5]:
        advice.append(("🟠", "信任", f"信任分 {trust_score_val}，低于中位。建议加强售后政策展示"))
    else:
        advice.append(("🟢", "信任", f"信任分 {trust_score_val}，处于中上水平"))

    # ========== 视频 ==========
    video_q = benchmarks['video_score']
    if vid_s < video_q[0.25]:
        advice.append(("🔴", "视频", "无产品视频，建议制作 1 个 30 秒开箱演示 + 1 个使用场景视频"))
    elif vid_s < video_q[0.5]:
        advice.append(("🟠", "视频", "视频数量偏少，建议补充使用场景视频"))
    else:
        advice.append(("🟢", "视频", "视频覆盖充分"))

    # ========== A+ ==========
    if not has_aplus:
        advice.append(("🟠", "A+ 内容", "未启用 A+ 内容，建议品牌备案后开通，加入品牌故事、对比模块、Q&A 模块"))
    else:
        advice.append(("🟢", "A+ 内容", "已启用 A+ 内容，建议加入竞品对比表和场景应用模块"))

    # ========== 图片 ==========
    img_q = benchmarks['image_score']
    if img_source == 'uploaded' and image_analysis_result:
        overall = image_analysis_result.get('overall_score', 0)
        dim_avgs = image_analysis_result.get('dim_avgs', {})
        img_count = image_analysis_result.get('image_count', 0)
        if overall < img_q[0.25]:
            advice.append(("🔴", "图片", f"图片综合得分 {overall:.1f}（{img_count} 张已分析），处于后 25%。具体诊断："))
        elif overall < img_q[0.5]:
            advice.append(("🟠", "图片", f"图片综合得分 {overall:.1f}（{img_count} 张已分析），低于中位水平。具体诊断："))
        elif overall < img_q[0.75]:
            advice.append(("🟡", "图片", f"图片综合得分 {overall:.1f}（{img_count} 张已分析），中上水平"))
        else:
            advice.append(("🟢", "图片", f"图片综合得分 {overall:.1f}（{img_count} 张已分析），前 25%"))
        dim_thresholds = [('attention', '注意力', 55, '主图不够抓眼，建议：白底高清+产品占图 85%'),
                          ('trust_signal', '信任感', 55, '信任感不足，建议：增加专业拍摄+品牌 LOGO'),
                          ('quality_perception', '品质感', 55, '品质感弱，建议：提升光线+特写材质纹理'),
                          ('value_perception', '价值感', 50, '价值感不强，建议：增加功能标注图/赠品展示'),
                          ('usage_imagination', '使用场景', 50, '缺场景图，建议：至少 1 张生活场景图'),
                          ('risk_reduction', '风险消除', 50, '风险信号弱，建议：增加尺寸标注图/功能拆解图'),
                          ('differentiation', '差异化', 50, '差异化不足，建议：增加竞品对比图')]
        weak_dims = []
        for dim_key, dim_name, threshold, suggestion in dim_thresholds:
            val = dim_avgs.get(dim_key, 0) * 100
            if val < threshold:
                weak_dims.append((dim_name, val, suggestion))
        if weak_dims:
            for dim_name, val, suggestion in weak_dims[:4]:
                advice.append(("📌", f"  图片·{dim_name}", f"{val:.1f}/100。{suggestion}"))
    else:
        advice.append(("ℹ️", "图片", f"未上传图片，图片维度用数据集中位数 {img_q[0.5]:.1f} 作为预估"))

    # ========== 竞品 ==========
    if not competitors.empty:
        comp_advice_lines = []
        for _, c in competitors.iterrows():
            title_short = (c.get('title') or '')[:50]
            unit_price_str = f"{c.get('unit_price', 'N/A'):.2f}€" if c.get('unit_price') else "N/A"
            comp_advice_lines.append(
                f"  • ASIN {c['asin']} | {title_short} | 单价 {unit_price_str} | 总分 {c.get('Total_Score', 0):.1f}")
        best_comp = competitors.iloc[0]
        new_total = (title_score_val + feat_avg + trust_score_val) / 3
        comp_total = (best_comp.get('title_score', 0) + best_comp.get('bullet_score', 0) +
                      best_comp.get('trust_score', 0)) / 3
        if comp_total > new_total:
            gap_dims = []
            if best_comp.get('bullet_score', 0) > feat_avg + 10: gap_dims.append("五点描述更具体")
            if best_comp.get('title_score', 0) > title_score_val + 10: gap_dims.append("标题关键词更全")
            if best_comp.get('image_score', 0) > 50: gap_dims.append("图片更丰富")
            if gap_dims:
                advice.append(("🎯", "竞品学习",
                               f"最相似竞品 {best_comp['asin']} 总分 {best_comp.get('Total_Score', 0):.1f}，可借鉴：{'、'.join(gap_dims)}"))
        advice.append(("📋", "Top3 竞品", "\n".join(comp_advice_lines)))

    return advice


# [CHANGE] 新增参数 new_appearance_emb 和 dataset_appearance_embeddings
def analyze_new_product_smart(new_title, new_price, new_stars, new_reviews,
                              new_features, has_aplus, has_brandstory,
                              video_count, full_df, new_quantity=1,
                              classifier_text2score=None,
                              image_analysis_result=None,
                              dataset_image_embeddings=None,
                              thumbnail_urls_map=None,
                              top_n_competitors=5,
                              new_appearance_emb=None,
                              dataset_appearance_embeddings=None):
    qs = (0.25, 0.5, 0.75)
    benchmarks = {
        'title_score': compute_quantiles(full_df.get('title_score', pd.Series()), qs),
        'price_score': compute_quantiles(full_df.get('price_score', pd.Series()), qs),
        'trust_score': compute_quantiles(full_df.get('trust_score', pd.Series()), qs),
        'bullet_score': compute_quantiles(full_df.get('bullet_score', pd.Series()), qs),
        'image_score': compute_quantiles(full_df.get('image_score', pd.Series()), qs),
        'video_score': compute_quantiles(full_df.get('video_score', pd.Series()), qs),
        'aplus_score': compute_quantiles(full_df.get('aplus_score', pd.Series()), qs),
        'search_score': compute_quantiles(full_df.get('search_score', pd.Series()), qs),
        'listing_score': compute_quantiles(full_df.get('listing_score', pd.Series()), qs),
        'Conversion_Score': compute_quantiles(full_df.get('Conversion_Score', pd.Series()), qs),
        'Total_Score': compute_quantiles(full_df.get('Total_Score', pd.Series()), qs),
    }

    # 计算单价
    if new_price is not None and new_quantity and new_quantity > 0:
        new_unit_price = new_price / new_quantity
    else:
        new_unit_price = new_price

    title_score_val, title_details = compute_title_score(new_title, classifier_text2score)

    # 价格得分：基于单价
    price_values = full_df['unit_price'].dropna().tolist() if 'unit_price' in full_df.columns else []
    if not price_values and 'price_value' in full_df.columns:
        price_values = full_df['price_value'].dropna().tolist()
    if price_values and new_unit_price:
        rank = sum(x > new_unit_price for x in price_values)
        price_score = round(rank / len(price_values) * 100, 2)
    else:
        price_score = 50

    rating = new_stars / 5 if new_stars else 0
    review_norm = min(math.log(new_reviews + 1) / math.log(100000), 1) if new_reviews > 0 else 0
    trust_score_val = round((rating * 0.6 + review_norm * 0.4) * 100, 2)

    features_list = [f.strip() for f in (new_features or '').split('\n') if f.strip()]
    if features_list:
        temp_text2score = batch_psych_scores_keyword(features_list)
        feat_scores = [score_features_batch([feat], temp_text2score) for feat in features_list]
        feat_avg = sum(feat_scores) / len(feat_scores)
        feat_details = []
        for feat in features_list:
            cov = check_coverage(feat, language='de')
            info_hit = check_info_points(feat, language='de')
            info_score = (sum(info_hit.values()) / len(info_hit)) * 100
            feat_details.append({'text': feat, 'score': score_features_batch([feat], temp_text2score),
                                 'coverage': cov['coverage'], 'coverage_score': cov['total_score'],
                                 'info_hit': info_hit, 'info_score': round(info_score, 2),
                                 'has_user_mention': bool(re.search(r'\b(sie|ihnen|ihr|you|your)\b', feat.lower())),
                                 'has_differentiation': any(w in feat.lower() for w in
                                                            ['unique', 'different', 'exclusive', 'only', 'best',
                                                             'einzigartig', 'exklusiv', 'beste']),
                                 'length': len(feat)})
    else:
        feat_avg = 0
        feat_details = []

    vid_s = score_video(video_count)
    aplus_score = 50 if has_aplus else 0
    brand_bonus = 10 if has_brandstory else 0

    if image_analysis_result and image_analysis_result.get('overall_score') is not None:
        img_avg = image_analysis_result['overall_score']
        img_source = 'uploaded'
    else:
        img_avg = benchmarks['image_score'][0.5]
        img_source = 'estimated'

    attr_s = 50
    imp_s = 50
    weights = {"features": 0.30, "attributes": 0.25, "important": 0.05,
               "aplus": 0.10, "video": 0.05, "image": 0.25}
    base = (feat_avg * weights["features"] + attr_s * weights["attributes"] +
            imp_s * weights["important"] + aplus_score * weights["aplus"] +
            vid_s * weights["video"] + img_avg * weights["image"])
    listing_score = min(base + brand_bonus + trust_score_val * 0.1, 100)
    avg_thumb = benchmarks.get('thumbnail_score', {0.5: 50})[0.5] if 'thumbnail_score' in benchmarks else 50
    avg_position = 50
    search_score = (title_score_val * 0.25 + avg_thumb * 0.30 + price_score * 0.15 +
                    trust_score_val * 0.20 + avg_position * 0.10)
    conv_score = benchmarks['Conversion_Score'][0.5]
    detail_conversion = 0.6 * listing_score + 0.4 * conv_score
    total_score = 0.5 * search_score + 0.5 * detail_conversion

    # [CHANGE] 调用 find_top_competitors 并传入外观向量及权重
    competitors, all_ranked = find_top_competitors(
        full_df,
        new_unit_price,
        new_title or "",
        top_n=top_n_competitors,
        new_appearance_emb=new_appearance_emb,
        dataset_appearance_embeddings=dataset_appearance_embeddings,
        title_weight=0.40,      # 标题权重 40%
        appearance_weight=0.40, # 外观权重 40%
        price_weight=0.20       # 价格权重 20%
    )

    dim_rows = [('标题得分', title_score_val, 'title_score'), ('价格得分', price_score, 'price_score'),
                ('信任得分', trust_score_val, 'trust_score'), ('五点描述', round(feat_avg, 2), 'bullet_score'),
                ('图片得分', round(img_avg, 2), 'image_score'), ('视频得分', vid_s, 'video_score'),
                ('A+ 得分', aplus_score, 'aplus_score'), ('搜索得分', round(search_score, 2), 'search_score'),
                ('详情得分', round(listing_score, 2), 'listing_score'),
                ('转化得分', round(conv_score, 2), 'Conversion_Score'),
                ('综合总分', round(total_score, 2), 'Total_Score')]
    compare_df = pd.DataFrame([{'维度': name, '新品得分': val,
                                '数据集 P25': benchmarks[col][0.25], '数据集 P50（中位）': benchmarks[col][0.5],
                                '数据集 P75': benchmarks[col][0.75], 'vs 中位': round(val - benchmarks[col][0.5], 2),
                                '位置': _quantile_position(val, benchmarks[col])} for name, val, col in dim_rows])

    advice = _generate_smart_advice(new_title, new_features, features_list, feat_details, title_details,
                                    title_score_val, price_score, trust_score_val, feat_avg, vid_s, aplus_score,
                                    has_aplus, has_brandstory, video_count, new_stars, new_reviews,
                                    benchmarks, competitors, img_avg=img_avg, img_source=img_source,
                                    image_analysis_result=image_analysis_result, full_df=full_df,
                                    new_unit_price=new_unit_price)

    # [CHANGE] 更新 has_image_sim：基于外观向量是否存在
    has_image_sim = (new_appearance_emb is not None) and (dataset_appearance_embeddings is not None)

    return {'compare_df': compare_df, 'advice': advice, 'competitors': competitors,
            'scores': {'title': title_score_val, 'price': price_score, 'trust': trust_score_val,
                       'bullet': round(feat_avg, 2), 'image': round(img_avg, 2), 'video': vid_s,
                       'aplus': aplus_score, 'search': round(search_score, 2),
                       'listing': round(listing_score, 2), 'conversion': round(conv_score, 2),
                       'total': round(total_score, 2)},
            'feat_details': feat_details, 'title_details': title_details,
            'image_analysis': image_analysis_result, 'img_source': img_source,
            'all_ranked': all_ranked,
            'has_image_sim': has_image_sim}  # 正确赋值


# ==================== Streamlit UI ====================
st.title("📊 Amazon 产品竞争力分析 v3.1")
st.caption("🚀 快速层(关键词) + 深度层(CLIP 按需) | 新增：双语检测、高频词参考、改写示例、单价自动计算")

with st.sidebar:
    st.header("⚙️ 分析选项")
    max_items = st.number_input("🔢 分析前多少个产品？（0=全部）", min_value=0, max_value=500, value=10, step=1)
    use_cache = st.checkbox("💾 启用数据集缓存（推荐）", value=True)
    st.markdown("---")
    st.markdown("**图片分析模式**")
    enable_clip_batch = st.checkbox("🚀 批量分析时启用 CLIP（每产品 1主图+2详情+2A+图）",
                                    value=False,
                                    help="开启后图片得分基于真实 CLIP 分析，100 产品约 5-10 分钟。关闭时图片得分=50（固定），分析 30 秒完成。")
    if enable_clip_batch:
        st.caption("⏱️ 预计耗时：100 产品约 5-10 分钟（首次），已分析则秒级返回")
    else:
        st.caption("⏱️ 预计耗时：100 产品约 30 秒")
    st.markdown("---")
    enable_deep_single = st.checkbox("启用单 ASIN CLIP 深度分析", value=False,
                                     help="开启后，选择 ASIN 时会加载 CLIP 模型分析图片（每个 ASIN 约 5-10 秒）")
    st.markdown("---")
    st.markdown("**关于性能**")
    st.markdown("- 快速层：纯关键词+数字，无 NLP 模型")
    st.markdown("- 批量 CLIP：每产品 5 张图，真实视觉评分")
    st.markdown("- 内存占用 < 500MB")

uploaded_file = st.file_uploader("上传原始爬虫 JSON 文件 (dataset_free-amazon-product-scraper_*.json)", type=["json"])

if uploaded_file is not None:
    raw_data = json.load(uploaded_file)
    st.success(f"✅ 文件上传成功，共 {len(raw_data)} 条原始数据")
    if isinstance(raw_data, list) and not ("搜索页信息" in raw_data[0] if raw_data else False):
        with st.spinner("正在执行数据分类..."):
            classified_data = classify_raw_data(raw_data)
        st.success("✅ 分类完成")
    else:
        classified_data = raw_data
    cache_key = f"{dataset_hash(classified_data)}_{max_items}_{enable_clip_batch}"
    cache_file = os.path.join(CONFIG["analysis_cache_dir"], f"v3_{cache_key}.parquet")
    final_df = None
    full_df = None
    clip_results = {}
    if use_cache and os.path.exists(cache_file):
        st.info("💾 命中数据集缓存，秒级加载...")
        cached = pd.read_parquet(cache_file)
        if not cached.empty:
            final_df = cached[cached['asin'].notna()][['asin', 'title', 'brand', 'search_score',
                                                       'Detail_Conversion', 'Total_Score', 'listing_score',
                                                       'Conversion_Score', 'search_rank', 'price_value',
                                                       'unit_price', 'quantity', 'stars',
                                                       'reviews']].copy() if 'asin' in cached.columns else cached
            full_df = cached
        st.toast("💾 已从缓存加载", icon="✅")
    if final_df is None:
        progress_bar = st.progress(0, text="开始分析...")
        status = st.empty()


        def progress_callback(p, msg):
            progress_bar.progress(p, text=msg)
            status.text(f"{msg} ({p}%)")


        with st.spinner("运行分析流水线..."):
            final_df, full_df, clip_results = run_fast_analysis(classified_data, max_items, progress_callback,
                                                                enable_clip=enable_clip_batch)
        progress_bar.empty()
        status.empty()
        if use_cache and full_df is not None and not full_df.empty:
            os.makedirs(CONFIG["analysis_cache_dir"], exist_ok=True)
            try:
                cols_to_save = [c for c in full_df.columns if not c.startswith('_')]
                full_df[cols_to_save].to_parquet(cache_file, index=False)
            except Exception as e:
                print(f"⚠️ 缓存保存失败: {e}")
    if final_df is None or final_df.empty:
        st.error("❌ 分析失败或无数据")
        st.stop()
    st.subheader("📋 产品竞争力排名")
    final_df_display = final_df.copy()
    final_df_display['商品链接'] = final_df_display['asin'].apply(lambda x: f'https://www.amazon.de/dp/{x}')
    cols = final_df_display.columns.tolist()
    order = ["asin", "title", "brand", "price_value", "unit_price", "quantity", "search_score", "Detail_Conversion",
             "Total_Score", "listing_score", "Conversion_Score", "search_rank", "stars", "reviews", "商品链接"]
    order = [c for c in order if c in cols]
    extra = [c for c in cols if c not in order]
    final_df_display = final_df_display[order + extra]
    st.dataframe(
        final_df_display,
        column_config={
            "商品链接": st.column_config.LinkColumn("详情页", display_text=r'^(https://www\.amazon\.de/dp/)(.*)'),
            "price_value": st.column_config.NumberColumn("总价 (€)", format="%.2f"),
            "unit_price": st.column_config.NumberColumn("单价 (€)", format="%.2f"),
            "quantity": st.column_config.NumberColumn("数量", format="%d"),
            "asin": None,
        },
        width='stretch',
        hide_index=True
    )
    csv = final_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 下载结果 CSV", data=csv, file_name="product_competitiveness_final.csv", mime="text/csv")
    st.subheader("📈 数据可视化")
    tab1, tab2, tab3 = st.tabs(["Top10 综合分", "维度趋势", "分布"])
    with tab1:
        st.bar_chart(final_df.head(10).set_index("asin")["Total_Score"])
        st.caption("Top10 综合总分")
    with tab2:
        chart_cols = ["search_score", "Detail_Conversion", "Total_Score"]
        st.area_chart(final_df[chart_cols].head(20))
        st.caption("各维度趋势（前 20）")
    with tab3:
        fig, ax = plt.subplots(figsize=(8, 4))
        final_df[chart_cols].boxplot(ax=ax)
        ax.set_title("得分分布")
        st.pyplot(fig)

    st.subheader("🔍 单品分析")
    asin_list = final_df["asin"].tolist()
    selected_asin = st.selectbox("选择 ASIN", asin_list)
    if selected_asin:
        row = full_df[full_df["asin"] == selected_asin].iloc[0]
        numeric_cols = ["search_score", "Detail_Conversion", "Total_Score", "listing_score", "Conversion_Score"]
        available_cols = [c for c in numeric_cols if c in full_df.columns]
        means = full_df[available_cols].mean()
        medians = full_df[available_cols].median()
        col_radar, col_table = st.columns([1, 1])
        with col_radar:
            st.markdown("**📊 雷达图**")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=[row[c] for c in available_cols], theta=available_cols, fill='toself',
                                          name=selected_asin))
            fig.add_trace(go.Scatterpolar(r=[medians[c] for c in available_cols], theta=available_cols, fill='toself',
                                          name='中位数'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.markdown("**📈 各维度详情**")
            detail_data = {'维度': available_cols, '本品': [row[c] for c in available_cols],
                           '中位': [medians[c] for c in available_cols],
                           '差值': [round(row[c] - medians[c], 2) for c in available_cols]}
            if 'unit_price' in row and 'quantity' in row:
                detail_data['维度'] += ['单价 (€)', '数量']
                detail_data['本品'] += [row['unit_price'], row['quantity']]
                detail_data['中位'] += [full_df['unit_price'].median() if 'unit_price' in full_df else None,
                                        full_df['quantity'].median() if 'quantity' in full_df else None]
                detail_data['差值'] += [
                    row['unit_price'] - (full_df['unit_price'].median() if 'unit_price' in full_df else 0),
                    row['quantity'] - (full_df['quantity'].median() if 'quantity' in full_df else 0)]
            detail_df = pd.DataFrame(detail_data)
            st.dataframe(detail_df, hide_index=True, use_container_width=True)
        if enable_deep_single:
            st.markdown("---")
            st.subheader("🖼️ 单品 CLIP 深度分析")
            if selected_asin in clip_results:
                st.success(f"✅ 该 ASIN 已有批量 CLIP 分析结果")
                clip_data = clip_results[selected_asin]
                img_metric_cols = st.columns(4)
                metrics = [("主图得分", clip_data.get('thumbnail_score', 0)),
                           ("详情图均分", clip_data.get('detail_score', 0)),
                           ("A+ 图得分", clip_data.get('aplus_score', 0)),
                           ("图片数量", clip_data.get('image_count', 0))]
                for i, (name, val) in enumerate(metrics):
                    with img_metric_cols[i]:
                        if name == "图片数量":
                            st.metric(name, f"{int(val)}")
                        else:
                            st.metric(name, f"{val:.1f}")
                dim_zh_map = {'attention': '注意力', 'product_understanding': '产品理解',
                              'value_perception': '价值感知', 'usage_imagination': '使用想象',
                              'trust_signal': '信任信号', 'risk_reduction': '风险消除',
                              'quality_perception': '品质感知', 'differentiation': '差异化',
                              'purchase_intent': '购买意愿'}
                dim_avgs = clip_data.get('dim_avgs', {})
                if dim_avgs:
                    fig_img = go.Figure()
                    fig_img.add_trace(go.Scatterpolar(
                        r=[dim_avgs.get(d, 0) for d in FEATURE_NAMES],
                        theta=[dim_zh_map[d] for d in FEATURE_NAMES], fill='toself', name=selected_asin))
                    fig_img.add_trace(go.Scatterpolar(
                        r=[50] * len(FEATURE_NAMES),
                        theta=[dim_zh_map[d] for d in FEATURE_NAMES], fill='toself', name='基线(50)',
                        line=dict(dash='dot', color='gray')))
                    fig_img.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                          showlegend=True, height=450)
                    st.plotly_chart(fig_img, use_container_width=True)
            else:
                if st.button(f"对 {selected_asin} 运行 CLIP 图片分析", type="primary"):
                    st.info("该功能需要从原始数据中提取图片 URL 并分析，请使用上方'批量分析时启用 CLIP'选项")

    st.markdown("---")
    st.subheader("📝 新品竞争力预测与智能优化建议")
    st.caption("基于数据集分位数 + 真实竞品对比，建议会根据你的输入动态变化")
    with st.form("new_product_form"):
        st.markdown("**填写你的产品信息**")
        col1, col2 = st.columns(2)
        with col1:
            new_title = st.text_input("产品标题", placeholder="例如: Premium Esszimmerstühle 4er Set")
            new_price = st.number_input("总价 (€)", min_value=0.0, step=0.01, value=29.99)
            new_quantity = st.number_input("套装数量（例如 4 把椅子）", min_value=1, step=1, value=1)
            new_stars = st.number_input("星级评分 (1-5)", min_value=0.0, max_value=5.0, step=0.1, value=4.3)
            new_reviews = st.number_input("评论数量", min_value=0, step=1, value=120)
        with col2:
            new_features = st.text_area("五点描述（每行一条）", placeholder="每条描述占一行", height=150)
            has_aplus = st.checkbox("是否有 A+ 内容")
            has_brandstory = st.checkbox("是否有品牌故事")
            video_count = st.selectbox("视频数量", [0, 1, 2, 3, 5], index=0)
            top_n_competitors = st.slider("🎯 找多少个相似竞品", min_value=3, max_value=10, value=5, step=1)
        st.markdown("**🖼️ 上传产品图片（可选）**")
        st.caption("上传主图后启用 CLIP 真实图片诊断 + 视觉相似竞品匹配（基于 DINOv2 外观）")
        uploaded_images = st.file_uploader("选择图片（jpg/png/webp）", type=['jpg', 'jpeg', 'png', 'webp'],
                                           accept_multiple_files=True, key="new_product_images")
        if uploaded_images and len(uploaded_images) > 6:
            st.warning("⚠️ 最多 6 张，已截取前 6 张")
            uploaded_images = uploaded_images[:6]
        image_types_list = []
        if uploaded_images:
            st.markdown("**为每张图片指定角色**")
            type_cols = st.columns(min(len(uploaded_images), 3))
            for i, img_file in enumerate(uploaded_images):
                with type_cols[i % 3]:
                    default_type = 'main' if i == 0 else ('detail' if i <= 3 else 'lifestyle')
                    img_type = st.selectbox(f"图 {i + 1}: {img_file.name[:20]}",
                                            options=['main', 'detail', 'lifestyle', 'aplus'],
                                            format_func=lambda x:
                                            {'main': '主图（白底）', 'detail': '细节图', 'lifestyle': '场景图',
                                             'aplus': 'A+图'}[x],
                                            index=['main', 'detail', 'lifestyle', 'aplus'].index(default_type),
                                            key=f"img_type_{i}")
                    image_types_list.append(img_type)
        submitted = st.form_submit_button("📊 分析新品竞争力", type="primary")

    if submitted:
        if final_df.empty:
            st.warning("当前没有可对比的数据集")
        else:
            image_analysis_result = None
            dataset_image_embeddings = None
            thumbnail_urls_map = None
            # [CHANGE] 新增外观向量相关变量
            new_appearance_emb = None
            dataset_appearance_embeddings = None

            if uploaded_images:
                with st.spinner("🖼️ 加载 CLIP 模型并分析上传图片（首次约 30 秒）..."):
                    try:
                        img_bytes_list = [f.getvalue() for f in uploaded_images]
                        image_analysis_result = analyze_uploaded_images_with_clip(tuple(img_bytes_list),
                                                                                  tuple(image_types_list))
                        if image_analysis_result:
                            st.success(f"✅ 图片分析完成，综合得分 {image_analysis_result['overall_score']:.1f}")
                    except Exception as e:
                        st.warning(f"图片分析失败: {e}")
                        image_analysis_result = None

                # [CHANGE] 提取新品主图的外观向量（DINOv2 + rembg）
                if image_analysis_result:
                    main_idx = image_analysis_result.get('main_image_index')
                    if main_idx is not None and main_idx < len(uploaded_images):
                        main_img_bytes = uploaded_images[main_idx].getvalue()
                        with st.spinner("🖼️ 提取产品外观特征（去背景+DINOv2）..."):
                            new_appearance_emb = extract_product_appearance_embedding(main_img_bytes)
                            if new_appearance_emb is not None:
                                st.success("✅ 外观特征提取成功")
                            else:
                                st.warning("⚠️ 外观特征提取失败，将仅使用标题+价格匹配")

                # 构建数据集主图 URL 列表（用于外观向量）
                if new_appearance_emb is not None:
                    thumb_urls_list = []
                    thumbnail_urls_map = {}
                    for item in classified_data["搜索页信息"]:
                        asin = item.get("asin")
                        url = item.get("thumbnailImage")
                        if asin and url:
                            thumb_urls_list.append((asin, url))
                            thumbnail_urls_map[asin] = url
                    if thumb_urls_list:
                        # [CHANGE] 调用新的外观向量计算函数（DINOv2 + rembg）
                        with st.spinner(f"🖼️ 计算数据集 {len(thumb_urls_list)} 个产品的外观向量（DINOv2）..."):
                            try:
                                # 这里可以限制数量，但函数内已有限制（200）
                                dataset_appearance_embeddings = compute_dataset_appearance_embeddings(tuple(thumb_urls_list))
                                st.success(f"✅ 数据集外观向量完成，{len(dataset_appearance_embeddings)} 个成功")
                            except Exception as e:
                                st.warning(f"数据集外观向量失败: {e}")
                                dataset_appearance_embeddings = None
                else:
                    # 如果新品外观向量提取失败，数据集外观也不必计算
                    dataset_appearance_embeddings = None

                # 如果不需要外观，仍可计算普通 CLIP 向量用于其他显示（可选）
                # 但外观匹配已用 DINOv2，无需重复计算 CLIP 数据集向量
                # 如果你还想要 CLIP 的 thumbnail 用于其他目的，可以保留，但这里不强制

            with st.spinner("智能分析中..."):
                result = analyze_new_product_smart(
                    new_title, new_price, new_stars, new_reviews,
                    new_features, has_aplus, has_brandstory, video_count,
                    full_df,
                    new_quantity=new_quantity,
                    image_analysis_result=image_analysis_result,
                    dataset_image_embeddings=dataset_image_embeddings,  # 保留但不再用于匹配
                    thumbnail_urls_map=thumbnail_urls_map,
                    top_n_competitors=top_n_competitors,
                    new_appearance_emb=new_appearance_emb,               # [CHANGE] 传入外观向量
                    dataset_appearance_embeddings=dataset_appearance_embeddings  # [CHANGE] 传入数据集外观向量
                )

            st.subheader("📊 新品 vs 数据集 分位数对比")
            st.dataframe(result['compare_df'], hide_index=True, use_container_width=True)
            st.caption("🟢 前25% / 🟡 中上 / 🟠 中下 / 🔴 后25%")

            st.subheader("🎯 新品核心得分")
            score_cols = st.columns(4)
            scores = result['scores']
            for i, (name, val) in enumerate([('综合总分', scores['total']), ('搜索得分', scores['search']),
                                             ('详情得分', scores['listing']), ('转化得分', scores['conversion'])]):
                with score_cols[i]:
                    st.metric(name, f"{val:.1f}")

            st.subheader("💡 智能优化建议")
            if not result['advice']:
                st.success("🎉 各项指标均优于数据集中位水平！")
            else:
                for icon, dim, text in result['advice']:
                    color = {"🟢": "green", "🟡": "#888", "🟠": "#FF8C00", "🔴": "red",
                             "❌": "red", "🎯": "#0066CC", "📌": "#663399", "📋": "#444",
                             "ℹ️": "#666", "✏️": "#CC6633"}.get(icon, "#333")
                    st.markdown(f"<div style='padding:8px 12px;margin:4px 0;border-left:3px solid {color};"
                                f"background:#f9f9f9;'><strong>{icon} {dim}</strong>：{text}</div>",
                                unsafe_allow_html=True)

            if result.get('img_source') == 'uploaded' and result.get('image_analysis'):
                st.subheader("🖼️ 图片深度分析明细")
                img_res = result['image_analysis']
                img_metric_cols = st.columns(4)
                metrics = [("综合得分", img_res['overall_score']), ("图片数量", img_res['image_count']),
                           ("主图得分", img_res['type_scores'].get('main', {}).get('score', 0)),
                           ("细节图均分", img_res['type_scores'].get('detail', {}).get('score', 0))]
                for i, (name, val) in enumerate(metrics):
                    with img_metric_cols[i]:
                        if name == "图片数量":
                            st.metric(name, f"{int(val)}")
                        else:
                            st.metric(name, f"{val:.1f}")
                dim_zh_map = {'attention': '注意力', 'product_understanding': '产品理解',
                              'value_perception': '价值感知', 'usage_imagination': '使用想象',
                              'trust_signal': '信任信号', 'risk_reduction': '风险消除',
                              'quality_perception': '品质感知', 'differentiation': '差异化',
                              'purchase_intent': '购买意愿'}
                dim_names = [dim_zh_map[d] for d in FEATURE_NAMES]
                dim_vals = [img_res['dim_avgs'].get(d, 0) for d in FEATURE_NAMES]
                fig_img = go.Figure()
                fig_img.add_trace(go.Scatterpolar(r=dim_vals, theta=dim_names, fill='toself', name='新品图片'))
                fig_img.add_trace(go.Scatterpolar(r=[50] * len(FEATURE_NAMES), theta=dim_names,
                                                  fill='toself', name='基线(50)', line=dict(dash='dot', color='gray')))
                fig_img.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                      showlegend=True, height=450)
                st.plotly_chart(fig_img, use_container_width=True)

            st.subheader("📊 核心维度雷达图")
            core_dims = ['标题', '价格', '信任', '五点描述', '图片', '视频', 'A+']
            core_new = [scores['title'], scores['price'], scores['trust'], scores['bullet'],
                        scores['image'], scores['video'], scores['aplus']]
            core_median = [result['compare_df'].iloc[i]['数据集 P50（中位）'] for i in range(7)]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=core_new, theta=core_dims, fill='toself', name='新品'))
            fig.add_trace(go.Scatterpolar(r=core_median, theta=core_dims, fill='toself', name='数据集中位'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              showlegend=True, height=450)
            st.plotly_chart(fig, use_container_width=True)

            if not result['competitors'].empty:
                n_comp = len(result['competitors'])
                st.subheader(f"🎯 最相似的 {n_comp} 个竞品")
                img_res = result.get('image_analysis') or {}
                main_emb = img_res.get('main_image_embedding')
                main_idx = img_res.get('main_image_index')
                if uploaded_images:
                    if main_emb is not None:
                        st.success(f"✅ 已基于新品主图（第 {main_idx + 1} 张）的 CLIP 视觉向量匹配竞品")
                    else:
                        st.warning("⚠️ 主图向量提取失败，仅基于标题+价格匹配")
                else:
                    st.info("ℹ️ 未上传新品图片，仅基于标题+价格匹配")

                # [CHANGE] 只有当 has_image_sim 为 True 时才显示视觉对比，且外观向量存在
                if result.get('has_image_sim') and thumbnail_urls_map:
                    st.markdown(f"**📸 视觉对比（新品 vs Top {min(n_comp, 3)} 竞品）**")
                    n_show = min(n_comp, 3)
                    img_compare_cols = st.columns(n_show + 1)
                    with img_compare_cols[0]:
                        st.markdown("**🆕 新品主图**")
                        if main_idx is not None and uploaded_images and main_idx < len(uploaded_images):
                            try:
                                st.image(Image.open(uploaded_images[main_idx]), use_container_width=True)
                            except:
                                st.warning("主图预览失败")
                        else:
                            found = False
                            for i, img_file in enumerate(uploaded_images or []):
                                if image_types_list[i] == 'main':
                                    try:
                                        st.image(Image.open(img_file), use_container_width=True); found = True
                                    except:
                                        st.warning("主图预览失败")
                                    break
                            if not found: st.info("未找到主图")
                    for idx, (_, comp) in enumerate(result['competitors'].head(n_show).iterrows()):
                        with img_compare_cols[idx + 1]:
                            asin = comp['asin']
                            sim_score = comp.get('sim', 0)
                            img_sim = comp.get('appearance_sim')  # [CHANGE] 使用 appearance_sim 而非 image_sim
                            sim_label = f"相似度 {sim_score * 100:.0f}%"
                            if img_sim is not None and not pd.isna(img_sim):
                                sim_label += f"\n\n外观 {img_sim * 100:.0f}%"
                            st.markdown(f"**#{idx + 1} {asin}**\n\n{sim_label}")
                            url = thumbnail_urls_map.get(asin)
                            if url:
                                try:
                                    st.image(url, use_container_width=True)
                                except:
                                    st.warning("竞品图加载失败")
                            else:
                                st.info("无主图")
                    st.markdown("")

                st.markdown(f"**📊 {n_comp} 个竞品详细对比**")
                # [CHANGE] 由于新增了 appearance_sim，调整显示
                if result.get('has_image_sim') and 'appearance_sim' in result['competitors'].columns:
                    comp_display = result['competitors'][['asin', 'title', 'brand', 'price_value', 'unit_price',
                                                          'Total_Score', 'sim', 'appearance_sim', 'title_sim',
                                                          'price_sim']].copy()
                    comp_display['排名'] = range(1, len(comp_display) + 1)
                    comp_display['综合相似度'] = comp_display['sim'].apply(lambda x: f"{x * 100:.1f}%")
                    comp_display['外观相似'] = comp_display['appearance_sim'].apply(
                        lambda x: f"{x * 100:.1f}%" if x is not None and not pd.isna(x) else "—")
                    comp_display['标题相似'] = comp_display['title_sim'].apply(lambda x: f"{x * 100:.1f}%")
                    comp_display['价格相似'] = comp_display['price_sim'].apply(lambda x: f"{x * 100:.1f}%")
                    comp_display = comp_display.drop(columns=['sim', 'appearance_sim', 'title_sim', 'price_sim'])
                    comp_display = comp_display.rename(columns={
                        'title': '标题',
                        'brand': '品牌',
                        'price_value': '总价(€)',
                        'unit_price': '单价(€)',
                        'Total_Score': '综合分'
                    })
                    comp_display = comp_display[['排名', 'asin', '标题', '品牌', '总价(€)', '单价(€)', '综合分',
                                                 '综合相似度', '外观相似', '标题相似', '价格相似']]
                else:
                    # 回退到无外观的显示
                    comp_display = result['competitors'][['asin', 'title', 'brand', 'price_value', 'unit_price',
                                                          'Total_Score', 'sim', 'title_sim', 'price_sim']].copy()
                    comp_display['排名'] = range(1, len(comp_display) + 1)
                    comp_display['综合相似度'] = comp_display['sim'].apply(lambda x: f"{x * 100:.1f}%")
                    comp_display['标题相似'] = comp_display['title_sim'].apply(lambda x: f"{x * 100:.1f}%")
                    comp_display['价格相似'] = comp_display['price_sim'].apply(lambda x: f"{x * 100:.1f}%")
                    comp_display = comp_display.drop(columns=['sim', 'title_sim', 'price_sim'])
                    comp_display = comp_display.rename(columns={
                        'title': '标题',
                        'brand': '品牌',
                        'price_value': '总价(€)',
                        'unit_price': '单价(€)',
                        'Total_Score': '综合分'
                    })
                    comp_display = comp_display[['排名', 'asin', '标题', '品牌', '总价(€)', '单价(€)', '综合分',
                                                 '综合相似度', '标题相似', '价格相似']]
                comp_display['详情页'] = comp_display['asin'].apply(lambda x: f'https://www.amazon.de/dp/{x}')
                st.dataframe(comp_display, hide_index=True, use_container_width=True,
                             column_config={"详情页": st.column_config.LinkColumn("Amazon",
                                                                                  display_text=r'^(https://www\.amazon\.de/dp/)(.*)')})

                if result.get('all_ranked') is not None and not result['all_ranked'].empty:
                    total_n = len(result['all_ranked'])
                    with st.expander(f"📊 查看数据集全部 {total_n} 个产品的相似度排名"):
                        all_disp = result['all_ranked'][['asin', 'title', 'price_value', 'unit_price',
                                                         'Total_Score', 'sim', 'appearance_sim' if 'appearance_sim' in result['all_ranked'].columns else 'image_sim']].copy()
                        # 统一列名
                        sim_col = 'appearance_sim' if 'appearance_sim' in all_disp.columns else 'image_sim'
                        all_disp['排名'] = range(1, len(all_disp) + 1)
                        all_disp['综合相似度'] = all_disp['sim'].apply(lambda x: f"{x * 100:.1f}%")
                        all_disp['外观相似度'] = all_disp[sim_col].apply(
                            lambda x: f"{x * 100:.1f}%" if x is not None and not pd.isna(x) else "—")
                        all_disp = all_disp[['排名', 'asin', 'title', 'price_value', 'unit_price',
                                             'Total_Score', '综合相似度', '外观相似度']]
                        all_disp = all_disp.rename(columns={
                            'title': '标题',
                            'price_value': '总价(€)',
                            'unit_price': '单价(€)',
                            'Total_Score': '综合分'
                        })
                        st.dataframe(all_disp, hide_index=True, use_container_width=True)

else:
    st.info("👈 请上传原始爬虫 JSON 文件开始分析")
    st.markdown("""
    ### 📖 使用说明
    **快速分析（默认）**
    - 100 产品约 30 秒，图片得分固定 50
    **批量 CLIP 分析（侧边栏开启）**
    - 每产品分析 1 主图+2 详情+2 A+图
    - 100 产品约 5-10 分钟
    - 图片得分基于真实 CLIP 分析
    **新品对比（启用外观匹配）**
    - 上传主图，系统会提取产品外观特征（去背景+DINOv2）
    - 竞品匹配权重：标题 40% + 外观 40% + 价格 20%
    - 如果 rembg 未安装，会自动回退到标题+价格匹配
    **增强功能**
    - 双语检测（英德同时识别）
    - 高频词参考（基于数据集）
    - 五点描述改写示例
    - **新增：自动从标题/attributes 提取套装数量，计算并显示单价**
    - **价格评分、竞品匹配、价格建议均基于「单价」而非「总价」**
    """)
