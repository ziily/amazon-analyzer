import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math
import time
from io import BytesIO
from PIL import Image
import requests
import torch
import re
from transformers import CLIPProcessor, CLIPModel, pipeline
import psutil
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ==================== 页面配置 & 隐藏水印 ====================
st.set_page_config(page_title="Amazon 产品竞争力分析", layout="wide")

# 隐藏 Streamlit 默认菜单和底部 "Built with Streamlit"
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== 分类映射（来自 classsify.py） ====================
CLASSIFICATION = {
    "搜索页信息": [
        "title", "url", "asin", "thumbnailImage", "price", "listPrice",
        "stars", "reviewsCount", "delivery", "fastestDelivery",
        "isAmazonChoice", "amazonChoiceText", "categoryPageData",
        "unNormalizedProductUrl", "input"
    ],
    "listing信息": [
        "originalAsin", "inStock", "inStockText", "brand", "author",
        "breadCrumbs", "videosCount", "visitStoreLink", "galleryThumbnails",
        "highResolutionImages", "importantInformation", "sustainabilityFeatures",
        "description", "features", "attributes", "productOverview",
        "variantAsins", "variantDetails", "variantAttributes",
        "manufacturerAttributes", "condition", "returnPolicy", "support",
        "aPlusContent", "brandStory", "bookDescription", "locationText",
        "loadedCountryCode"
    ],
    "评论信息": [
        "starsBreakdown", "answeredQuestions", "reviewsLink", "hasReviews",
        "aiReviewsSummary", "productPageReviews", "productPageReviewsFromOtherCountries"
    ],
    "市场信息": [
        "shippingPrice", "seller", "bestsellerRanks", "priceRange",
        "offers", "monthlyPurchaseVolume", "productComparison"
    ]
}


def classify_raw_data(products):
    """输入原始产品列表，返回分类后的字典 {'搜索页信息': [...], ...}"""
    field_to_module = {}
    for module, fields in CLASSIFICATION.items():
        for field in fields:
            field_to_module[field] = module

    classified = {module: [] for module in CLASSIFICATION.keys()}
    for product in products:
        search_dict = {}
        listing_dict = {}
        review_dict = {}
        market_dict = {}
        for key, value in product.items():
            module = field_to_module.get(key)
            if module == "搜索页信息":
                search_dict[key] = value
            elif module == "listing信息":
                listing_dict[key] = value
            elif module == "评论信息":
                review_dict[key] = value
            elif module == "市场信息":
                market_dict[key] = value
        classified["搜索页信息"].append(search_dict)
        classified["listing信息"].append(listing_dict)
        classified["评论信息"].append(review_dict)
        classified["市场信息"].append(market_dict)
    return classified


# ==================== 全局模型缓存 ====================
@st.cache_resource
def load_clip():
    print("🖼️ 加载 CLIP 模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("✅ CLIP 加载完成")
    return model, processor


@st.cache_resource
def load_zero_shot():
    try:
        print("🔧 加载零样本分类模型 (轻量级 distilbert) ...")
        return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


@st.cache_resource
def load_sentiment_pipeline():
    try:
        print("📊 加载情感分析模型 (german-sentiment-bert)...")
        return pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert", device=-1, top_k=None)
    except:
        try:
            print("📊 加载情感分析模型 (twitter-xlm-roberta)...")
            return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=-1,
                            top_k=None)
        except:
            print("❌ 情感模型加载失败")
            return None


@st.cache_resource
def load_gliner():
    try:
        print("🧠 加载 GLiNER2 多语言模型...")
        from gliner import GLiNER
        model = GLiNER.from_pretrained("urchade/gliner2_multi-v1")
        print("✅ GLiNER2 加载完成")
        return model
    except Exception as e:
        print(f"❌ GLiNER2 加载失败: {e}")
        return None


# ==================== 图片分析（CLIP） ====================
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


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB")
    except Exception as e:
        print(f"   ⚠️ 图片下载失败: {url[:80]}... ({e})")
        return None


def analyze_image_with_clip(url, model, processor):
    img = load_image_from_url(url)
    if img is None:
        return None
    inputs = processor(text=TEXT_PROMPTS, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image[0]
    positive = logits[:-1]
    baseline = logits[-1]
    scores = torch.sigmoid(positive - baseline)
    return {name: float(scores[i]) for i, name in enumerate(FEATURE_NAMES)}


def calculate_consumer_score(row):
    itype = row["image_type"]
    if itype == "thumbnail":
        score = (row["attention"] * 0.35 + row["product_understanding"] * 0.30 +
                 row["quality_perception"] * 0.20 + row["differentiation"] * 0.15)
    elif itype == "high_resolution":
        score = (row["value_perception"] * 0.30 + row["usage_imagination"] * 0.25 +
                 row["risk_reduction"] * 0.25 + row["trust_signal"] * 0.20)
    elif itype == "a_plus":
        score = (row["trust_signal"] * 0.35 + row["quality_perception"] * 0.25 +
                 row["differentiation"] * 0.20 + row["value_perception"] * 0.20)
    else:
        score = 0
    return round(score * 100, 2)


def aggregate_images(image_records):
    import hashlib
    print(f"🖼️ 开始图片分析，共 {len(image_records)} 张")

    # ===== 缓存检查 =====
    url_str = "".join(sorted([r.get("image_url", "") for r in image_records]))
    cache_hash = hashlib.md5(url_str.encode()).hexdigest()[:12]
    cache_dir = "data/cache"  # 已修正为相对路径，云上可用
    cache_file = os.path.join(cache_dir, f"image_cache_{cache_hash}.csv")

    if os.path.exists(cache_file):
        print(f"✅ 命中图片缓存，直接加载（跳过CLIP推理）")
        return pd.read_csv(cache_file)
    # ====================

    model, processor = load_clip()
    results = []
    total = len(image_records)
    success = 0
    fail = 0

    for i, rec in enumerate(image_records):
        if rec.get("image_url"):
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  图片进度 {i + 1}/{total}, 成功 {success}, 失败 {fail}")
            features = analyze_image_with_clip(rec["image_url"], model, processor)
            if features:
                success += 1
                item = {"asin": rec["asin"], "image_type": rec["image_type"], "image_url": rec["image_url"]}
                item.update(features)
                results.append(item)
            else:
                fail += 1
                if fail % 5 == 0:
                    print(f"   ⚠️ 已有 {fail} 张图片下载失败，继续...")

    print(f"✅ 图片分析完成，成功 {success}，失败 {fail}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["consumer_score"] = df.apply(calculate_consumer_score, axis=1)

    # 聚合
    summary = []
    for asin, group in df.groupby("asin"):
        row = {"asin": asin}
        thumb = group[group["image_type"] == "thumbnail"]
        if len(thumb) > 0:
            row["thumbnail_score"] = thumb["consumer_score"].mean()
            row["thumbnail_attention"] = thumb["attention"].mean()
            row["thumbnail_quality"] = thumb["quality_perception"].mean()
            row["thumbnail_purchase"] = thumb["purchase_intent"].mean()
        else:
            row["thumbnail_score"] = row["thumbnail_attention"] = row["thumbnail_quality"] = row[
                "thumbnail_purchase"] = 0
        detail = group[group["image_type"] == "high_resolution"]
        if len(detail) > 0:
            row["detail_image_count"] = len(detail)
            row["detail_visual_score"] = detail["consumer_score"].mean()
            row["detail_best_score"] = detail["consumer_score"].max()
            row["detail_trust"] = detail["trust_signal"].mean()
            row["detail_value"] = detail["value_perception"].mean()
            row["detail_usage"] = detail["usage_imagination"].mean()
            row["detail_risk"] = detail["risk_reduction"].mean()
        else:
            row["detail_image_count"] = 0
            row["detail_visual_score"] = row["detail_best_score"] = row["detail_trust"] = row["detail_value"] = row[
                "detail_usage"] = row["detail_risk"] = 0
        aplus = group[group["image_type"] == "a_plus"]
        if len(aplus) > 0:
            row["aplus_count"] = len(aplus)
            row["aplus_mean_score"] = aplus["consumer_score"].mean()
            row["aplus_trust"] = aplus["trust_signal"].mean()
            row["aplus_quality"] = aplus["quality_perception"].mean()
            row["aplus_value"] = aplus["value_perception"].mean()
            row["aplus_brand"] = aplus["differentiation"].mean()
        else:
            row["aplus_count"] = 0
            row["aplus_mean_score"] = row["aplus_trust"] = row["aplus_quality"] = row["aplus_value"] = row[
                "aplus_brand"] = 0
        summary.append(row)

    result_df = pd.DataFrame(summary)

    # 保存缓存
    os.makedirs(cache_dir, exist_ok=True)
    result_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    print(f"💾 图片缓存已保存: {cache_file}")

    return result_df


# ==================== 心理评分 ====================
PSYCH_LABELS = ["quality", "convenience", "cost_saving", "safety", "social_status", "health", "durability",
                "aesthetics", "innovation", "trust"]


def batch_psych_scores(texts, classifier, batch_size=16):
    cache = {}
    unique = list(set([t for t in texts if t and len(t.strip()) >= 3]))
    if not unique:
        return {}
    if classifier is None:
        for t in unique:
            cache[t] = 0.0
        return cache

    total_batches = (len(unique) + batch_size - 1) // batch_size
    print(f"🧠 开始心理评分，共 {len(unique)} 条唯一文本，分 {total_batches} 个批次")

    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  批次 {batch_num}/{total_batches}，处理 {len(batch)} 条文本")
        try:
            results = classifier(batch, PSYCH_LABELS)
            if isinstance(results, dict):
                results = [results]
            for text, res in zip(batch, results):
                scores = res['scores']
                max_score = max(scores)
                diversity = min(len([s for s in scores if s > 0.3]) / 3, 1.0)
                final = max_score * 0.7 + diversity * 0.3
                cache[text] = round(final * 100, 2)
        except Exception as e:
            print(f"   ⚠️ 批次 {batch_num} 推理失败: {e}")
            for t in batch:
                cache[t] = 0.0
    print(f"✅ 心理评分完成")
    return cache
# ==================== 新增：信息点检测函数 ====================
def check_info_points(text: str, language: str = "de") -> dict:
    """检测文本是否包含材质、颜色、尺寸、功能、场景等信息点"""
    patterns = {
        "material": {
            "en": r"leather|fabric|wood|metal|plastic|steel|oak|velvet|linen|pu|abs",
            "de": r"leder|stoff|holz|metall|kunststoff|stahl|eiche|samt|leinen|pu|abs",
        },
        "color": {
            "en": r"black|white|grey|gray|brown|blue|red|green|beige|navy",
            "de": r"schwarz|weiß|grau|braun|blau|rot|grün|beige|marine",
        },
        "size": {
            "en": r"\d+[\'\"]|\d+\s?(inch|cm|mm|ft)|\d+\s?(lb|kg|g)|\d+\s?set of \d+|\d+\s?pack",
            "de": r"\d+\s?(cm|mm|m|kg|g|l|ml)|\d+\s?stück|\d+\s?set",
        },
        "function": {
            "en": r"foldable|stackable|swivel|adjustable|ergonomic|reclining|portable|waterproof|breathable",
            "de": r"faltbar|stapelbar|drehbar|verstellbar|ergonomisch|neigbar|tragbar|wasserdicht|atmungsaktiv",
        },
        "scenario": {
            "en": r"kitchen|dining|office|living room|bedroom|outdoor|bar|home|work|study",
            "de": r"küche|esszimmer|büro|wohnzimmer|schlafzimmer|draußen|bar|zuhause|arbeit|studium",
        },
    }
    lang = "de" if language.lower() in ["de", "ger"] else "en"
    text_lower = text.lower()
    hit = {}
    for dim, lang_dict in patterns.items():
        pattern = lang_dict.get(lang, lang_dict["en"])
        hit[dim] = 1 if re.search(pattern, text_lower, re.IGNORECASE) else 0
    return hit

# ==================== 新增：标题评分函数 ====================
def compute_title_score(title: str, classifier=None) -> tuple:
    """返回 (总分, 详情字典) 详情包含 length, psych, info, has_num, has_unit, hit"""
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

    psych_score = 0.0
    if classifier is not None and len(title) >= 3:
        try:
            res = classifier(title, PSYCH_LABELS)
            scores = res['scores']
            max_score = max(scores)
            diversity = min(len([s for s in scores if s > 0.3]) / 3, 1.0)
            psych_score = (max_score * 0.7 + diversity * 0.3) * 100
        except:
            pass

    info_hit = check_info_points(title, language='de')
    hit_count = sum(info_hit.values())
    info_score = (hit_count / len(info_hit)) * 100

    total = len_score * 0.25 + psych_score * 0.50 + info_score * 0.25
    has_num = any(c.isdigit() for c in title)
    has_unit = any(u in title.lower() for u in ['cm', 'mm', 'kg', 'g', 'ml', 'l', 'w', 'h'])
    total += (5 if has_num else 0) + (5 if has_unit else 0)
    total = round(min(total, 100), 2)

    details = {
        "length": len_score,
        "psych": round(psych_score, 2),
        "info": round(info_score, 2),
        "has_num": has_num,
        "has_unit": has_unit,
        "hit": info_hit
    }
    return total, details

# ==================== Listing 各维度评分 ====================
def score_features_batch(features, text2score):
    if not features:
        return 0.0
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


# ==================== 信息覆盖度检查（基于 GLiNER2） ====================
def fallback_check_coverage(text, language='en'):
    """原关键词匹配方案，作为 GLiNER2 的降级备份"""
    keywords = {
        'en': {
            'size': ['size', 'dimension', 'length', 'width', 'height', 'cm', 'mm', 'kg', 'g', 'ml', 'l'],
            'material': ['material', 'fabric', 'leather', 'plastic', 'metal', 'wood', 'cotton', 'polyester'],
            'warranty': ['warranty', 'guarantee', '2-year', '1-year', 'lifetime'],
            'usage': ['use', 'applicable', 'suitable for', 'perfect for', 'ideal for', 'scenario'],
            'differentiation': ['unique', 'exclusive', 'only', 'best', 'unlike', 'compare', 'than others', 'superior'],
            'user_oriented': ['you', 'your', 'user', 'customer', 'comfort', 'enjoy', 'experience']
        },
        'de': {
            'size': ['größe', 'abmessungen', 'länge', 'breite', 'höhe', 'cm', 'mm', 'kg', 'g', 'ml', 'l'],
            'material': ['material', 'stoff', 'leder', 'kunststoff', 'holz', 'baumwolle', 'polyester'],
            'warranty': ['garantie', 'gewährleistung', '2-jährig', '1-jährig', 'lebenslang'],
            'usage': ['verwendung', 'geeignet für', 'perfekt für', 'ideal für', 'szenario', 'einsatz'],
            'differentiation': ['einzigartig', 'exklusiv', 'nur', 'beste', 'anders als', 'vergleichen', 'überlegen'],
            'user_oriented': ['sie', 'ihnen', 'ihr', 'benutzer', 'kunde', 'komfort', 'genießen', 'erleben']
        }
    }
    lang = 'de' if language in ['de', 'ger'] else 'en'
    kw = keywords.get(lang, keywords['en'])
    text_lower = text.lower()
    coverage = {}
    for dim, words in kw.items():
        coverage[dim] = 1 if any(w in text_lower for w in words) else 0
    weights = {'size': 0.2, 'material': 0.2, 'warranty': 0.1, 'usage': 0.15, 'differentiation': 0.2,
               'user_oriented': 0.15}
    total = sum(coverage[dim] * weights[dim] for dim in weights) * 100
    return {'coverage': coverage, 'total_score': round(total, 2)}


def check_coverage(text, language='de'):
    """
    使用 GLiNER2 检查文本是否覆盖关键信息维度。
    返回各维度的覆盖情况（0/1）和总分（0-100）。
    """
    if not text or len(text.strip()) < 5:
        return {'coverage': {}, 'total_score': 0}

    model = load_gliner()
    if model is None:
        return fallback_check_coverage(text, language)

    entity_types = {
        'size': "product dimensions, size, weight, volume, length, width, height",
        'material': "product material, fabric, leather, plastic, metal, wood, cotton, polyester",
        'warranty': "warranty, guarantee, after-sales service, return policy",
        'usage': "product use, application, suitable scenarios, target users",
        'differentiation': "unique features, competitive advantages, exclusivity, unlike others",
        'user_oriented': "user-centered benefits, customer comfort, user experience, satisfaction"
    }

    try:
        entities = model.extract_entities(text, labels=list(entity_types.values()))
    except Exception as e:
        print(f"⚠️ GLiNER2 提取实体失败: {e}")
        return fallback_check_coverage(text, language)

    coverage = {}
    for dim, desc in entity_types.items():
        matched = any(e['label'] == desc for e in entities)
        coverage[dim] = 1 if matched else 0

    weights = {'size': 0.2, 'material': 0.2, 'warranty': 0.1, 'usage': 0.15,
               'differentiation': 0.2, 'user_oriented': 0.15}
    total = sum(coverage[dim] * weights[dim] for dim in weights) * 100
    return {'coverage': coverage, 'total_score': round(total, 2)}


def score_attributes(attributes):
    if not attributes:
        return 0.0
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
    if not aplus:
        return 0.0
    modules = aplus.get("modules", [])
    mod_score = min(len(modules) / 5, 1.0) * 60
    video_bonus = min(len(aplus.get("rawVideos", [])), 2) * 15
    img_count = sum(1 for img in aplus.get("rawImages", []) if img.get("url"))
    img_bonus = min(img_count, 3) * 3
    return round(min(mod_score + video_bonus + img_bonus, 100), 2)


def score_video(count):
    if count is None or count == 0:
        return 0.0
    return 100.0 if count >= 3 else (90.0 if count == 2 else 70.0)


def score_images(asin, image_dict):
    if asin not in image_dict:
        return 20.0
    info = image_dict[asin]
    thumb = info.get("thumbnail_score", 0)
    thumb_purchase = info.get("thumbnail_purchase", 0)
    main_score = thumb * 0.6 + thumb_purchase * 0.4
    detail_visual = info.get("detail_visual_score", 0)
    detail_risk = info.get("detail_risk", 0)
    detail_score = detail_visual * 0.6 + detail_risk * 0.4
    aplus_mean = info.get("aplus_mean_score", 0)
    aplus_trust = info.get("aplus_trust", 0)
    aplus_score = aplus_mean * 0.5 + aplus_trust * 0.5
    total = main_score * 0.40 + detail_score * 0.40 + aplus_score * 0.20
    return round(total, 2)


def score_brandstory(story):
    return 100.0 if story and story.get("items") else 0.0


# ==================== 评论分析 ====================
def analyze_reviews(review_data, sentiment_pipeline):
    print(f"📝 开始评论分析，共 {len(review_data)} 个商品")
    results = []
    for idx, rev in enumerate(review_data):
        reviews_link = rev.get("reviewsLink", "")
        asin = reviews_link.split("/")[-1].split("?")[0] if reviews_link else None
        if not asin:
            asin = f"Unknown_{idx + 1}"
        if idx % 10 == 0:
            print(f"  评论进度 {idx + 1}/{len(review_data)}")

        stars_br = rev.get("starsBreakdown") or {}
        avg_stars = (stars_br.get("5star", 0) * 5 + stars_br.get("4star", 0) * 4 +
                     stars_br.get("3star", 0) * 3 + stars_br.get("2star", 0) * 2 + stars_br.get("1star", 0) * 1)
        ai_summary = rev.get("aiReviewsSummary") or {}
        keywords = ai_summary.get("keywords", [])
        pos_mentions = sum(kw.get("customersMentionedCount", {}).get("total", 0) for kw in keywords if
                           kw.get("sentiment") == "positive")
        neg_mentions = sum(kw.get("customersMentionedCount", {}).get("total", 0) for kw in keywords if
                           kw.get("sentiment") == "negative")
        total_mentions = pos_mentions + neg_mentions
        ai_pos_ratio = pos_mentions / total_mentions if total_mentions > 0 else 0.5

        all_reviews = []
        for r in rev.get("productPageReviews", []):
            desc = r.get("reviewDescription", "")
            rating = r.get("ratingScore")
            if desc and rating:
                all_reviews.append((desc, rating))
        for r in rev.get("productPageReviewsFromOtherCountries", []):
            desc = r.get("reviewDescription", "")
            rating = r.get("ratingScore")
            if desc and rating:
                all_reviews.append((desc, rating))

        sentiment_scores = []
        if sentiment_pipeline:
            for text, _ in all_reviews:
                if len(text.strip()) < 3:
                    continue
                text = text[:512]
                try:
                    res = sentiment_pipeline(text)
                    if res and isinstance(res, list):
                        if isinstance(res[0], list):
                            labels = res[0]
                        else:
                            labels = res
                        probs = {item['label']: item['score'] for item in labels}
                        if 'positive' in probs:
                            score = probs.get('positive', 0) - probs.get('negative', 0)
                        else:
                            score_map = {'LABEL_0': -1, 'LABEL_1': 0, 'LABEL_2': 1}
                            score = sum(probs.get(label, 0) * score_map[label] for label in score_map)
                        sentiment_scores.append(score)
                except:
                    pass
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        if sentiment_scores:
            positive_ratio = sum(1 for s in sentiment_scores if s > 0.3) / len(sentiment_scores)
            neutral_ratio = sum(1 for s in sentiment_scores if -0.3 <= s <= 0.3) / len(sentiment_scores)
            negative_ratio = sum(1 for s in sentiment_scores if s < -0.3) / len(sentiment_scores)
        else:
            positive_ratio = neutral_ratio = negative_ratio = 0

        total_comments = len(all_reviews)
        stars_norm = avg_stars / 5 if avg_stars else 0.5
        sent_norm = (avg_sentiment + 1) / 2 if avg_sentiment is not None else 0.5
        pos_norm = positive_ratio if positive_ratio is not None else 0.5
        count_norm = min(math.log(total_comments + 1) / math.log(1001), 1.0) if total_comments else 0.0
        ai_norm = ai_pos_ratio if ai_pos_ratio is not None else 0.5
        w = [0.30, 0.30, 0.20, 0.10, 0.10]
        final_score = (stars_norm * w[0] + sent_norm * w[1] + pos_norm * w[2] + count_norm * w[3] + ai_norm * w[
            4]) * 100
        results.append({
            "ASIN": asin,
            "Avg_Stars": round(avg_stars, 2),
            "Total_Reviews_Count": total_comments,
            "Avg_Sentiment_Score": round(avg_sentiment, 3),
            "Positive%": round(positive_ratio * 100, 1),
            "Neutral%": round(neutral_ratio * 100, 1),
            "Negative%": round(negative_ratio * 100, 1),
            "Conversion_Score": round(final_score, 2)
        })
    print("✅ 评论分析完成")
    return pd.DataFrame(results)


# ==================== 主分析流水线 ====================
def run_full_analysis(classified_data, limit=10):
    # ===== 根据用户选择的 limit 截取数据 =====
    if limit > 0:
        classified_data = {
            "搜索页信息": classified_data["搜索页信息"][:limit],
            "listing信息": classified_data["listing信息"][:limit],
            "评论信息": classified_data["评论信息"][:limit]
        }
    # ==========================================

    total_start = time.time()
    progress_bar = st.progress(0, text="开始分析...")
    data = classified_data
    search_list = data.get("搜索页信息", [])
    listing_list = data.get("listing信息", [])
    review_list = data.get("评论信息", [])
    print(f"📊 数据加载完成，搜索页 {len(search_list)} 条，详情页 {len(listing_list)} 条，评论 {len(review_list)} 条")
    progress_bar.progress(5, "分类数据加载完成")

    # 提取图片URL
    print("🖼️ 正在提取图片URL...")
    image_rows = []
    for product in listing_list:
        asin = product.get("originalAsin")
        high = product.get("highResolutionImages", [])
        for idx, url in enumerate(high):
            image_rows.append({"asin": asin, "image_type": "high_resolution", "image_index": idx, "image_url": url})
        aplus = product.get("aPlusContent")
        if aplus:
            def extract_urls(obj):
                urls = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "url" and isinstance(v, str):
                            urls.append(v)
                        else:
                            urls.extend(extract_urls(v))
                elif isinstance(obj, list):
                    for item in obj:
                        urls.extend(extract_urls(item))
                return urls

            aplus_urls = extract_urls(aplus)
            for idx, url in enumerate(aplus_urls):
                image_rows.append({"asin": asin, "image_type": "a_plus", "image_index": idx, "image_url": url})
    for product in search_list:
        asin = product.get("asin")
        thumb = product.get("thumbnailImage")
        if thumb:
            image_rows.append({"asin": asin, "image_type": "thumbnail", "image_index": 0, "image_url": thumb})
    print(f"🖼️ 共提取 {len(image_rows)} 张图片URL")
    progress_bar.progress(15, f"提取图片 {len(image_rows)} 张")

    # 图片分析
    t_img = time.time()
    if image_rows:
        image_summary_df = aggregate_images(image_rows)
        image_dict = image_summary_df.set_index("asin").to_dict("index") if not image_summary_df.empty else {}
    else:
        image_dict = {}
    print(f"⏱️ 图片分析耗时: {time.time() - t_img:.2f} 秒")
    progress_bar.progress(40, "图片分析完成")

    # 心理评分
    t_psych = time.time()
    all_features = []
    for listing in listing_list:
        feats = listing.get("features", []) or []
        all_features.extend(feats)
    for item in search_list:
        title = item.get("title")
        if title and len(title.strip()) >= 3:
            all_features.append(title)
    print(f"🧠 收集到 {len(all_features)} 条待评分文本（包含标题和五点描述）")
    classifier = load_zero_shot()
    text2score = batch_psych_scores(all_features, classifier)
    print(f"⏱️ 心理评分耗时: {time.time() - t_psych:.2f} 秒")
    progress_bar.progress(55, "心理评分完成（含标题批量推理）")

    # 搜索页评分
    print("🔍 开始搜索页评分...")
    t_search = time.time()
    prices = [p.get("price", {}).get("value") for p in search_list if p.get("price")]
    search_results = []
    for idx, item in enumerate(search_list):
        asin = item.get("asin")
        print(f"  搜索 ASIN {idx + 1}/{len(search_list)}: {asin}")
        title = item.get("title")
        price = item.get("price", {}).get("value")
        stars = item.get("stars")
        reviews = item.get("reviewsCount")
        position = item.get("categoryPageData", {}).get("productPosition")
        # 使用新的标题评分函数
        title_score_val, title_details = compute_title_score(title, classifier)
        thumb_score = image_dict.get(asin, {}).get("thumbnail_score", 0)
        if price is not None and prices:
            valid_prices = [x for x in prices if x is not None]
            if valid_prices:
                rank = sum(x > price for x in valid_prices)
                price_score = round(rank / len(valid_prices) * 100, 2)
            else:
                price_score = 50
        else:
            price_score = 50
        if stars is None: stars = 0
        if reviews is None: reviews = 0
        rating = stars / 5
        review_norm = min(math.log(reviews + 1) / math.log(100000), 1) if reviews > 0 else 0
        trust = rating * 0.6 + review_norm * 0.4
        trust_score_val = round(trust * 100, 2)
        if position is not None and position > 0:
            pos_score = 1 / math.log(position + 2)
        else:
            pos_score = 0
        pos_score = round(pos_score * 100, 2)
        search_score = (title_score_val * 0.25 + thumb_score * 0.30 + price_score * 0.15 +
                        trust_score_val * 0.20 + pos_score * 0.10)
        search_score = round(search_score, 2)
        search_results.append({
            "asin": asin,
            "title_score": title_score_val,
            "thumbnail_score": thumb_score,
            "price_score": price_score,  # 保留得分（用于其他对比）
            "price_value": price,  # 新增：实际价格
            "trust_score": trust_score_val,
            "position_score": pos_score,
            "search_score": search_score,
            "stars": stars,
            "reviews": reviews
        })
        if (idx + 1) % 5 == 0 or idx == len(search_list) - 1:
            print(f"  搜索进度 {idx + 1}/{len(search_list)}")
    search_df = pd.DataFrame(search_results)
    search_df["search_rank"] = search_df["search_score"].rank(ascending=False, method="min")
    print(f"✅ 搜索页评分完成，耗时 {time.time() - t_search:.2f} 秒")
    progress_bar.progress(70, "搜索页评分完成")

    # Listing评分
    print("📄 开始详情页评分...")
    t_list = time.time()
    listing_results = []
    for idx, listing in enumerate(listing_list):
        asin = listing.get("originalAsin")
        if not asin:
            continue
        print(f"  详情 ASIN {idx + 1}/{len(listing_list)}: {asin}")
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
        img_s = score_images(asin, image_dict)
        brand_s = score_brandstory(brand)
        weights = {"features": 0.30, "attributes": 0.25, "important": 0.05, "aplus": 0.10, "video": 0.05, "image": 0.25}
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
        # 计算覆盖度
        cov_scores = []
        for feat in features:
            cov = check_coverage(feat, language='de')
            cov_scores.append(cov['total_score'])
        avg_cov = np.mean(cov_scores) if cov_scores else 0

        listing_results.append({
            "asin": asin,
            "bullet_score": feat_s,
            "attributes_score": attr_s,
            "important_score": imp_s,
            "aplus_score": aplus_s,
            "video_score": vid_s,
            "image_score": img_s,
            "brand_bonus": brand_bonus,
            "trust_bonus": round(trust_bonus, 2),
            "listing_score": round(final, 2),
            "coverage_score": round(avg_cov, 2)  # 新增
        })
        if (idx + 1) % 5 == 0 or idx == len(listing_list) - 1:
            print(f"  详情进度 {idx + 1}/{len(listing_list)}")
    listing_df = pd.DataFrame(listing_results)
    listing_df["listing_rank"] = listing_df["listing_score"].rank(ascending=False, method="min")
    print(f"✅ 详情页评分完成，耗时 {time.time() - t_list:.2f} 秒")
    progress_bar.progress(85, "Listing评分完成")

    # 评论分析
    print("📝 开始评论分析...")
    t_rev = time.time()
    sentiment_pipeline = load_sentiment_pipeline()
    review_df = analyze_reviews(review_list, sentiment_pipeline)
    print(f"✅ 评论分析完成，耗时 {time.time() - t_rev:.2f} 秒")
    progress_bar.progress(95, "评论分析完成")

    # 合并
    print("🔄 合并所有评分...")
    merged = pd.merge(search_df, listing_df, on="asin", how="inner")
    merged = pd.merge(merged, review_df, left_on="asin", right_on="ASIN", how="inner")
    if 'ASIN' in merged.columns:
        merged.drop(columns=['ASIN'], inplace=True)
    merged["Detail_Conversion"] = 0.6 * merged["listing_score"] + 0.4 * merged["Conversion_Score"]
    merged["Total_Score"] = 0.5 * merged["search_score"] + 0.5 * merged["Detail_Conversion"]
    merged = merged.sort_values("Total_Score", ascending=False)
    final_cols = ["asin", "search_score", "Detail_Conversion", "Total_Score",
                  "listing_score", "Conversion_Score", "search_rank"]
    final_df = merged[[c for c in final_cols if c in merged.columns]]
    progress_bar.progress(100, "分析完成！")
    print(f"✅ 全部分析完成！总耗时 {time.time() - total_start:.2f} 秒")
    return final_df, merged


# ==================== Streamlit UI ====================
st.title("📊 Amazon 产品竞争力分析仪表板")

# ===== 产品数量选择滑块 =====
max_items = st.number_input(
    "🔢 分析前多少个产品？（输入 0 表示分析全部）",
    min_value=0,
    max_value=500,
    value=10,
    step=1
)

uploaded_file = st.file_uploader("上传原始爬虫 JSON 文件 (dataset_free-amazon-product-scraper_*.json)", type=["json"])

if uploaded_file is not None:
    raw_data = json.load(uploaded_file)
    st.success("文件上传成功，正在分类...")

    if isinstance(raw_data, list) and not ("搜索页信息" in raw_data[0] if raw_data else False):
        with st.spinner("正在执行数据分类..."):
            classified_data = classify_raw_data(raw_data)
        st.success("分类完成，开始分析...")
    else:
        classified_data = raw_data

    with st.spinner("运行分析流水线，请耐心等待（首次运行需加载模型）..."):
        final_df, full_df = run_full_analysis(classified_data, max_items)

    st.subheader("📋 产品竞争力排名")
    # ========== 修改：将 ASIN 列显示为链接 ==========
    final_df_display = final_df.copy()
    final_df_display['商品链接'] = final_df_display['asin'].apply(
        lambda x: f'https://www.amazon.de/dp/{x}'
    )
    st.dataframe(
        final_df_display,
        column_config={
            "商品链接": st.column_config.LinkColumn("商品详情页", display_text=r'^(https://www\.amazon\.de/dp/)(.*)'),
            "asin": None,  # 隐藏原始ASIN列
        },
        width='stretch'
    )
    # ===============================================

    csv = final_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 下载结果 CSV", data=csv, file_name="product_competitiveness_final.csv", mime="text/csv")

    st.subheader("📈 数据可视化")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(final_df.set_index("asin")["Total_Score"].head(10))
        st.caption("Top10 综合总分")
    with col2:
        st.area_chart(final_df[["search_score", "Detail_Conversion", "Total_Score"]].head(20))
        st.caption("各维度趋势（前20）")

    col3, col4 = st.columns(2)
    with col3:
        st.line_chart(final_df[["search_score", "Detail_Conversion"]].head(30))
        st.caption("搜索分 vs 转化分（前30）")
    with col4:
        fig, ax = plt.subplots(figsize=(8, 4))
        final_df[["search_score", "Detail_Conversion", "Total_Score"]].boxplot(ax=ax)
        st.pyplot(fig)

    st.subheader("🔍 单品对比分析")
    asin_list = final_df["asin"].tolist()
    selected_asin = st.selectbox("选择或输入 ASIN", asin_list)

    if selected_asin:
        row = full_df[full_df["asin"] == selected_asin].iloc[0]
        numeric_cols = ["search_score", "Detail_Conversion", "Total_Score", "listing_score", "Conversion_Score"]
        means = full_df[numeric_cols].mean()

        st.subheader("📊 雷达图对比")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in numeric_cols],
            theta=numeric_cols,
            fill='toself',
            name=selected_asin
        ))
        fig.add_trace(go.Scatterpolar(
            r=[means[c] for c in numeric_cols],
            theta=numeric_cols,
            fill='toself',
            name='整体平均'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("✨ 亮点与痛点")
        diff = {dim: row[dim] - means[dim] for dim in numeric_cols}
        highlights = [dim for dim, d in diff.items() if d > 0.2 * means[dim]]
        painpoints = [dim for dim, d in diff.items() if d < -0.2 * means[dim]]
        if highlights:
            st.success(f"💡 亮点维度：{', '.join(highlights)}（高于平均20%以上）")
        else:
            st.info("暂无显著亮点")
        if painpoints:
            st.error(f"⚠️ 痛点维度：{', '.join(painpoints)}（低于平均20%以上）")
        else:
            st.info("暂无显著痛点")

    # ==================== 新增：新品竞争力预测与优化建议 ====================
    # ==================== 新品竞争力预测与优化建议（细化版 + 图片上传） ====================
    st.subheader("📝 新品竞争力预测与优化建议")

    with st.form("new_product_form"):
        st.markdown("**填写你的产品信息（点击下方按钮后分析）**")
        col1, col2 = st.columns(2)
        with col1:
            new_title = st.text_input("产品标题", placeholder="例如: Premium Bluetooth Headphones")
            new_price = st.number_input("价格 (€)", min_value=0.0, step=0.01, value=29.99)
            new_stars = st.number_input("星级评分 (1-5)", min_value=0.0, max_value=5.0, step=0.1, value=4.3)
            new_reviews = st.number_input("评论数量", min_value=0, step=1, value=120)
        with col2:
            new_features = st.text_area("五点描述（每行一条）", placeholder="每条描述占一行", height=150)
            has_aplus = st.checkbox("是否有 A+ 内容")
            has_brandstory = st.checkbox("是否有品牌故事")
            video_count = st.selectbox("视频数量", [0, 1, 2, 3, 5], index=0)

        # 新增：图片上传（支持多张）
        st.markdown("**上传产品图片（用于图片维度分析，最多5张）**")
        uploaded_images = st.file_uploader(
            "选择图片（支持 jpg/png）",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        if uploaded_images and len(uploaded_images) > 5:
            st.warning("最多分析5张图片，已自动截取前5张")
            uploaded_images = uploaded_images[:5]

        submitted = st.form_submit_button("📊 分析新品竞争力", type="primary")

    # ========== 只有当按钮被点击后才执行分析 ==========
    if submitted:
        if final_df.empty:
            st.warning("当前没有可对比的数据集，请先上传 JSON 文件。")
        else:
            # 1. 获取分类器（用于心理评分）
            classifier = load_zero_shot()

            # ========= 2. 计算文本维度得分 =========
            # 2a. 标题详细分析
            # 使用新标题评分函数
            title_score_val, title_details = compute_title_score(new_title, classifier)
            has_num = title_details.get('has_num', False)
            has_unit = title_details.get('has_unit', False)
            title_word_count = len(new_title.split()) if new_title else 0
            info_hit_title = title_details.get('hit', {})

            # 2b. 价格得分
            # 从 full_df 中获取所有产品的实际价格
            price_values = full_df['price_value'].dropna().tolist()
            if price_values and new_price:
                rank = sum(x > new_price for x in price_values)
                price_score = round(rank / len(price_values) * 100, 2) if price_values else 50
            else:
                price_score = 50

            # 2c. 信任得分
            rating = new_stars / 5
            review_norm = min(math.log(new_reviews + 1) / math.log(100000), 1) if new_reviews > 0 else 0
            trust_score_val = round((rating * 0.6 + review_norm * 0.4) * 100, 2)

            # 2d. 五点描述逐条分析（列表）
            features_list = [f.strip() for f in new_features.split('\n') if f.strip()]
            feat_scores = []
            feat_details = []
            if features_list and classifier:
                temp_text2score = batch_psych_scores(features_list, classifier)
                for idx, feat in enumerate(features_list):
                    single_score = score_features_batch([feat], temp_text2score)
                    has_user_mention = 'you' in feat.lower() or 'your' in feat.lower()
                    has_differentiation = any(
                        w in feat.lower() for w in ['unique', 'different', 'exclusive', 'only', 'best'])
                    feat_scores.append(single_score)
                    cov_result = check_coverage(feat, language='de')
                    info_hit = check_info_points(feat, language='de')
                    info_score = (sum(info_hit.values()) / len(info_hit)) * 100
                    feat_details.append({
                        'text': feat[:50] + '...' if len(feat) > 50 else feat,
                        'score': single_score,
                        'has_user_mention': has_user_mention,
                        'has_differentiation': has_differentiation,
                        'coverage': cov_result['coverage'],
                        'coverage_score': cov_result['total_score'],
                        'info_hit': info_hit,
                        'info_score': round(info_score, 2)
                    })
                feat_avg = sum(feat_scores) / len(feat_scores) if feat_scores else 0
            else:
                feat_avg = 0
                feat_details = []

            # 2e. 图片分析（如果上传了图片）
            img_scores = []
            img_details = []
            if uploaded_images:
                # 加载CLIP模型
                clip_model, clip_processor = load_clip()
                for img_file in uploaded_images:
                    try:
                        img = Image.open(img_file).convert("RGB")
                        # 调用analyze_image_with_clip需要URL，我们直接传入图片对象
                        # 修改analyze_image_with_clip支持PIL Image直接传入（复制函数稍改）
                        # 这里为了简化，我们直接调用底层CLIP
                        inputs = clip_processor(text=TEXT_PROMPTS, images=img, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            outputs = clip_model(**inputs)
                        logits = outputs.logits_per_image[0]
                        positive = logits[:-1]
                        baseline = logits[-1]
                        scores = torch.sigmoid(positive - baseline)
                        result = {name: float(scores[i]) for i, name in enumerate(FEATURE_NAMES)}
                        # 计算消费者得分（类似thumbnail/high_resolution，这里统一用高分辨率权重）
                        consumer_score = calculate_consumer_score({
                            'image_type': 'high_resolution',
                            'attention': result.get('attention', 0),
                            'product_understanding': result.get('product_understanding', 0),
                            'quality_perception': result.get('quality_perception', 0),
                            'differentiation': result.get('differentiation', 0),
                            'value_perception': result.get('value_perception', 0),
                            'usage_imagination': result.get('usage_imagination', 0),
                            'risk_reduction': result.get('risk_reduction', 0),
                            'trust_signal': result.get('trust_signal', 0)
                        })
                        img_scores.append(consumer_score)
                        img_details.append({
                            'name': img_file.name,
                            'score': consumer_score,
                            'trust': result.get('trust_signal', 0) * 100,
                            'quality': result.get('quality_perception', 0) * 100,
                            'value': result.get('value_perception', 0) * 100,
                            'usage': result.get('usage_imagination', 0) * 100,
                            'risk': result.get('risk_reduction', 0) * 100
                        })
                    except Exception as e:
                        st.warning(f"图片 {img_file.name} 分析失败: {e}")
                img_avg = np.mean(img_scores) if img_scores else 0
            else:
                img_avg = full_df['image_score'].mean() if 'image_score' in full_df.columns else 20.0
                img_details = []

            # 2f. 视频得分
            vid_s = score_video(video_count)

            # 2g. A+ 和品牌故事
            aplus_score = 50 if has_aplus else 0
            brand_bonus = 10 if has_brandstory else 0

            # 2h. 属性等（取均值）
            attr_s = full_df['attributes_score'].mean() if 'attributes_score' in full_df.columns else 50
            imp_s = full_df['important_score'].mean() if 'important_score' in full_df.columns else 50

            # 2i. 计算最终得分
            weights = {"features": 0.30, "attributes": 0.25, "important": 0.05, "aplus": 0.10, "video": 0.05,
                       "image": 0.25}
            base = (feat_avg * weights["features"] + attr_s * weights["attributes"] +
                    imp_s * weights["important"] + aplus_score * weights["aplus"] +
                    vid_s * weights["video"] + img_avg * weights["image"])
            listing_score = min(base + brand_bonus + trust_score_val * 0.1, 100)

            avg_thumb = full_df['thumbnail_score'].mean() if 'thumbnail_score' in full_df.columns else 50
            avg_position = full_df['position_score'].mean() if 'position_score' in full_df.columns else 50
            search_score = (title_score_val * 0.25 + avg_thumb * 0.30 + price_score * 0.15 +
                            trust_score_val * 0.20 + avg_position * 0.10)
            search_score = round(search_score, 2)

            conv_score = full_df['Conversion_Score'].mean() if 'Conversion_Score' in full_df.columns else 50
            detail_conversion = 0.6 * listing_score + 0.4 * conv_score
            total_score = 0.5 * search_score + 0.5 * detail_conversion

            # ========= 3. 显示详细对比表格 =========
            st.subheader("📊 新品 vs 数据集 细分得分对比")
            new_product = pd.DataFrame({
                '维度': ['标题得分', '价格得分', '信任得分', '五点描述得分', '图片得分', '视频得分', 'A+得分',
                         '搜索得分', '详情得分', '转化得分', '综合总分'],
                '新品得分': [title_score_val, price_score, trust_score_val, round(feat_avg, 2), round(img_avg, 2),
                             vid_s, aplus_score, search_score, round(listing_score, 2), round(conv_score, 2),
                             round(total_score, 2)]
            })
            avg_vals = {
                '标题得分': full_df['title_score'].mean() if 'title_score' in full_df.columns else 50,
                '价格得分': full_df['price_score'].mean() if 'price_score' in full_df.columns else 50,
                '信任得分': full_df['trust_score'].mean() if 'trust_score' in full_df.columns else 50,
                '五点描述得分': full_df['bullet_score'].mean() if 'bullet_score' in full_df.columns else 50,
                '图片得分': full_df['image_score'].mean() if 'image_score' in full_df.columns else 50,
                '视频得分': full_df['video_score'].mean() if 'video_score' in full_df.columns else 50,
                'A+得分': full_df['aplus_score'].mean() if 'aplus_score' in full_df.columns else 50,
                '搜索得分': full_df['search_score'].mean() if 'search_score' in full_df.columns else 50,
                '详情得分': full_df['listing_score'].mean() if 'listing_score' in full_df.columns else 50,
                '转化得分': full_df['Conversion_Score'].mean() if 'Conversion_Score' in full_df.columns else 50,
                '综合总分': full_df['Total_Score'].mean() if 'Total_Score' in full_df.columns else 50
            }
            new_product['数据集平均'] = new_product['维度'].map(avg_vals)
            new_product['差值'] = new_product['新品得分'] - new_product['数据集平均']
            st.dataframe(new_product, use_container_width=True, hide_index=True)

            # ========= 4. 细化优化建议 =========
            st.subheader("💡 针对性优化建议（细化版）")

            # 4a. 标题具体建议
            title_advice = []
            if title_score_val < avg_vals['标题得分'] - 5:
                if not new_title:
                    title_advice.append("❌ **标题为空**，请填写标题。")
                else:
                    if len(new_title) < 30:
                        title_advice.append(
                            "🔤 **标题过短**（当前 {} 字符），建议增加到 50-80 字符，包含品牌、核心关键词、规格。".format(
                                len(new_title)))
                    if not has_num:
                        title_advice.append(
                            "🔢 **标题缺少数字**（如容量、尺寸、功率），建议添加具体规格，例如 '500ml'、'24W' 等。")
                    if not has_unit:
                        title_advice.append("📏 **标题缺少单位**（如 cm, kg, W），建议加入单位词增强专业性。")
                    if title_word_count < 5:
                        title_advice.append(
                            "📝 **标题词数过少**（当前 {} 词），建议使用 5-10 个关键词组合。".format(title_word_count))
                    # 检查是否包含品牌名（简单假设品牌为第一个词）
                    if new_title.split()[0].lower() not in ['premium', 'professional', 'high', 'quality']:
                        title_advice.append(
                            "🏷️ **建议在标题开头加入品牌名或强度词**（如 Premium, Professional），提升品质感。")
            else:
                title_advice.append("✅ 标题得分较高，继续保持。")
            # 检查标题信息点缺失
            info_hit_title = title_details.get('hit', {})
            missing_info_title = [dim for dim, val in info_hit_title.items() if val == 0]
            if missing_info_title:
                info_dim_map = {
                    'material': '材质', 'color': '颜色', 'size': '尺寸/规格',
                    'function': '功能', 'scenario': '使用场景'
                }
                missing_desc = ', '.join([info_dim_map.get(m, m) for m in missing_info_title])
                title_advice.append(f"📋 **标题缺少信息点**：{missing_desc}，建议补充这些关键词以提升搜索覆盖。")

            # 4b. 五点描述具体建议
            feat_advice = []
            if feat_avg < avg_vals['五点描述得分'] - 5:
                if not features_list:
                    feat_advice.append("❌ **五点描述为空**，请至少填写 3-5 条卖点。")
                else:
                    # 逐条分析
                    for i, detail in enumerate(feat_details):
                        issues = []
                        # 1. 得分
                        if detail['score'] < 50:
                            issues.append("得分较低")
                        # 2. 用户导向
                        if not detail.get('has_user_mention', False):
                            issues.append("缺少'您/你的'等用户导向词汇")
                        # 3. 差异化
                        if not detail.get('has_differentiation', False):
                            issues.append("缺少差异化卖点（如'独特'、'独家'）")
                        # 4. 长度
                        if len(detail['text']) < 10:
                            issues.append("描述过短")
                        # 5. 信息覆盖度缺失
                        coverage = detail.get('coverage', {})
                        # 合并覆盖度缺失 + 具体信息点缺失
                        missing = [dim for dim, val in detail.get('coverage', {}).items() if val == 0]
                        info_hit = detail.get('info_hit', {})
                        missing_info = [dim for dim, val in info_hit.items() if val == 0]

                        # 统一映射表
                        dim_map = {
                            'size': '尺寸/规格',
                            'material': '材质',
                            'warranty': '保修/售后',
                            'usage': '使用场景',
                            'differentiation': '差异化优势',
                            'user_oriented': '用户导向',
                            'color': '颜色',
                            'function': '功能',
                            'scenario': '使用场景'
                        }
                        # 合并去重
                        all_missing = set(missing) | set(missing_info)
                        if all_missing:
                            missing_desc = ', '.join([dim_map.get(m, m) for m in all_missing])
                            issues.append(f"缺少信息维度：{missing_desc}")
                        if issues:
                            feat_advice.append(
                                f"📌 第 {i + 1} 条（{detail['text']}）：{'；'.join(issues)}。"
                            )
                            # 给出具体优化方向
                            if "得分较低" in issues:
                                feat_advice.append("   - 建议重新组织语言，突出核心卖点，增强说服力。")
                            if "缺少'您/你的'等用户导向词汇" in issues:
                                feat_advice.append("   - 可加入'您将获得…'、'为您设计'等表达，拉近与用户距离。")
                            if "缺少差异化卖点" in issues:
                                feat_advice.append("   - 强调与竞品不同的独特功能或设计，展示独家优势。")
                            if "描述过短" in issues:
                                feat_advice.append("   - 扩展描述，说明该特点如何解决用户的实际问题。")
                            if "缺少信息" in issues[0]:
                                feat_advice.append("   - 请补充上述缺失的信息维度，使描述更完整、更有说服力。")
                    if len(features_list) < 3:
                        feat_advice.append(
                            "📋 **五点描述数量不足**（当前 {} 条），建议至少 3 条，最好 5 条。".format(len(features_list)))
            else:
                feat_advice.append("✅ 五点描述得分较高，继续保持。")

            # 4c. 图片具体建议
            img_advice = []
            if uploaded_images:
                # 与数据集图片得分对比（如果数据集有图片得分）
                dataset_img_mean = full_df['image_score'].mean() if 'image_score' in full_df.columns else 50
                if img_avg < dataset_img_mean - 5:
                    # 细化到心理维度
                    # 计算各维度平均（从img_details）
                    avg_trust = np.mean([d['trust'] for d in img_details]) if img_details else 0
                    avg_quality = np.mean([d['quality'] for d in img_details]) if img_details else 0
                    avg_value = np.mean([d['value'] for d in img_details]) if img_details else 0
                    avg_usage = np.mean([d['usage'] for d in img_details]) if img_details else 0
                    avg_risk = np.mean([d['risk'] for d in img_details]) if img_details else 0
                    img_advice.append("🖼️ **图片得分偏低**，具体建议：")
                    if avg_trust < 60:
                        img_advice.append("   - **信任感不足**：图片不够专业或清晰，建议使用白底高清图，或加入实物对比图。")
                    if avg_quality < 60:
                        img_advice.append("   - **品质感知弱**：图片看起来廉价，建议提升拍摄光线和构图，突出产品材质。")
                    if avg_value < 60:
                        img_advice.append("   - **价值感不强**：图片未展示产品附加价值，可加入赠品或功能标注。")
                    if avg_usage < 60:
                        img_advice.append("   - **使用场景缺失**：建议加入 1-2 张使用场景图，让用户想象实际应用。")
                    if avg_risk < 60:
                        img_advice.append("   - **风险消除不足**：建议加入尺寸标注、功能介绍图，降低购买疑虑。")
                else:
                    img_advice.append("✅ 图片得分较高，继续保持。")
            else:
                img_advice.append("ℹ️ 未上传图片，无法提供图片优化建议。建议上传主图、场景图进行诊断。")

            # 4d. 其他维度建议
            other_advice = []
            if price_score < avg_vals['价格得分'] - 5:
                other_advice.append("💰 **价格竞争力不足**：当前价格高于数据集同类产品，建议适当降价或提供更多赠品。")
            if trust_score_val < avg_vals['信任得分'] - 5:
                other_advice.append("⭐ **信任得分偏低**：星级评分或评论数量不足，建议邀请用户留评、展示售后保障。")
            if aplus_score < avg_vals['A+得分'] - 5 and not has_aplus:
                other_advice.append("📄 **A+内容缺失**：建议添加 A+ 页面或品牌故事，增强品牌背书。")
            if vid_s < avg_vals['视频得分'] - 5:
                other_advice.append("🎬 **视频得分偏低**：缺少产品视频，建议制作 1-2 个使用演示或介绍视频。")

            # 汇总所有建议
            all_advice = title_advice + feat_advice + img_advice + other_advice
            if not all_advice or all(a.startswith("✅") for a in all_advice):
                st.success("🎉 新品各项指标均优于或接近数据集平均水平，竞争力较强！")
            else:
                for a in all_advice:
                    if a.startswith("✅"):
                        st.markdown(f"<span style='color:green'>{a}</span>", unsafe_allow_html=True)
                    elif a.startswith("❌") or a.startswith("⚠️"):
                        st.markdown(f"<span style='color:red'>{a}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:#FF8C00'>{a}</span>", unsafe_allow_html=True)
            # ========= 5. 雷达图 =========
            core_dims = ['搜索得分', '详情得分', '转化得分', '综合总分', '信任得分']
            core_new = [title_score_val, round(listing_score, 2), round(conv_score, 2), round(total_score, 2),
                        trust_score_val]
            core_avg = [avg_vals[d] for d in core_dims]
            st.subheader("📊 核心维度雷达图对比")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=core_new,
                theta=core_dims,
                fill='toself',
                name='新品'
            ))
            fig.add_trace(go.Scatterpolar(
                r=core_avg,
                theta=core_dims,
                fill='toself',
                name='数据集平均'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("请上传原始爬虫 JSON 文件开始分析")
