"""
Amazon 产品竞争力分析 v2.0 (优化版)
====================================
核心改进：
1. 分层分析架构
   - 快速层(默认)：纯关键词+数字指标，100 产品 < 30 秒
   - 深度层(按需)：单 ASIN 触发 CLIP/Zero-shot 深度分析
2. 智能新品对比
   - 基于分位数(P25/P50/P75)而非均值
   - 自动找出最接近的 3 个竞品做具体对比
   - 建议根据真实差距动态生成，不再千篇一律
3. 性能优化
   - 图片采样：每产品最多 3 张 + ThreadPool 并发下载
   - 评论分析：用 Amazon 原生 starsBreakdown，跳过 sentiment 模型
   - 心理评分：关键词词典替代 zero-shot
   - 覆盖度：fallback 关键词方案默认启用，GLiNER 可选
4. 缓存
   - 数据集级缓存：相同数据集二次分析秒级返回
   - 图片缓存：持久化到磁盘
   - 模型懒加载：首次深度分析时才加载
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
from io import BytesIO
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# ==================== 页面配置 ====================
st.set_page_config(page_title="Amazon 产品竞争力分析 v2", layout="wide", page_icon="📊")
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
    "max_images_per_product": 3,      # 每产品最多分析几张图（快速层不分析，深度层用）
    "image_download_workers": 6,       # 图片下载并发数
    "image_cache_dir": "data/img_cache",
    "analysis_cache_dir": "data/analysis_cache",
    # 默认全部关闭重型模型，由用户在 UI 中按需开启
    "enable_clip": False,
    "enable_gliner": False,
    "enable_zero_shot": False,
    "enable_sentiment_model": False,
}

# 数据集缓存（避免重复分析）
def dataset_hash(data):
    """对数据集生成稳定 hash"""
    try:
        s = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:16]
    except Exception:
        return str(int(time.time()))


# ==================== 分类映射 ====================
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
        "reviewsLink", "starsBreakdown", "aiReviewsSummary",
        "productPageReviews", "productPageReviewsFromOtherCountries",
        "reviewImages", "totalRatings", "customerId"
    ]
}


def classify_raw_data(products):
    """将扁平的爬虫数据分类成三个列表"""
    classified = {"搜索页信息": [], "listing信息": [], "评论信息": []}
    for prod in products:
        search_item = {k: prod.get(k) for k in CLASSIFICATION["搜索页信息"] if k in prod}
        listing_item = {k: prod.get(k) for k in CLASSIFICATION["listing信息"] if k in prod}
        review_item = {k: prod.get(k) for k in CLASSIFICATION["评论信息"] if k in prod}
        # 保留原始数据中未分类的字段到 listing
        for k, v in prod.items():
            if k not in search_item and k not in listing_item and k not in review_item:
                listing_item[k] = v
        classified["搜索页信息"].append(search_item)
        classified["listing信息"].append(listing_item)
        classified["评论信息"].append(review_item)
    return classified


# ==================== 关键词评分系统（替代所有 NLP 模型） ====================
# 心理维度关键词（德语+英语，覆盖 Amazon.de 主流类目）
PSYCH_KEYWORDS = {
    "quality": ["premium", "hochwertig", "qualität", "profi", "erstklassig",
                "top quality", "certificate", "geprüft"],
    "convenience": ["einfach", "mühelos", "schnell", "bequem", "praktisch",
                    "easy", "handlich", "plug and play"],
    "cost_saving": ["sparen", "günstig", "preiswert", "bestes preis",
                    "value", "save", "billig", "discount"],
    "safety": ["sicher", "stabil", "robust", "rutschfest", "safety",
               "sichere", "kindersicher", "geprüfte sicherheit"],
    "social_status": ["luxus", "exklusiv", "designer", "elegant",
                      "premium design", "lifestyle"],
    "health": ["gesund", "ergonomisch", "atmungsaktiv", "schadstofffrei",
               "natural", "bio", "öko", "ungiftig"],
    "durability": ["langlebig", "dauerhaft", "robust", "widerstandsfähig",
                   "garantie", "jahre", "rostfrei"],
    "aesthetics": ["schön", "elegant", "modern", "stilvoll", "design",
                   "ästhetisch", "chic"],
    "innovation": ["innovativ", "neuartig", "einzigartig", "new",
                   "revolutionär", "smart", "intelligent"],
    "trust": ["vertrauen", "garantie", "zertifiziert", "geprüft",
              "trusted", "official", "autorisiert"],
}


def keyword_psych_score(text):
    """基于关键词的心理评分（0-100），替代 zero-shot 分类器"""
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
    """批量心理评分（关键词版），秒级完成"""
    return {t: keyword_psych_score(t) for t in texts if t and len(t.strip()) >= 3}


# ==================== 信息点检测（保留原逻辑） ====================
INFO_PATTERNS = {
    "material": {
        "en": r"leather|fabric|wood|metal|plastic|steel|oak|velvet|linen|pu|abs|cotton|polyester",
        "de": r"leder|stoff|holz|metall|kunststoff|stahl|eiche|samt|leinen|pu|abs|baumwolle|polyester",
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


def check_info_points(text, language="de"):
    """检测文本是否包含材质/颜色/尺寸/功能/场景等信息点"""
    lang = "de" if language.lower() in ["de", "ger"] else "en"
    text_lower = text.lower() if text else ""
    hit = {}
    for dim, lang_dict in INFO_PATTERNS.items():
        pattern = lang_dict.get(lang, lang_dict["en"])
        hit[dim] = 1 if re.search(pattern, text_lower, re.IGNORECASE) else 0
    return hit


# ==================== 覆盖度（关键词版，默认走这条） ====================
COVERAGE_KEYWORDS = {
    'de': {
        'size': ['größe', 'abmessungen', 'länge', 'breite', 'höhe', 'cm', 'mm', 'kg', 'g', 'ml', 'l', 'dimension'],
        'material': ['material', 'stoff', 'leder', 'kunststoff', 'holz', 'baumwolle', 'polyester', 'metall', 'samt'],
        'warranty': ['garantie', 'gewährleistung', '2-jährig', '1-jährig', 'lebenslang', 'rückgabe'],
        'usage': ['verwendung', 'geeignet für', 'perfekt für', 'ideal für', 'szenario', 'einsatz', 'anwendung'],
        'differentiation': ['einzigartig', 'exklusiv', 'nur', 'beste', 'anders als', 'vergleichen', 'überlegen', 'unique'],
        'user_oriented': ['sie', 'ihnen', 'ihr', 'benutzer', 'kunde', 'komfort', 'genießen', 'erleben', 'personalisiert']
    },
    'en': {
        'size': ['size', 'dimension', 'length', 'width', 'height', 'cm', 'mm', 'kg', 'g', 'ml', 'l'],
        'material': ['material', 'fabric', 'leather', 'plastic', 'metal', 'wood', 'cotton', 'polyester'],
        'warranty': ['warranty', 'guarantee', '2-year', '1-year', 'lifetime', 'return'],
        'usage': ['use', 'applicable', 'suitable for', 'perfect for', 'ideal for', 'scenario'],
        'differentiation': ['unique', 'exclusive', 'only', 'best', 'unlike', 'compare', 'superior'],
        'user_oriented': ['you', 'your', 'user', 'customer', 'comfort', 'enjoy', 'experience']
    }
}
COVERAGE_WEIGHTS = {'size': 0.2, 'material': 0.2, 'warranty': 0.1, 'usage': 0.15,
                    'differentiation': 0.2, 'user_oriented': 0.15}


def check_coverage(text, language='de'):
    """关键词版覆盖度检测（替代 GLiNER2，速度快 100 倍）"""
    if not text or len(text.strip()) < 5:
        return {'coverage': {}, 'total_score': 0}
    lang = 'de' if language in ['de', 'ger'] else 'en'
    kw = COVERAGE_KEYWORDS.get(lang, COVERAGE_KEYWORDS['en'])
    text_lower = text.lower()
    coverage = {dim: (1 if any(w in text_lower for w in words) else 0)
                for dim, words in kw.items()}
    total = sum(coverage[dim] * COVERAGE_WEIGHTS[dim] for dim in COVERAGE_WEIGHTS) * 100
    return {'coverage': coverage, 'total_score': round(total, 2)}


# ==================== 各维度评分（轻量版） ====================
def compute_title_score(title, text2score=None):
    """标题评分：长度 + 关键词心理分 + 信息点覆盖"""
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

    # 心理评分：优先用预计算结果，否则现算
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

    details = {
        "length": len_score,
        "psych": round(psych_score, 2),
        "info": round(info_score, 2),
        "has_num": has_num,
        "has_unit": has_unit,
        "hit": info_hit
    }
    return total, details


def score_features_batch(features, text2score):
    """五点描述评分"""
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
    if count is None or count == 0:
        return 0.0
    return 100.0 if count >= 3 else (90.0 if count == 2 else 70.0)


def score_brandstory(story):
    return 100.0 if story and story.get("items") else 0.0


def score_images_simple(asin, listing, image_dict=None):
    """快速层图片评分：基于图片数量+结构，不调 CLIP"""
    if image_dict and asin in image_dict:
        # 如果有 CLIP 结果（深度分析后），用之
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
        return round(main_score * 0.40 + detail_score * 0.40 + aplus_score * 0.20, 2)

    # 快速层：基于数量
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
                for item in obj:
                    n += _count_urls(item)
            return n
        aplus_count = _count_urls(aplus)

    high_count = len(high_imgs)
    gallery_count = len(gallery)
    total_img = high_count + gallery_count + aplus_count

    # 数量得分：1张=40, 3张=60, 5张=75, 7+张=90
    count_score = min(40 + total_img * 7.5, 90) if total_img > 0 else 20
    # 多样性加分：同时有主图+A+图
    diversity_bonus = 0
    if high_count > 0:
        diversity_bonus += 5
    if aplus_count > 0:
        diversity_bonus += 5

    return round(min(count_score + diversity_bonus, 100), 2)


# ==================== 评论分析（轻量版，用 Amazon 原生 starsBreakdown） ====================
def analyze_reviews_fast(review_data):
    """用 Amazon 自带的 starsBreakdown + aiReviewsSummary，跳过 sentiment 模型"""
    print(f"📝 快速评论分析，共 {len(review_data)} 个商品")
    results = []
    for idx, rev in enumerate(review_data):
        reviews_link = rev.get("reviewsLink", "")
        asin = reviews_link.split("/")[-1].split("?")[0] if reviews_link else None
        if not asin:
            asin = f"Unknown_{idx + 1}"

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

        # 评论数量（从 productPageReviews 也算上）
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

        total_comments = len(all_reviews)

        # 综合分（无需 sentiment 模型）
        stars_norm = avg_stars / 5 if avg_stars else 0.5
        pos_norm = ai_pos_ratio if ai_pos_ratio is not None else 0.5
        count_norm = min(math.log(total_comments + 1) / math.log(1001), 1.0) if total_comments else 0.0
        # 5星占比作为情感信号
        five_star_ratio = stars_br.get("5star", 0)
        one_star_ratio = stars_br.get("1star", 0)
        sent_proxy = five_star_ratio - one_star_ratio  # -1 ~ 1
        sent_norm = (sent_proxy + 1) / 2

        w = [0.30, 0.25, 0.20, 0.15, 0.10]
        final_score = (stars_norm * w[0] + sent_norm * w[1] + pos_norm * w[2] +
                       count_norm * w[3] + ai_pos_ratio * w[4]) * 100

        results.append({
            "ASIN": asin,
            "Avg_Stars": round(avg_stars, 2),
            "Total_Reviews_Count": total_comments,
            "5star_ratio": round(stars_br.get("5star", 0) * 100, 1),
            "1star_ratio": round(stars_br.get("1star", 0) * 100, 1),
            "Positive%": round(ai_pos_ratio * 100, 1),
            "Conversion_Score": round(final_score, 2)
        })
    print("✅ 快速评论分析完成")
    return pd.DataFrame(results)


# ==================== 主分析流水线（快速层） ====================
def run_fast_analysis(classified_data, limit=0, progress_callback=None):
    """快速分析：纯关键词+数字，无 NLP 模型，100 产品 < 30 秒"""
    def _update(p, msg):
        if progress_callback:
            progress_callback(p, msg)

    total_start = time.time()
    if limit > 0:
        classified_data = {
            "搜索页信息": classified_data["搜索页信息"][:limit],
            "listing信息": classified_data["listing信息"][:limit],
            "评论信息": classified_data["评论信息"][:limit]
        }

    data = classified_data
    search_list = data.get("搜索页信息", [])
    listing_list = data.get("listing信息", [])
    review_list = data.get("评论信息", [])

    print(f"📊 数据加载：搜索页 {len(search_list)}，详情页 {len(listing_list)}，评论 {len(review_list)}")
    _update(5, f"分类数据加载完成：{len(search_list)} 个产品")

    # ===== 心理评分（关键词版，秒级） =====
    t_psych = time.time()
    all_features = []
    for listing in listing_list:
        feats = listing.get("features", []) or []
        all_features.extend(feats)
    for item in search_list:
        title = item.get("title")
        if title and len(title.strip()) >= 3:
            all_features.append(title)
    text2score = batch_psych_scores_keyword(all_features)
    print(f"⏱️ 心理评分(关键词)耗时: {time.time() - t_psych:.2f} 秒, 处理 {len(all_features)} 条")
    _update(20, "心理评分完成")

    # ===== 搜索页评分 =====
    t_search = time.time()
    prices = [p.get("price", {}).get("value") for p in search_list if p.get("price")]
    valid_prices = [x for x in prices if x is not None]
    search_results = []
    for idx, item in enumerate(search_list):
        asin = item.get("asin")
        title = item.get("title")
        price = item.get("price", {}).get("value")
        stars = item.get("stars") or 0
        reviews = item.get("reviewsCount") or 0
        position = (item.get("categoryPageData") or {}).get("productPosition")

        title_score_val, title_details = compute_title_score(title, text2score)

        # 价格分（百分位）
        if price is not None and valid_prices:
            rank = sum(x > price for x in valid_prices)
            price_score = round(rank / len(valid_prices) * 100, 2)
        else:
            price_score = 50

        # 信任分
        rating = stars / 5 if stars else 0
        review_norm = min(math.log(reviews + 1) / math.log(100000), 1) if reviews > 0 else 0
        trust_score_val = round((rating * 0.6 + review_norm * 0.4) * 100, 2)

        # 位置分
        pos_score = round(1 / math.log(position + 2) * 100, 2) if position and position > 0 else 0

        # 缩略图分（快速层用默认 50，深度层用 CLIP 结果覆盖）
        thumb_score = 50.0

        search_score = (title_score_val * 0.25 + thumb_score * 0.30 + price_score * 0.15 +
                        trust_score_val * 0.20 + pos_score * 0.10)
        search_results.append({
            "asin": asin,
            "title": (title or "")[:80],
            "title_score": title_score_val,
            "thumbnail_score": thumb_score,
            "price_score": price_score,
            "price_value": price,
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
    _update(50, "搜索页评分完成")

    # ===== Listing 评分 =====
    t_list = time.time()
    listing_results = []
    for idx, listing in enumerate(listing_list):
        asin = listing.get("originalAsin")
        if not asin:
            continue
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
        img_s = score_images_simple(asin, listing)
        brand_s = score_brandstory(brand)

        weights = {"features": 0.30, "attributes": 0.25, "important": 0.05,
                   "aplus": 0.10, "video": 0.05, "image": 0.25}
        base = (feat_s * weights["features"] + attr_s * weights["attributes"] +
                imp_s * weights["important"] + aplus_s * weights["aplus"] +
                vid_s * weights["video"] + img_s * weights["image"])
        brand_bonus = 10 if brand_s > 0 else 0

        # 信任加成
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

        # 覆盖度（关键词版，毫秒级）
        cov_scores = []
        for feat in features:
            cov = check_coverage(feat, language='de')
            cov_scores.append(cov['total_score'])
        avg_cov = np.mean(cov_scores) if cov_scores else 0

        listing_results.append({
            "asin": asin,
            "brand": listing.get("brand", ""),
            "bullet_score": feat_s,
            "attributes_score": attr_s,
            "important_score": imp_s,
            "aplus_score": aplus_s,
            "video_score": vid_s,
            "image_score": img_s,
            "brand_bonus": brand_bonus,
            "trust_bonus": round(trust_bonus, 2),
            "listing_score": round(final, 2),
            "coverage_score": round(avg_cov, 2),
            # 保留原始信息用于深度分析
            "_features": features,
            "_attributes": attributes,
            "_title": (search_df[search_df["asin"] == asin].iloc[0]["title"]
                       if not search_df[search_df["asin"] == asin].empty else ""),
        })
    listing_df = pd.DataFrame(listing_results)
    if not listing_df.empty:
        listing_df["listing_rank"] = listing_df["listing_score"].rank(ascending=False, method="min")
    print(f"⏱️ Listing 评分耗时: {time.time() - t_list:.2f} 秒")
    _update(80, "详情页评分完成")

    # ===== 评论分析（快速版） =====
    t_rev = time.time()
    review_df = analyze_reviews_fast(review_list)
    print(f"⏱️ 评论分析耗时: {time.time() - t_rev:.2f} 秒")
    _update(95, "评论分析完成")

    # ===== 合并 =====
    merged = pd.merge(search_df, listing_df, on="asin", how="inner")
    merged = pd.merge(merged, review_df, left_on="asin", right_on="ASIN", how="inner")
    if 'ASIN' in merged.columns:
        merged.drop(columns=['ASIN'], inplace=True)
    merged["Detail_Conversion"] = 0.6 * merged["listing_score"] + 0.4 * merged["Conversion_Score"]
    merged["Total_Score"] = 0.5 * merged["search_score"] + 0.5 * merged["Detail_Conversion"]
    merged = merged.sort_values("Total_Score", ascending=False)

    final_cols = ["asin", "title", "brand", "search_score", "Detail_Conversion", "Total_Score",
                  "listing_score", "Conversion_Score", "search_rank", "price_value", "stars", "reviews"]
    final_df = merged[[c for c in final_cols if c in merged.columns]]

    _update(100, "分析完成！")
    print(f"✅ 全部分析完成！总耗时 {time.time() - total_start:.2f} 秒")
    return final_df, merged


# ==================== 深度分析（按需触发，单 ASIN） ====================
@st.cache_resource
def load_clip():
    """懒加载 CLIP 模型"""
    print("🖼️ 加载 CLIP 模型...")
    import torch
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("✅ CLIP 加载完成")
    return model, processor


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


def _load_image_from_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"   ⚠️ 图片下载失败: {url[:80]}... ({e})")
        return None


def _analyze_one_image_with_clip(url, model, processor, image_type='high_resolution'):
    import torch
    img = _load_image_from_url(url)
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


def calculate_consumer_score(features, image_type):
    """根据图片类型计算消费者得分"""
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


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def analyze_uploaded_images_with_clip(uploaded_image_bytes_list, image_types_list):
    """对用户上传的新品图片做 CLIP 深度分析
    Args:
        uploaded_image_bytes_list: tuple of bytes (Streamlit UploadedFile.getvalue())
        image_types_list: tuple of str, 每张图的角色
                          - 'main' (主图/缩略图)
                          - 'detail' (高分辨率细节图)
                          - 'lifestyle' (场景图，按高分辨率权重算)
                          - 'aplus' (A+ 图)
    Returns:
        dict: 包含每张图的特征分 + 整体聚合分 + 主图 embedding（用于找视觉相似竞品）
    """
    import torch
    if not uploaded_image_bytes_list:
        return {}

    model, processor = load_clip()

    per_image_results = []
    main_image_embedding = None  # 保存主图 embedding 用于视觉相似度
    main_image_index = None  # 记录哪张图被当作主图

    # 确定主图：优先用户标记的 'main'，否则用第一张图
    if 'main' in image_types_list:
        main_image_index = list(image_types_list).index('main')
    else:
        main_image_index = 0  # 默认第一张作为主图

    print(f"🖼️ 新品图片分析：共 {len(uploaded_image_bytes_list)} 张，主图索引 = {main_image_index}")

    for idx, (img_bytes, img_type) in enumerate(zip(uploaded_image_bytes_list, image_types_list)):
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"⚠️ 上传图片打开失败: {e}")
            continue

        try:
            # 1. 先单独提取 embedding（用 processor 处理图片，再用 get_image_features）
            img_inputs = processor(images=img, return_tensors="pt")
            pixel_values = img_inputs.get('pixel_values')
            if pixel_values is None:
                print(f"⚠️ 图 {idx}: processor 没返回 pixel_values")
                continue

            with torch.no_grad():
                img_emb = model.get_image_features(pixel_values=pixel_values)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # L2 归一化
                img_emb_np = img_emb[0].cpu().numpy()

            # 如果是主图，保存 embedding
            if idx == main_image_index:
                main_image_embedding = img_emb_np
                print(f"✅ 主图 embedding 提取成功，维度 = {img_emb_np.shape}")

            # 2. 用 text+image 一起做 zero-shot 分类（用于心理评分）
            inputs = processor(text=TEXT_PROMPTS, images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            positive = logits[:-1]
            baseline = logits[-1]
            scores_arr = torch.sigmoid(positive - baseline)
            features = {name: float(scores_arr[i]) for i, name in enumerate(FEATURE_NAMES)}
            consumer_type = {
                'main': 'thumbnail',
                'detail': 'high_resolution',
                'lifestyle': 'high_resolution',
                'aplus': 'a_plus',
            }.get(img_type, 'high_resolution')
            consumer = calculate_consumer_score(features, consumer_type)
            per_image_results.append({
                'image_type': img_type,
                'consumer_score': consumer,
                'features': features,
            })
        except Exception as e:
            import traceback
            print(f"⚠️ CLIP 推理失败 (图 {idx}): {e}")
            traceback.print_exc()
            continue

    if not per_image_results:
        return {}

    # 聚合：每张图都参与，主图权重高
    type_weights = {'main': 0.40, 'detail': 0.30, 'lifestyle': 0.20, 'aplus': 0.10}
    total_weight = sum(type_weights.get(r['image_type'], 0.1) for r in per_image_results)
    weighted_score = sum(r['consumer_score'] * type_weights.get(r['image_type'], 0.1)
                         for r in per_image_results) / total_weight if total_weight > 0 else 0

    # 各心理维度平均
    dim_avgs = {}
    for dim in FEATURE_NAMES:
        vals = [r['features'].get(dim, 0) * 100 for r in per_image_results]
        dim_avgs[dim] = sum(vals) / len(vals) if vals else 0

    # 各图类型平均分
    type_scores = {}
    for t in ['main', 'detail', 'lifestyle', 'aplus']:
        sub = [r for r in per_image_results if r['image_type'] == t]
        type_scores[t] = {
            'score': sum(r['consumer_score'] for r in sub) / len(sub) if sub else 0,
            'count': len(sub)
        }

    print(f"✅ 新品图片分析完成，main_image_embedding = {'有' if main_image_embedding is not None else '无'}")

    return {
        'overall_score': round(weighted_score, 2),
        'per_image': per_image_results,
        'dim_avgs': dim_avgs,
        'type_scores': type_scores,
        'image_count': len(per_image_results),
        'main_image_embedding': main_image_embedding,  # numpy array 或 None
        'main_image_index': main_image_index,
    }


@st.cache_data(show_spinner="🖼️ 计算数据集图片向量中（首次约 1-3 分钟，之后秒级）", ttl=3600 * 24 * 7)
def compute_dataset_image_embeddings(thumbnail_urls_tuple):
    """批量计算数据集中所有产品主图（thumbnail）的 CLIP embedding
    Args:
        thumbnail_urls_tuple: tuple of (asin, url)，必须可哈希
    Returns:
        dict: {asin: embedding (numpy array, L2 normalized)}
    """
    import torch
    if not thumbnail_urls_tuple:
        return {}

    model, processor = load_clip()
    embeddings = {}

    # 并发下载 + 串行推理（推理必须串行，下载可并发）
    def _download(url):
        try:
            resp = requests.get(url, timeout=10)
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None

    urls = [u for _, u in thumbnail_urls_tuple]
    with ThreadPoolExecutor(max_workers=CONFIG["image_download_workers"]) as executor:
        images = list(executor.map(_download, urls))

    # 批量推理（一次处理多张图，比逐张快很多）
    batch_size = 8
    valid_records = [(asin, img) for (asin, _), img in zip(thumbnail_urls_tuple, images) if img is not None]
    print(f"🖼️ 数据集图片：{len(thumbnail_urls_tuple)} 个 ASIN，{len(valid_records)} 张下载成功")

    for i in range(0, len(valid_records), batch_size):
        batch = valid_records[i:i + batch_size]
        batch_imgs = [img for _, img in batch]
        batch_asins = [asin for asin, _ in batch]
        try:
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
            pixel_values = inputs.get('pixel_values')
            if pixel_values is None:
                print(f"⚠️ 批量 processor 没返回 pixel_values")
                continue
            with torch.no_grad():
                embs = model.get_image_features(pixel_values=pixel_values)
                embs = embs / embs.norm(dim=-1, keepdim=True)  # L2 归一化
            for asin, emb in zip(batch_asins, embs):
                embeddings[asin] = emb.cpu().numpy()
        except Exception as e:
            import traceback
            print(f"⚠️ 批量 embedding 失败 (batch {i//batch_size}): {e}")
            traceback.print_exc()
            continue
        if (i + batch_size) % 32 == 0 or i + len(batch) >= len(valid_records):
            print(f"  embedding 进度 {min(i + batch_size, len(valid_records))}/{len(valid_records)}")

    print(f"✅ 数据集图片 embedding 完成，{len(embeddings)}/{len(thumbnail_urls_tuple)}")
    return embeddings


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def deep_analyze_asin_images(_listing, _search_item, max_images=3):
    """对单个 ASIN 的图片做 CLIP 深度分析（带缓存）
    注意: _listing 和 _search_item 前缀下划线告诉 Streamlit 不要 hash 它们的内容
    """
    import torch
    model, processor = load_clip()

    asin = _listing.get("originalAsin") or _search_item.get("asin")
    image_records = []

    # 主图（thumbnail）
    thumb = _search_item.get("thumbnailImage")
    if thumb:
        image_records.append({"image_type": "thumbnail", "image_url": thumb})

    # 高分辨率图（采样 max-1 张）
    high = _listing.get("highResolutionImages", []) or []
    for url in high[:max_images - 1]:
        image_records.append({"image_type": "high_resolution", "image_url": url})

    # A+ 图（1 张）
    aplus = _listing.get("aPlusContent")
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
                for item in obj:
                    urls.extend(_extract_urls(item))
            return urls
        aplus_urls = _extract_urls(aplus)
        if aplus_urls:
            image_records.append({"image_type": "a_plus", "image_url": aplus_urls[0]})

    if not image_records:
        return {}

    # 并发下载+推理
    results = []
    with ThreadPoolExecutor(max_workers=CONFIG["image_download_workers"]) as executor:
        futures = {executor.submit(_analyze_one_image_with_clip, rec["image_url"],
                                   model, processor, rec["image_type"]): rec
                   for rec in image_records}
        for future in as_completed(futures):
            rec = futures[future]
            try:
                features = future.result()
                if features:
                    consumer = calculate_consumer_score(features, rec["image_type"])
                    results.append({
                        "image_type": rec["image_type"],
                        "image_url": rec["image_url"],
                        "consumer_score": consumer,
                        **features
                    })
            except Exception as e:
                print(f"⚠️ 图片分析失败: {e}")

    if not results:
        return {}

    # 聚合
    summary = {"asin": asin}
    df = pd.DataFrame(results)
    for itype, prefix in [("thumbnail", "thumbnail"), ("high_resolution", "detail"), ("a_plus", "aplus")]:
        sub = df[df["image_type"] == itype]
        if len(sub) > 0:
            summary[f"{prefix}_score"] = sub["consumer_score"].mean()
            summary[f"{prefix}_attention"] = sub.get("attention", pd.Series([0])).mean() if itype == "thumbnail" else 0
            summary[f"{prefix}_purchase"] = sub.get("purchase_intent", pd.Series([0])).mean() if itype == "thumbnail" else 0
            summary[f"{prefix}_quality"] = sub.get("quality_perception", pd.Series([0])).mean()
            summary[f"{prefix}_trust"] = sub.get("trust_signal", pd.Series([0])).mean()
            summary[f"{prefix}_value"] = sub.get("value_perception", pd.Series([0])).mean()
            summary[f"{prefix}_usage"] = sub.get("usage_imagination", pd.Series([0])).mean()
            summary[f"{prefix}_risk"] = sub.get("risk_reduction", pd.Series([0])).mean()
            if itype == "high_resolution":
                summary["detail_image_count"] = len(sub)
                summary["detail_visual_score"] = sub["consumer_score"].mean()
                summary["detail_best_score"] = sub["consumer_score"].max()
            if itype == "a_plus":
                summary["aplus_count"] = len(sub)
                summary["aplus_mean_score"] = sub["consumer_score"].mean()
                summary["aplus_brand"] = sub.get("differentiation", pd.Series([0])).mean()
        else:
            summary[f"{prefix}_score"] = 0
    return summary


# ==================== 智能新品对比（核心改进） ====================
def compute_quantiles(series, qs=(0.25, 0.5, 0.75)):
    """计算分位数，处理空值"""
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return {q: 50 for q in qs}
    return {q: float(s.quantile(q)) for q in qs}


def find_top_competitors(full_df, new_title, new_price, top_n=3,
                         new_image_embedding=None, dataset_embeddings=None,
                         image_weight=0.45, title_weight=0.35, price_weight=0.20):
    """根据标题相似度 + 价格接近度 + 图片视觉相似度找出最相似的 N 个竞品
    Args:
        new_image_embedding: numpy array, 新品主图的 CLIP embedding（L2 归一化）
        dataset_embeddings: dict {asin: embedding}, 数据集所有产品的 embedding
        image_weight: 图片相似度权重（仅当 new_image_embedding 和 dataset_embeddings 都有值时生效）
        title_weight: 标题相似度权重
        price_weight: 价格接近度权重
    """
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

    # 2. 价格接近度
    if new_price and 'price_value' in df.columns:
        prices = df['price_value'].dropna()
        if len(prices) > 0:
            price_range = max(prices.max() - prices.min(), 1)
            df['price_sim'] = 1 - (df['price_value'].fillna(prices.mean()) - new_price).abs() / price_range
        else:
            df['price_sim'] = 0.5
    else:
        df['price_sim'] = 0.5

    # 3. 图片视觉相似度（如果新品和数据集都有 embedding）
    has_image_sim = (new_image_embedding is not None and
                     dataset_embeddings and
                     len(dataset_embeddings) > 0)
    if has_image_sim:
        import numpy as np
        def _cosine_sim(asin):
            emb = dataset_embeddings.get(asin)
            if emb is None:
                return None
            # 两者都已 L2 归一化，点积 = 余弦相似度
            return float(np.dot(new_image_embedding, emb))

        df['image_sim'] = df['asin'].apply(_cosine_sim)
        # 缺失的图片相似度用中位数填充（避免拉低排名）
        valid_sims = df['image_sim'].dropna()
        fill_val = valid_sims.median() if len(valid_sims) > 0 else 0.5
        df['image_sim'] = df['image_sim'].fillna(fill_val)

        # 综合相似度：标题 + 价格 + 图片
        df['sim'] = (df['title_sim'] * title_weight +
                     df['price_sim'] * price_weight +
                     df['image_sim'] * image_weight)
        df['sim_breakdown'] = df.apply(
            lambda r: f"标题 {r['title_sim']:.2f} · 价格 {r['price_sim']:.2f} · 图片 {r['image_sim']:.2f}",
            axis=1
        )
    else:
        # 没有图片 embedding，退化为标题+价格
        df['image_sim'] = None
        # 重新归一化权重
        total = title_weight + price_weight
        df['sim'] = (df['title_sim'] * (title_weight / total) +
                     df['price_sim'] * (price_weight / total))
        df['sim_breakdown'] = df.apply(
            lambda r: f"标题 {r['title_sim']:.2f} · 价格 {r['price_sim']:.2f}",
            axis=1
        )

    top = df.nlargest(top_n, 'sim')
    return top, df


def analyze_new_product_smart(new_title, new_price, new_stars, new_reviews,
                              new_features, has_aplus, has_brandstory,
                              video_count, full_df, classifier_text2score=None,
                              image_analysis_result=None,
                              dataset_image_embeddings=None,
                              thumbnail_urls_map=None,
                              top_n_competitors=5):
    """智能新品分析：基于分位数 + 真实竞品对比
    Args:
        image_analysis_result: dict, 由 analyze_uploaded_images_with_clip 返回。
                               None 表示用户未上传图片，用数据集中位数。
        dataset_image_embeddings: dict {asin: embedding}, 数据集所有 thumbnail 的 CLIP embedding
        thumbnail_urls_map: dict {asin: thumbnail_url}, 用于在 UI 中显示竞品图片
        top_n_competitors: int, 找多少个最相似竞品（3-10）
    """
    # ===== 计算数据集分位数基准 =====
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
        'stars': compute_quantiles(full_df.get('stars', pd.Series()), qs),
        'reviews': compute_quantiles(full_df.get('reviews', pd.Series()), qs),
        'coverage_score': compute_quantiles(full_df.get('coverage_score', pd.Series()), qs),
    }

    # ===== 新品各维度得分 =====
    title_score_val, title_details = compute_title_score(new_title, classifier_text2score)

    # 价格分（百分位）
    price_values = full_df['price_value'].dropna().tolist() if 'price_value' in full_df.columns else []
    if price_values and new_price:
        rank = sum(x > new_price for x in price_values)
        price_score = round(rank / len(price_values) * 100, 2)
    else:
        price_score = 50

    # 信任分
    rating = new_stars / 5 if new_stars else 0
    review_norm = min(math.log(new_reviews + 1) / math.log(100000), 1) if new_reviews > 0 else 0
    trust_score_val = round((rating * 0.6 + review_norm * 0.4) * 100, 2)

    # 五点描述
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
            feat_details.append({
                'text': feat,
                'score': score_features_batch([feat], temp_text2score),
                'coverage': cov['coverage'],
                'coverage_score': cov['total_score'],
                'info_hit': info_hit,
                'info_score': round(info_score, 2),
                'has_user_mention': bool(re.search(r'\b(sie|ihnen|ihr|you|your)\b', feat.lower())),
                'has_differentiation': any(w in feat.lower() for w in
                    ['unique', 'different', 'exclusive', 'only', 'best', 'einzigartig', 'exklusiv', 'beste']),
                'length': len(feat),
            })
    else:
        feat_avg = 0
        feat_details = []

    # 视频得分
    vid_s = score_video(video_count)
    aplus_score = 50 if has_aplus else 0
    brand_bonus = 10 if has_brandstory else 0

    # 图片：如果上传了图片，用真实 CLIP 分析结果；否则用数据集 P50
    if image_analysis_result and image_analysis_result.get('overall_score') is not None:
        img_avg = image_analysis_result['overall_score']
        img_source = 'uploaded'  # 标记来源，给建议用
    else:
        img_avg = benchmarks['image_score'][0.5]
        img_source = 'estimated'

    # 属性等用 P50
    attr_s = benchmarks.get('attributes_score', {0.5: 50})[0.5] if 'attributes_score' in benchmarks else 50
    imp_s = 50

    # Listing 综合分
    weights = {"features": 0.30, "attributes": 0.25, "important": 0.05,
               "aplus": 0.10, "video": 0.05, "image": 0.25}
    base = (feat_avg * weights["features"] + attr_s * weights["attributes"] +
            imp_s * weights["important"] + aplus_score * weights["aplus"] +
            vid_s * weights["video"] + img_avg * weights["image"])
    listing_score = min(base + brand_bonus + trust_score_val * 0.1, 100)

    # 搜索分
    avg_thumb = benchmarks.get('thumbnail_score', {0.5: 50})[0.5] if 'thumbnail_score' in benchmarks else 50
    avg_position = 50
    search_score = (title_score_val * 0.25 + avg_thumb * 0.30 + price_score * 0.15 +
                    trust_score_val * 0.20 + avg_position * 0.10)

    conv_score = benchmarks['Conversion_Score'][0.5]
    detail_conversion = 0.6 * listing_score + 0.4 * conv_score
    total_score = 0.5 * search_score + 0.5 * detail_conversion

    # ===== 找出最相似竞品（标题+价格+图片视觉相似度） =====
    new_img_emb = (image_analysis_result or {}).get('main_image_embedding')
    competitors, all_ranked = find_top_competitors(
        full_df, new_title or "", new_price, top_n=top_n_competitors,
        new_image_embedding=new_img_emb,
        dataset_embeddings=dataset_image_embeddings
    )

    # ===== 构造对比表 =====
    dim_rows = [
        ('标题得分', title_score_val, 'title_score'),
        ('价格得分', price_score, 'price_score'),
        ('信任得分', trust_score_val, 'trust_score'),
        ('五点描述', round(feat_avg, 2), 'bullet_score'),
        ('图片得分', round(img_avg, 2), 'image_score'),
        ('视频得分', vid_s, 'video_score'),
        ('A+ 得分', aplus_score, 'aplus_score'),
        ('搜索得分', round(search_score, 2), 'search_score'),
        ('详情得分', round(listing_score, 2), 'listing_score'),
        ('转化得分', round(conv_score, 2), 'Conversion_Score'),
        ('综合总分', round(total_score, 2), 'Total_Score'),
    ]
    compare_df = pd.DataFrame([
        {
            '维度': name,
            '新品得分': val,
            '数据集 P25': benchmarks[col][0.25],
            '数据集 P50（中位）': benchmarks[col][0.5],
            '数据集 P75': benchmarks[col][0.75],
            'vs 中位': round(val - benchmarks[col][0.5], 2),
            '位置': _quantile_position(val, benchmarks[col]),
        }
        for name, val, col in dim_rows
    ])

    # ===== 智能建议生成 =====
    advice = _generate_smart_advice(
        new_title, new_features, features_list, feat_details, title_details,
        title_score_val, price_score, trust_score_val, feat_avg, vid_s, aplus_score,
        has_aplus, has_brandstory, video_count, new_stars, new_reviews,
        benchmarks, competitors,
        img_avg=img_avg, img_source=img_source,
        image_analysis_result=image_analysis_result
    )

    return {
        'compare_df': compare_df,
        'advice': advice,
        'competitors': competitors,
        'scores': {
            'title': title_score_val,
            'price': price_score,
            'trust': trust_score_val,
            'bullet': round(feat_avg, 2),
            'image': round(img_avg, 2),
            'video': vid_s,
            'aplus': aplus_score,
            'search': round(search_score, 2),
            'listing': round(listing_score, 2),
            'conversion': round(conv_score, 2),
            'total': round(total_score, 2),
        },
        'feat_details': feat_details,
        'title_details': title_details,
        'image_analysis': image_analysis_result,
        'img_source': img_source,
        'competitors': competitors,
        'all_ranked': all_ranked,  # 全部产品的相似度排名（用于显示 Top10）
        'has_image_sim': new_img_emb is not None and bool(dataset_image_embeddings),
    }


def _quantile_position(val, q_dict):
    """判断 val 落在哪个分位区间"""
    p25, p50, p75 = q_dict[0.25], q_dict[0.5], q_dict[0.75]
    if val >= p75:
        return "🟢 前 25%"
    elif val >= p50:
        return "🟡 中上"
    elif val >= p25:
        return "🟠 中下"
    else:
        return "🔴 后 25%"


def _generate_smart_advice(new_title, new_features, features_list, feat_details,
                           title_details, title_score_val, price_score, trust_score_val,
                           feat_avg, vid_s, aplus_score, has_aplus, has_brandstory,
                           video_count, new_stars, new_reviews, benchmarks, competitors,
                           img_avg=None, img_source='estimated', image_analysis_result=None):
    """根据实际差距动态生成建议，避免千篇一律
    Args:
        img_source: 'uploaded' 用真实CLIP分析；'estimated' 用数据集中位数
        image_analysis_result: 上传图片时的 CLIP 分析结果字典
    """
    advice = []

    # ===== 1. 标题建议 =====
    title_q = benchmarks['title_score']
    if not new_title:
        advice.append(("❌", "标题", "标题为空，请填写完整标题（建议 60-180 字符，含品牌+核心关键词+规格）"))
    elif title_score_val < title_q[0.25]:
        issues = []
        if len(new_title) < 30:
            issues.append(f"标题过短（{len(new_title)} 字符，建议 60-180）")
        if not title_details['has_num']:
            issues.append("缺少数字（如尺寸/容量/数量）")
        if not title_details['has_unit']:
            issues.append("缺少单位（cm/kg/W 等）")
        missing_info = [k for k, v in title_details['hit'].items() if v == 0]
        if missing_info:
            info_map = {'material': '材质', 'color': '颜色', 'size': '尺寸',
                        'function': '功能', 'scenario': '场景'}
            missing_cn = '、'.join(info_map.get(m, m) for m in missing_info)
            issues.append(f"缺少信息点：{missing_cn}")
        advice.append(("🔴", "标题", f"标题得分 {title_score_val}，处于后 25%。问题：{'；'.join(issues)}"))
    elif title_score_val < title_q[0.5]:
        advice.append(("🟠", "标题", f"标题得分 {title_score_val}，低于中位水平（{title_q[0.5]:.1f}），可继续优化关键词覆盖"))
    else:
        advice.append(("🟢", "标题", f"标题得分 {title_score_val}，已超过数据集中位水平，保持"))

    # ===== 2. 五点描述建议（逐条） =====
    bullet_q = benchmarks['bullet_score']
    if not features_list:
        advice.append(("❌", "五点描述", "未填写五点描述，请至少提供 3-5 条卖点"))
    elif feat_avg < bullet_q[0.25]:
        advice.append(("🔴", "五点描述",
                       f"五点描述均分 {feat_avg:.1f}，处于后 25%（中位 {bullet_q[0.5]:.1f}）。逐条诊断："))
        for i, d in enumerate(feat_details):
            sub_issues = []
            if d['score'] < 40:
                sub_issues.append("得分偏低")
            if not d['has_user_mention']:
                sub_issues.append("缺用户导向词（Sie/Ihnen）")
            if not d['has_differentiation']:
                sub_issues.append("缺差异化词（einzigartig/exklusiv）")
            if d['length'] < 50:
                sub_issues.append(f"过短（{d['length']} 字符）")
            missing_cov = [k for k, v in d['coverage'].items() if v == 0]
            if missing_cov:
                cov_map = {'size': '尺寸', 'material': '材质', 'warranty': '保修',
                           'usage': '用途', 'differentiation': '差异化', 'user_oriented': '用户导向'}
                missing_cn = '、'.join(cov_map.get(m, m) for m in missing_cov[:3])
                sub_issues.append(f"缺信息维度：{missing_cn}")
            if sub_issues:
                advice.append(("📌", f"  第{i+1}条", f"{d['text'][:60]}... → {'；'.join(sub_issues)}"))
    elif feat_avg < bullet_q[0.5]:
        weak = [d for d in feat_details if d['score'] < bullet_q[0.5]]
        if weak:
            advice.append(("🟠", "五点描述",
                          f"五点描述均分 {feat_avg:.1f}，低于中位。其中 {len(weak)} 条偏弱，建议加强差异化表达"))
    else:
        advice.append(("🟢", "五点描述", f"五点描述均分 {feat_avg:.1f}，超过中位水平"))

    # ===== 3. 价格建议（带具体竞品价格对比） =====
    price_q = benchmarks['price_score']
    if price_score < price_q[0.25]:
        # 价格得分低 = 价格偏高
        if not competitors.empty and 'price_value' in competitors.columns:
            comp_prices = competitors['price_value'].dropna().tolist()
            if comp_prices:
                comp_avg = sum(comp_prices) / len(comp_prices)
                advice.append(("🔴", "价格",
                              f"价格 {new_price}€ 处于后 25%（百分位 {price_score}），最相似 3 个竞品均价 {comp_avg:.2f}€，"
                              f"建议降价至 {comp_avg * 0.95:.2f}€ 以下或增加赠品/延保提升价值感"))
        else:
            advice.append(("🔴", "价格", f"价格竞争力处于后 25%（百分位 {price_score}），建议调价"))
    elif price_score > price_q[0.75]:
        advice.append(("🟢", "价格", f"价格优势明显（百分位 {price_score}，前 25%），可考虑提升利润空间或加强品质宣传"))
    else:
        advice.append(("🟡", "价格", f"价格处于中位水平（百分位 {price_score}），竞争力一般"))

    # ===== 4. 信任分（星级+评论数） =====
    trust_q = benchmarks['trust_score']
    if trust_score_val < trust_q[0.25]:
        advice.append(("🔴", "信任",
                      f"信任分 {trust_score_val}，处于后 25%。当前 {new_stars} 星 / {new_reviews} 评论，"
                      f"建议：1) 上线后立即用 Vine 计划收集 10+ 评论；2) 提供 30 天无理由退货；3) 突出售后保障"))
    elif trust_score_val < trust_q[0.5]:
        advice.append(("🟠", "信任",
                      f"信任分 {trust_score_val}，低于中位。建议加强售后政策展示，主动邀请留评"))
    else:
        advice.append(("🟢", "信任", f"信任分 {trust_score_val}，处于中上水平"))

    # ===== 5. 视频建议 =====
    video_q = benchmarks['video_score']
    if vid_s < video_q[0.25]:
        advice.append(("🔴", "视频",
                      "无产品视频，建议至少制作 1 个 30 秒开箱演示 + 1 个使用场景视频，可显著提升转化"))
    elif vid_s < video_q[0.5]:
        advice.append(("🟠", "视频", "视频数量偏少，建议补充使用场景视频"))
    else:
        advice.append(("🟢", "视频", "视频覆盖充分"))

    # ===== 6. A+ 内容建议 =====
    aplus_q = benchmarks['aplus_score']
    if not has_aplus:
        if aplus_q[0.5] > 30:
            advice.append(("🟠", "A+ 内容",
                          "未启用 A+ 内容。数据集中 {}/{} 产品有 A+，建议品牌备案后启用，可提升详情页转化 5-15%".format(
                              int((benchmarks.get('_aplus_count', 0))), 0)))
        advice.append(("🟠", "A+ 内容", "未启用 A+ 内容，建议品牌备案后开通，加入品牌故事、对比模块、Q&A 模块"))
    else:
        advice.append(("🟢", "A+ 内容", "已启用 A+ 内容，建议加入竞品对比表和场景应用模块"))

    # ===== 6.5 图片建议（基于真实 CLIP 分析或预估） =====
    img_q = benchmarks['image_score']
    if img_source == 'uploaded' and image_analysis_result:
        # 用 CLIP 真实结果生成具体建议
        overall = image_analysis_result.get('overall_score', 0)
        dim_avgs = image_analysis_result.get('dim_avgs', {})
        type_scores = image_analysis_result.get('type_scores', {})
        img_count = image_analysis_result.get('image_count', 0)

        if overall < img_q[0.25]:
            advice.append(("🔴", "图片",
                          f"图片综合得分 {overall:.1f}（{img_count} 张已分析），处于后 25%。具体诊断："))
        elif overall < img_q[0.5]:
            advice.append(("🟠", "图片",
                          f"图片综合得分 {overall:.1f}（{img_count} 张已分析），低于中位水平。具体诊断："))
        elif overall < img_q[0.75]:
            advice.append(("🟡", "图片",
                          f"图片综合得分 {overall:.1f}（{img_count} 张已分析），中上水平，仍有提升空间："))
        else:
            advice.append(("🟢", "图片",
                          f"图片综合得分 {overall:.1f}（{img_count} 张已分析），前 25%，保持"))

        # 维度细分：6 个核心心理维度
        dim_thresholds = [
            ('attention', '注意力', 55, '主图不够抓眼，建议：白底高清+产品占图 85%+45度角'),
            ('trust_signal', '信任感', 55, '信任感不足，建议：增加专业拍摄+品牌 LOGO 角标'),
            ('quality_perception', '品质感', 55, '品质感弱，建议：提升光线+特写材质纹理'),
            ('value_perception', '价值感', 50, '价值感不强，建议：增加功能标注图/赠品展示'),
            ('usage_imagination', '使用场景', 50, '缺场景图，建议：至少 1 张生活场景图，让用户想象使用'),
            ('risk_reduction', '风险消除', 50, '风险信号弱，建议：增加尺寸标注图/功能拆解图'),
            ('differentiation', '差异化', 50, '差异化不足，建议：增加竞品对比图或独家设计展示'),
        ]
        weak_dims = []
        for dim_key, dim_name, threshold, suggestion in dim_thresholds:
            val = dim_avgs.get(dim_key, 0) * 100
            if val < threshold:
                weak_dims.append((dim_name, val, suggestion))
        if weak_dims:
            for dim_name, val, suggestion in weak_dims[:4]:  # 最多列 4 个
                advice.append(("📌", f"  图片·{dim_name}", f"{val:.1f}/100。{suggestion}"))

        # 图片类型结构建议
        main_info = type_scores.get('main', {})
        detail_info = type_scores.get('detail', {})
        lifestyle_info = type_scores.get('lifestyle', {})
        aplus_info = type_scores.get('aplus', {})

        struct_issues = []
        if main_info.get('count', 0) == 0:
            struct_issues.append("缺少主图（白底图），Amazon 必备")
        if detail_info.get('count', 0) == 0:
            struct_issues.append("缺少细节图，建议 2-3 张不同角度/材质特写")
        if lifestyle_info.get('count', 0) == 0:
            struct_issues.append("缺少场景图，建议 1-2 张使用场景照")
        if aplus_info.get('count', 0) == 0 and has_aplus:
            struct_issues.append("已勾选 A+ 但未上传 A+ 图，建议补充")
        if struct_issues:
            advice.append(("📌", "  图片·结构", "；".join(struct_issues)))

        # 主图特别诊断（最重要的图）
        if main_info.get('count', 0) > 0:
            main_score = main_info['score']
            if main_score < 60:
                advice.append(("📌", "  图片·主图",
                              f"主图得分 {main_score:.1f}（这是搜索结果页第一印象，最关键）。"
                              "建议：纯白背景（RGB 255,255,255）、产品占比 85%、无水印无文字"))
    else:
        # 未上传图片：用数据集中位数，给结构性建议
        advice.append(("ℹ️", "图片",
                      f"未上传图片，图片维度用数据集中位数 {img_q[0.5]:.1f} 作为预估。"
                      "建议上传主图（白底）+ 2-3 张细节图 + 1 张场景图，可获得基于 CLIP 的真实图片诊断"))

    # ===== 7. 竞品对比建议 =====
    if not competitors.empty:
        comp_advice_lines = []
        for _, c in competitors.iterrows():
            title_short = (c.get('title') or '')[:50]
            comp_advice_lines.append(
                f"  • ASIN {c['asin']} | {title_short} | 价格 {c.get('price_value', 'N/A')}€ | "
                f"总分 {c.get('Total_Score', 0):.1f}"
            )
        # 找出新品可以学习的具体维度
        best_comp = competitors.iloc[0]
        new_total = (title_score_val + feat_avg + trust_score_val) / 3
        comp_total = (best_comp.get('title_score', 0) + best_comp.get('bullet_score', 0) +
                      best_comp.get('trust_score', 0)) / 3
        if comp_total > new_total:
            gap_dims = []
            if best_comp.get('bullet_score', 0) > feat_avg + 10:
                gap_dims.append("五点描述更具体")
            if best_comp.get('title_score', 0) > title_score_val + 10:
                gap_dims.append("标题关键词更全")
            if best_comp.get('image_score', 0) > 50:
                gap_dims.append("图片更丰富")
            if best_comp.get('video_score', 0) > vid_s:
                gap_dims.append("有视频")
            if gap_dims:
                advice.append(("🎯", "竞品学习",
                              f"最相似竞品 {best_comp['asin']} 总分 {best_comp.get('Total_Score', 0):.1f}，"
                              f"可借鉴：{'、'.join(gap_dims)}"))
        advice.append(("📋", "Top3 竞品", "\n".join(comp_advice_lines)))

    return advice


# ==================== Streamlit UI ====================
st.title("📊 Amazon 产品竞争力分析 v2")
st.caption("🚀 优化版：100 产品 < 30 秒 | 默认无模型加载 | 按需深度分析")

# 侧边栏：分析选项
with st.sidebar:
    st.header("⚙️ 分析选项")
    max_items = st.number_input(
        "🔢 分析前多少个产品？（0=全部）",
        min_value=0, max_value=500, value=10, step=1
    )
    use_cache = st.checkbox("💾 启用数据集缓存（推荐）", value=True)
    st.markdown("---")
    st.markdown("**深度分析（按需）**")
    enable_deep = st.checkbox("启用单 ASIN CLIP 图片深度分析", value=False,
                              help="开启后，选择 ASIN 时会加载 CLIP 模型分析图片（每个 ASIN 约 5-10 秒）")
    st.markdown("---")
    st.markdown("**关于性能**")
    st.markdown("""
    - 快速层：纯关键词+数字，无 NLP 模型
    - 100 产品约 20-30 秒
    - 内存占用 < 200MB
    - 适配 Streamlit Cloud 免费额度
    """)

uploaded_file = st.file_uploader(
    "上传原始爬虫 JSON 文件 (dataset_free-amazon-product-scraper_*.json)",
    type=["json"]
)

if uploaded_file is not None:
    raw_data = json.load(uploaded_file)
    st.success(f"✅ 文件上传成功，共 {len(raw_data)} 条原始数据")

    # 数据分类
    if isinstance(raw_data, list) and not ("搜索页信息" in raw_data[0] if raw_data else False):
        with st.spinner("正在执行数据分类..."):
            classified_data = classify_raw_data(raw_data)
        st.success("✅ 分类完成")
    else:
        classified_data = raw_data

    # 缓存检查
    cache_key = f"{dataset_hash(classified_data)}_{max_items}"
    cache_file = os.path.join(CONFIG["analysis_cache_dir"], f"fast_{cache_key}.parquet")

    final_df = None
    full_df = None

    if use_cache and os.path.exists(cache_file):
        st.info("💾 命中数据集缓存，秒级加载...")
        cached = pd.read_parquet(cache_file)
        final_df = cached[cached['asin'].notna()][['asin', 'title', 'brand', 'search_score',
                                                     'Detail_Conversion', 'Total_Score',
                                                     'listing_score', 'Conversion_Score',
                                                     'search_rank', 'price_value', 'stars', 'reviews'
                                                     ]].copy() if not cached.empty else cached
        full_df = cached
    else:
        progress_bar = st.progress(0, text="开始快速分析...")
        status = st.empty()

        def progress_callback(p, msg):
            progress_bar.progress(p, text=msg)
            status.text(f"{msg} ({p}%)")

        with st.spinner("运行快速分析流水线..."):
            final_df, full_df = run_fast_analysis(classified_data, max_items, progress_callback)
        progress_bar.empty()
        status.empty()

        # 保存缓存
        if use_cache and full_df is not None and not full_df.empty:
            os.makedirs(CONFIG["analysis_cache_dir"], exist_ok=True)
            try:
                # _features 等内部字段不缓存
                cols_to_save = [c for c in full_df.columns if not c.startswith('_')]
                full_df[cols_to_save].to_parquet(cache_file, index=False)
                st.toast("💾 已缓存分析结果", icon="✅")
            except Exception as e:
                print(f"⚠️ 缓存保存失败: {e}")

    if final_df is None or final_df.empty:
        st.error("❌ 分析失败或无数据")
        st.stop()

    # ===== 排名表 =====
    st.subheader("📋 产品竞争力排名")
    final_df_display = final_df.copy()
    final_df_display['商品链接'] = final_df_display['asin'].apply(lambda x: f'https://www.amazon.de/dp/{x}')
    st.dataframe(
        final_df_display,
        column_config={
            "商品链接": st.column_config.LinkColumn("详情页", display_text=r'^(https://www\.amazon\.de/dp/)(.*)'),
            "asin": None,
        },
        width='stretch',
        hide_index=True
    )

    csv = final_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 下载结果 CSV", data=csv,
                       file_name="product_competitiveness_final.csv", mime="text/csv")

    # ===== 可视化 =====
    st.subheader("📈 数据可视化")
    tab1, tab2, tab3 = st.tabs(["Top10 综合分", "维度趋势", "分布"])

    with tab1:
        top10 = final_df.head(10).set_index("asin")["Total_Score"]
        st.bar_chart(top10)
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

    # ===== 单品深度分析 =====
    st.subheader("🔍 单品分析")
    asin_list = final_df["asin"].tolist()
    selected_asin = st.selectbox("选择 ASIN", asin_list)

    if selected_asin:
        row = full_df[full_df["asin"] == selected_asin].iloc[0]
        numeric_cols = ["search_score", "Detail_Conversion", "Total_Score",
                        "listing_score", "Conversion_Score"]
        available_cols = [c for c in numeric_cols if c in full_df.columns]
        means = full_df[available_cols].mean()
        medians = full_df[available_cols].median()

        col_radar, col_table = st.columns([1, 1])
        with col_radar:
            st.markdown("**📊 雷达图（vs 均值/中位）**")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[row[c] for c in available_cols],
                theta=available_cols, fill='toself', name=selected_asin
            ))
            fig.add_trace(go.Scatterpolar(
                r=[means[c] for c in available_cols],
                theta=available_cols, fill='toself', name='均值'
            ))
            fig.add_trace(go.Scatterpolar(
                r=[medians[c] for c in available_cols],
                theta=available_cols, fill='toself', name='中位数', visible='legendonly'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**📈 各维度详情**")
            detail_df = pd.DataFrame({
                '维度': available_cols,
                '本品': [row[c] for c in available_cols],
                '均值': [means[c] for c in available_cols],
                '中位': [medians[c] for c in available_cols],
                '差值(vs 中位)': [round(row[c] - medians[c], 2) for c in available_cols],
            })
            st.dataframe(detail_df, hide_index=True, use_container_width=True)

        # 亮点 & 痛点
        st.markdown("**✨ 亮点与痛点**")
        diff = {dim: row[dim] - medians[dim] for dim in available_cols}
        highlights = [dim for dim, d in diff.items() if d > 5]
        painpoints = [dim for dim, d in diff.items() if d < -5]
        col_h, col_p = st.columns(2)
        with col_h:
            if highlights:
                st.success(f"💡 亮点：{', '.join(highlights)}（高于中位 5 分以上）")
            else:
                st.info("无明显亮点")
        with col_p:
            if painpoints:
                st.error(f"⚠️ 痛点：{', '.join(painpoints)}（低于中位 5 分以上）")
            else:
                st.info("无明显痛点")

        # 深度分析（CLIP 图片）
        if enable_deep:
            st.markdown("---")
            st.subheader("🖼️ 图片深度分析（CLIP）")
            if st.button(f"对 {selected_asin} 运行 CLIP 图片分析", type="primary"):
                with st.spinner("加载 CLIP 模型并分析图片（首次约 30 秒）..."):
                    try:
                        # 从原始分类数据中找到该 ASIN
                        listing_item = None
                        search_item = None
                        for i, l in enumerate(classified_data["listing信息"]):
                            if l.get("originalAsin") == selected_asin:
                                listing_item = l
                                if i < len(classified_data["搜索页信息"]):
                                    search_item = classified_data["搜索页信息"][i]
                                break
                        if listing_item and search_item:
                            img_summary = deep_analyze_asin_images(
                                listing_item, search_item,
                                max_images=CONFIG["max_images_per_product"]
                            )
                            if img_summary:
                                st.success("✅ 图片分析完成")
                                img_cols = st.columns(3)
                                metrics = [
                                    ("主图得分", img_summary.get('thumbnail_score', 0)),
                                    ("详情图均分", img_summary.get('detail_score', 0)),
                                    ("A+ 图得分", img_summary.get('aplus_score', 0)),
                                    ("信任信号", img_summary.get('detail_trust', 0)),
                                    ("价值感知", img_summary.get('detail_value', 0)),
                                    ("使用想象", img_summary.get('detail_usage', 0)),
                                ]
                                for i, (name, val) in enumerate(metrics):
                                    with img_cols[i % 3]:
                                        st.metric(name, f"{val:.1f}")
                            else:
                                st.warning("未提取到图片")
                        else:
                            st.error(f"未找到 ASIN {selected_asin} 的数据")
                    except Exception as e:
                        st.error(f"深度分析失败: {e}")

    # ===== 智能新品对比 =====
    st.markdown("---")
    st.subheader("📝 新品竞争力预测与智能优化建议")
    st.caption("基于数据集分位数 + 真实竞品对比，建议会根据你的输入动态变化")

    with st.form("new_product_form"):
        st.markdown("**填写你的产品信息**")
        col1, col2 = st.columns(2)
        with col1:
            new_title = st.text_input("产品标题", placeholder="例如: Premium Esszimmerstühle 4er Set, Samtbezogen, 150kg belastbar")
            new_price = st.number_input("价格 (€)", min_value=0.0, step=0.01, value=29.99)
            new_stars = st.number_input("星级评分 (1-5)", min_value=0.0, max_value=5.0, step=0.1, value=4.3)
            new_reviews = st.number_input("评论数量", min_value=0, step=1, value=120)
        with col2:
            new_features = st.text_area("五点描述（每行一条）", placeholder="每条描述占一行", height=150)
            has_aplus = st.checkbox("是否有 A+ 内容")
            has_brandstory = st.checkbox("是否有品牌故事")
            video_count = st.selectbox("视频数量", [0, 1, 2, 3, 5], index=0)
            top_n_competitors = st.slider("🎯 找多少个相似竞品", min_value=3, max_value=10, value=5, step=1,
                                          help="基于图片视觉相似度+标题+价格综合匹配，3-10 个可调")

        # ===== 新增：图片上传 =====
        st.markdown("**🖼️ 上传产品图片（可选，上传后启用 CLIP 真实图片诊断）**")
        st.caption("建议上传：1 张主图（白底）+ 2-3 张细节图 + 1 张场景图，最多 6 张。"
                   "首次分析需加载 CLIP 模型（约 30 秒），之后每张图 1-2 秒。")

        uploaded_images = st.file_uploader(
            "选择图片（支持 jpg/png/webp）",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            key="new_product_images"
        )
        if uploaded_images and len(uploaded_images) > 6:
            st.warning("⚠️ 最多分析 6 张图片，已自动截取前 6 张")
            uploaded_images = uploaded_images[:6]

        # 为每张上传的图片选择类型
        image_types_list = []
        if uploaded_images:
            st.markdown("**为每张图片指定角色**（影响 CLIP 评分权重）")
            type_cols = st.columns(min(len(uploaded_images), 3))
            for i, img_file in enumerate(uploaded_images):
                with type_cols[i % 3]:
                    default_type = 'main' if i == 0 else ('detail' if i <= 3 else 'lifestyle')
                    img_type = st.selectbox(
                        f"图 {i+1}: {img_file.name[:25]}",
                        options=['main', 'detail', 'lifestyle', 'aplus'],
                        format_func=lambda x: {
                            'main': '主图（白底）',
                            'detail': '细节图（多角度/材质）',
                            'lifestyle': '场景图（使用场景）',
                            'aplus': 'A+ 图（品牌/对比）'
                        }[x],
                        index=['main', 'detail', 'lifestyle', 'aplus'].index(default_type),
                        key=f"img_type_{i}"
                    )
                    image_types_list.append(img_type)

            # 缩略图预览
            st.markdown("**图片预览**")
            preview_cols = st.columns(min(len(uploaded_images), 6))
            for i, img_file in enumerate(uploaded_images):
                with preview_cols[i]:
                    try:
                        img = Image.open(img_file)
                        st.image(img, caption=f"{image_types_list[i]}", width=100)
                    except Exception:
                        st.warning(f"图 {i+1} 预览失败")

        submitted = st.form_submit_button("📊 分析新品竞争力", type="primary")

    if submitted:
        if final_df.empty:
            st.warning("当前没有可对比的数据集，请先上传 JSON 文件。")
        else:
            # ===== 第一步：如果有上传图片，先跑 CLIP 分析 =====
            image_analysis_result = None
            dataset_image_embeddings = None
            thumbnail_urls_map = None

            if uploaded_images:
                with st.spinner("🖼️ 加载 CLIP 模型并分析上传图片（首次约 30 秒，之后每张 1-2 秒）..."):
                    try:
                        # 把 UploadedFile 转成 bytes（用于 @st.cache_data 哈希）
                        img_bytes_list = [f.getvalue() for f in uploaded_images]
                        image_analysis_result = analyze_uploaded_images_with_clip(
                            tuple(img_bytes_list),  # tuple 可哈希
                            tuple(image_types_list)
                        )
                        if image_analysis_result:
                            st.success(f"✅ 图片分析完成，综合得分 {image_analysis_result['overall_score']:.1f}")
                    except Exception as e:
                        st.warning(f"图片分析失败（图片维度将用数据集中位数估算）: {e}")
                        image_analysis_result = None

                # ===== 第一步半：如果新品图片分析成功且有主图，计算数据集 embedding 用于找视觉相似竞品 =====
                if (image_analysis_result and
                        image_analysis_result.get('main_image_embedding') is not None):
                    # 提取数据集所有 thumbnail URL
                    thumb_urls_list = []
                    thumbnail_urls_map = {}
                    for i, item in enumerate(classified_data["搜索页信息"]):
                        asin = item.get("asin")
                        url = item.get("thumbnailImage")
                        if asin and url:
                            thumb_urls_list.append((asin, url))
                            thumbnail_urls_map[asin] = url

                    if thumb_urls_list:
                        with st.spinner(f"🔍 计算数据集 {len(thumb_urls_list)} 张主图的视觉向量（首次约 1-3 分钟，已计算则秒级）..."):
                            try:
                                dataset_image_embeddings = compute_dataset_image_embeddings(
                                    tuple(thumb_urls_list)
                                )
                                st.success(f"✅ 数据集图片向量计算完成，{len(dataset_image_embeddings)} 张成功")
                            except Exception as e:
                                st.warning(f"数据集图片向量计算失败（将退化为标题+价格匹配）: {e}")
                                dataset_image_embeddings = None

            # ===== 第二步：智能新品分析 =====
            with st.spinner("智能分析中..."):
                result = analyze_new_product_smart(
                    new_title, new_price, new_stars, new_reviews,
                    new_features, has_aplus, has_brandstory,
                    video_count, full_df,
                    image_analysis_result=image_analysis_result,
                    dataset_image_embeddings=dataset_image_embeddings,
                    thumbnail_urls_map=thumbnail_urls_map,
                    top_n_competitors=top_n_competitors
                )

            # 对比表
            st.subheader("📊 新品 vs 数据集 分位数对比")
            st.dataframe(result['compare_df'], hide_index=True, use_container_width=True)
            st.caption("🟢 前25% / 🟡 中上 / 🟠 中下 / 🔴 后25%")

            # 核心得分
            st.subheader("🎯 新品核心得分")
            score_cols = st.columns(4)
            scores = result['scores']
            for i, (name, val) in enumerate([
                ('综合总分', scores['total']),
                ('搜索得分', scores['search']),
                ('详情得分', scores['listing']),
                ('转化得分', scores['conversion']),
            ]):
                with score_cols[i]:
                    st.metric(name, f"{val:.1f}")

            # 智能建议
            st.subheader("💡 智能优化建议")
            if not result['advice']:
                st.success("🎉 各项指标均优于数据集中位水平，竞争力较强！")
            else:
                for icon, dim, text in result['advice']:
                    color = {"🟢": "green", "🟡": "#888", "🟠": "#FF8C00",
                             "🔴": "red", "❌": "red", "🎯": "#0066CC",
                             "📌": "#663399", "📋": "#444", "ℹ️": "#666"}.get(icon, "#333")
                    st.markdown(
                        f"<div style='padding:8px 12px;margin:4px 0;border-left:3px solid {color};"
                        f"background:#f9f9f9;'><strong>{icon} {dim}</strong>：{text}</div>",
                        unsafe_allow_html=True
                    )

            # ===== 图片深度分析明细 =====
            if result.get('img_source') == 'uploaded' and result.get('image_analysis'):
                st.subheader("🖼️ 图片深度分析明细")
                img_res = result['image_analysis']

                # 顶部 4 个核心指标
                img_metric_cols = st.columns(4)
                metrics = [
                    ("综合得分", img_res['overall_score']),
                    ("图片数量", img_res['image_count']),
                    ("主图得分", img_res['type_scores'].get('main', {}).get('score', 0)),
                    ("细节图均分", img_res['type_scores'].get('detail', {}).get('score', 0)),
                ]
                for i, (name, val) in enumerate(metrics):
                    with img_metric_cols[i]:
                        if name == "图片数量":
                            st.metric(name, f"{int(val)}")
                        else:
                            st.metric(name, f"{val:.1f}")

                # 各类型图片得分
                st.markdown("**各类型图片得分**")
                type_data = []
                type_name_map = {'main': '主图', 'detail': '细节图', 'lifestyle': '场景图', 'aplus': 'A+图'}
                for t, info in img_res['type_scores'].items():
                    if info['count'] > 0:
                        type_data.append({
                            '类型': type_name_map.get(t, t),
                            '数量': info['count'],
                            '平均得分': round(info['score'], 2),
                        })
                if type_data:
                    st.dataframe(pd.DataFrame(type_data), hide_index=True, use_container_width=True)

                # 9 个心理维度雷达图
                st.markdown("**9 维度 CLIP 心理感知得分**")
                dim_zh_map = {
                    'attention': '注意力',
                    'product_understanding': '产品理解',
                    'value_perception': '价值感知',
                    'usage_imagination': '使用想象',
                    'trust_signal': '信任信号',
                    'risk_reduction': '风险消除',
                    'quality_perception': '品质感知',
                    'differentiation': '差异化',
                    'purchase_intent': '购买意愿',
                }
                dim_names = [dim_zh_map[d] for d in FEATURE_NAMES]
                dim_vals = [img_res['dim_avgs'].get(d, 0) * 100 for d in FEATURE_NAMES]
                # 添加数据集基准（如果有）
                # 这里用 50 作为基线
                dim_baseline = [50] * len(FEATURE_NAMES)

                fig_img = go.Figure()
                fig_img.add_trace(go.Scatterpolar(
                    r=dim_vals, theta=dim_names, fill='toself', name='新品图片'
                ))
                fig_img.add_trace(go.Scatterpolar(
                    r=dim_baseline, theta=dim_names, fill='toself', name='基线 (50)',
                    line=dict(dash='dot', color='gray')
                ))
                fig_img.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True, height=450
                )
                st.plotly_chart(fig_img, use_container_width=True)

                # 每张图独立得分
                if img_res.get('per_image'):
                    st.markdown("**每张图片独立得分**")
                    per_img_data = []
                    type_name_map_full = {'main': '主图', 'detail': '细节图',
                                          'lifestyle': '场景图', 'aplus': 'A+图'}
                    for i, r in enumerate(img_res['per_image']):
                        row = {
                            '图片序号': i + 1,
                            '类型': type_name_map_full.get(r['image_type'], r['image_type']),
                            '消费者得分': round(r['consumer_score'], 2),
                            '注意力': round(r['features'].get('attention', 0) * 100, 1),
                            '信任': round(r['features'].get('trust_signal', 0) * 100, 1),
                            '品质': round(r['features'].get('quality_perception', 0) * 100, 1),
                            '价值': round(r['features'].get('value_perception', 0) * 100, 1),
                            '场景': round(r['features'].get('usage_imagination', 0) * 100, 1),
                        }
                        per_img_data.append(row)
                    st.dataframe(pd.DataFrame(per_img_data), hide_index=True, use_container_width=True)

            # 雷达图
            st.subheader("📊 核心维度雷达图")
            core_dims = ['标题', '价格', '信任', '五点描述', '图片', '视频', 'A+']
            core_new = [scores['title'], scores['price'], scores['trust'],
                        scores['bullet'], scores['image'], scores['video'], scores['aplus']]
            core_median = [
                result['compare_df'].iloc[0]['数据集 P50（中位）'],  # 标题
                result['compare_df'].iloc[1]['数据集 P50（中位）'],  # 价格
                result['compare_df'].iloc[2]['数据集 P50（中位）'],  # 信任
                result['compare_df'].iloc[3]['数据集 P50（中位）'],  # 五点
                result['compare_df'].iloc[4]['数据集 P50（中位）'],  # 图片
                result['compare_df'].iloc[5]['数据集 P50（中位）'],  # 视频
                result['compare_df'].iloc[6]['数据集 P50（中位）'],  # A+
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=core_new, theta=core_dims, fill='toself', name='新品'))
            fig.add_trace(go.Scatterpolar(r=core_median, theta=core_dims, fill='toself', name='数据集中位'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              showlegend=True, height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Top N 竞品详情（含图片对比）
            if not result['competitors'].empty:
                n_comp = len(result['competitors'])
                st.subheader(f"🎯 最相似的 {n_comp} 个竞品")

                # 诊断信息：让用户知道为什么有/没有图片相似度
                img_res = result.get('image_analysis') or {}
                main_emb = img_res.get('main_image_embedding')
                main_idx = img_res.get('main_image_index')

                if uploaded_images:
                    if main_emb is not None:
                        st.success(f"✅ 已基于新品主图（第 {main_idx + 1} 张）的 CLIP 视觉向量匹配竞品")
                    else:
                        st.warning(
                            "⚠️ 新品图片分析完成但主图向量提取失败，竞品匹配仅基于标题+价格。"
                            "可能原因：图片格式不支持或 CLIP 模型加载异常，请查看后台日志。"
                        )
                else:
                    st.info("ℹ️ 未上传新品图片，竞品匹配仅基于标题+价格。上传主图后可获得视觉相似度匹配。")

                # ===== 视觉对比区（仅当有图片相似度时显示）=====
                if result.get('has_image_sim') and thumbnail_urls_map:
                    st.markdown(f"**📸 视觉对比（新品 vs Top {min(n_comp, 4)} 竞品）**")
                    st.caption("基于 CLIP 视觉向量 + 标题 + 价格综合匹配，图片权重 45% / 标题 35% / 价格 20%")

                    # 新品主图 + 前 3 个竞品主图并排（最多 4 列）
                    n_show = min(n_comp, 3)
                    img_compare_cols = st.columns(n_show + 1)
                    # 第 1 列：新品主图
                    with img_compare_cols[0]:
                        st.markdown("**🆕 新品主图**")
                        # 用 main_image_index 找到主图（而不是只找 'main' 标记的）
                        if main_idx is not None and uploaded_images and main_idx < len(uploaded_images):
                            try:
                                st.image(Image.open(uploaded_images[main_idx]), use_container_width=True)
                            except Exception:
                                st.warning("主图预览失败")
                        else:
                            # 退化：找标记为 main 的
                            found = False
                            for i, img_file in enumerate(uploaded_images or []):
                                if image_types_list[i] == 'main':
                                    try:
                                        st.image(Image.open(img_file), use_container_width=True)
                                        found = True
                                    except Exception:
                                        st.warning("主图预览失败")
                                    break
                            if not found:
                                st.info("未找到主图")

                    # 后续列：竞品主图
                    for idx, (_, comp) in enumerate(result['competitors'].head(n_show).iterrows()):
                        with img_compare_cols[idx + 1]:
                            asin = comp['asin']
                            sim_score = comp.get('sim', 0)
                            img_sim = comp.get('image_sim')
                            sim_label = f"相似度 {sim_score*100:.0f}%"
                            if img_sim is not None and not pd.isna(img_sim):
                                sim_label += f"\n\n图片 {img_sim*100:.0f}%"
                            st.markdown(f"**#{idx+1} {asin}**\n\n{sim_label}")
                            url = thumbnail_urls_map.get(asin)
                            if url:
                                try:
                                    st.image(url, use_container_width=True)
                                except Exception:
                                    st.warning("竞品图加载失败")
                            else:
                                st.info("无主图")

                    st.markdown("")  # 空行

                # ===== 详细对比表（所有 N 个竞品）=====
                st.markdown(f"**📊 {n_comp} 个竞品详细对比**")
                if result.get('has_image_sim'):
                    comp_display = result['competitors'][['asin', 'title', 'brand', 'price_value',
                                                          'Total_Score', 'sim', 'image_sim',
                                                          'title_sim', 'price_sim']].copy()
                    comp_display['排名'] = range(1, len(comp_display) + 1)
                    comp_display['综合相似度'] = comp_display['sim'].apply(lambda x: f"{x*100:.1f}%")
                    comp_display['图片相似'] = comp_display['image_sim'].apply(
                        lambda x: f"{x*100:.1f}%" if x is not None and not pd.isna(x) else "—")
                    comp_display['标题相似'] = comp_display['title_sim'].apply(lambda x: f"{x*100:.1f}%")
                    comp_display['价格相似'] = comp_display['price_sim'].apply(lambda x: f"{x*100:.1f}%")
                    comp_display = comp_display.drop(columns=['sim', 'image_sim', 'title_sim', 'price_sim'])
                    comp_display = comp_display.rename(columns={
                        'title': '标题', 'brand': '品牌', 'price_value': '价格(€)',
                        'Total_Score': '综合分'
                    })
                    comp_display = comp_display[['排名', 'asin', '标题', '品牌', '价格(€)',
                                                  '综合分', '综合相似度', '图片相似', '标题相似', '价格相似']]
                else:
                    comp_display = result['competitors'][['asin', 'title', 'brand', 'price_value',
                                                          'Total_Score', 'sim', 'title_sim', 'price_sim']].copy()
                    comp_display['排名'] = range(1, len(comp_display) + 1)
                    comp_display['综合相似度'] = comp_display['sim'].apply(lambda x: f"{x*100:.1f}%")
                    comp_display['标题相似'] = comp_display['title_sim'].apply(lambda x: f"{x*100:.1f}%")
                    comp_display['价格相似'] = comp_display['price_sim'].apply(lambda x: f"{x*100:.1f}%")
                    comp_display = comp_display.drop(columns=['sim', 'title_sim', 'price_sim'])
                    comp_display = comp_display.rename(columns={
                        'title': '标题', 'brand': '品牌', 'price_value': '价格(€)',
                        'Total_Score': '综合分'
                    })
                    comp_display = comp_display[['排名', 'asin', '标题', '品牌', '价格(€)',
                                                  '综合分', '综合相似度', '标题相似', '价格相似']]

                comp_display['详情页'] = comp_display['asin'].apply(lambda x: f'https://www.amazon.de/dp/{x}')
                st.dataframe(comp_display, hide_index=True, use_container_width=True,
                             column_config={
                                 "详情页": st.column_config.LinkColumn("Amazon", display_text=r'^(https://www\.amazon\.de/dp/)(.*)')
                             })

                # ===== Top 10 完整排名（可展开）=====
                if result.get('all_ranked') is not None and not result['all_ranked'].empty:
                    total_n = len(result['all_ranked'])
                    with st.expander(f"📊 查看数据集全部 {total_n} 个产品的相似度排名（点击展开）"):
                        all_disp = result['all_ranked'][['asin', 'title', 'price_value',
                                                          'Total_Score', 'sim', 'image_sim']].copy()
                        all_disp['排名'] = range(1, len(all_disp) + 1)
                        all_disp['综合相似度'] = all_disp['sim'].apply(lambda x: f"{x*100:.1f}%")
                        all_disp['图片相似度'] = all_disp['image_sim'].apply(
                            lambda x: f"{x*100:.1f}%" if x is not None and not pd.isna(x) else "—")
                        all_disp = all_disp[['排名', 'asin', 'title', 'price_value', 'Total_Score',
                                              '综合相似度', '图片相似度']]
                        all_disp = all_disp.rename(columns={
                            'title': '标题', 'price_value': '价格(€)', 'Total_Score': '综合分'
                        })
                        st.dataframe(all_disp, hide_index=True, use_container_width=True)

else:
    st.info("👈 请上传原始爬虫 JSON 文件开始分析")
    st.markdown("""
    ### 📖 使用说明

    **快速分析（默认）**
    - 上传 JSON 文件后约 20-30 秒完成 100 个产品分析
    - 不加载任何 NLP 模型，内存占用 < 200MB
    - 结果自动缓存，相同数据集二次分析秒级返回

    **单品深度分析（可选）**
    - 在侧边栏勾选"启用单 ASIN CLIP 图片深度分析"
    - 选择某个 ASIN 后点击按钮触发
    - 首次加载 CLIP 模型约 30 秒，之后每个 ASIN 约 5-10 秒

    **新品智能对比**
    - 输入新品信息后点击"分析新品竞争力"
    - 系统会基于数据集分位数（P25/P50/P75）给出位置判断
    - 自动找出最相似的 3 个竞品作为参考
    - 建议根据你的实际输入动态生成，不再千篇一律
    """)
