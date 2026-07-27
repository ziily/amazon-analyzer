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
        print("🔧 加载零样本分类模型 (distilbert) ...")
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
            return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=-1, top_k=None)
        except:
            print("❌ 情感模型加载失败")
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
        score = (row["attention"]*0.35 + row["product_understanding"]*0.30 +
                 row["quality_perception"]*0.20 + row["differentiation"]*0.15)
    elif itype == "high_resolution":
        score = (row["value_perception"]*0.30 + row["usage_imagination"]*0.25 +
                 row["risk_reduction"]*0.25 + row["trust_signal"]*0.20)
    elif itype == "a_plus":
        score = (row["trust_signal"]*0.35 + row["quality_perception"]*0.25 +
                 row["differentiation"]*0.20 + row["value_perception"]*0.20)
    else:
        score = 0
    return round(score * 100, 2)

def aggregate_images(image_records):
    import hashlib
    print(f"🖼️ 开始图片分析，共 {len(image_records)} 张")

    # ===== 缓存检查 =====
    url_str = "".join(sorted([r.get("image_url", "") for r in image_records]))
    cache_hash = hashlib.md5(url_str.encode()).hexdigest()[:12]
    cache_dir = "../data/cache"
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
            if (i+1) % 10 == 0 or i == total-1:
                print(f"  图片进度 {i+1}/{total}, 成功 {success}, 失败 {fail}")
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
            row["thumbnail_score"] = row["thumbnail_attention"] = row["thumbnail_quality"] = row["thumbnail_purchase"] = 0
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
            row["detail_visual_score"] = row["detail_best_score"] = row["detail_trust"] = row["detail_value"] = row["detail_usage"] = row["detail_risk"] = 0
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
            row["aplus_mean_score"] = row["aplus_trust"] = row["aplus_quality"] = row["aplus_value"] = row["aplus_brand"] = 0
        summary.append(row)

    result_df = pd.DataFrame(summary)

    # 保存缓存
    os.makedirs(cache_dir, exist_ok=True)
    result_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    print(f"💾 图片缓存已保存: {cache_file}")

    return result_df

# ==================== 心理评分 ====================
PSYCH_LABELS = ["quality","convenience","cost_saving","safety","social_status","health","durability","aesthetics","innovation","trust"]

def batch_psych_scores(texts, classifier, batch_size=16):
    cache = {}
    unique = list(set([t for t in texts if t and len(t.strip())>=3]))
    if not unique:
        return {}
    if classifier is None:
        for t in unique:
            cache[t] = 0.0
        return cache

    total_batches = (len(unique) + batch_size - 1) // batch_size
    print(f"🧠 开始心理评分，共 {len(unique)} 条唯一文本，分 {total_batches} 个批次")

    for i in range(0, len(unique), batch_size):
        batch = unique[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"  批次 {batch_num}/{total_batches}，处理 {len(batch)} 条文本")
        try:
            results = classifier(batch, PSYCH_LABELS)
            if isinstance(results, dict):
                results = [results]
            for text, res in zip(batch, results):
                scores = res['scores']
                max_score = max(scores)
                diversity = min(len([s for s in scores if s>0.3])/3, 1.0)
                final = max_score*0.7 + diversity*0.3
                cache[text] = round(final*100, 2)
        except Exception as e:
            print(f"   ⚠️ 批次 {batch_num} 推理失败: {e}")
            for t in batch:
                cache[t] = 0.0
    print(f"✅ 心理评分完成")
    return cache

# ==================== Listing 各维度评分 ====================
def score_features_batch(features, text2score):
    if not features:
        return 0.0
    count = len(features)
    count_score = min(count/5, 1.0)*100
    avg_len = sum(len(f) for f in features)/count
    len_score = min(avg_len/80, 1.0)*100
    psycho_scores = [text2score.get(f, 0.0) for f in features]
    avg_psycho = sum(psycho_scores)/len(psycho_scores)
    max_psycho = max(psycho_scores)
    psycho_score = avg_psycho*0.7 + max_psycho*0.3
    total = count_score*0.3 + len_score*0.3 + psycho_score*0.4
    return round(min(total,100),2)

def score_attributes(attributes):
    if not attributes:
        return 0.0
    count = len(attributes)
    count_score = min(count/15, 1.0)*100
    avg_len = sum(len(str(a.get("value",""))) for a in attributes)/count
    len_score = min(avg_len/20, 1.0)*100
    has_number = any(any(c.isdigit() for c in str(a.get("value",""))) for a in attributes)
    numeric_bonus = 10 if has_number else 0
    total = count_score*0.5 + len_score*0.4 + numeric_bonus
    return round(min(total,100),2)

def score_important(info):
    return 100.0 if info and info.get("items") else 0.0

def score_aplus(aplus):
    if not aplus:
        return 0.0
    modules = aplus.get("modules", [])
    mod_score = min(len(modules)/5, 1.0)*60
    video_bonus = min(len(aplus.get("rawVideos", [])), 2)*15
    img_count = sum(1 for img in aplus.get("rawImages", []) if img.get("url"))
    img_bonus = min(img_count,3)*3
    return round(min(mod_score+video_bonus+img_bonus,100),2)

def score_video(count):
    if count is None or count==0:
        return 0.0
    return 100.0 if count>=3 else (90.0 if count==2 else 70.0)

def score_images(asin, image_dict):
    if asin not in image_dict:
        return 20.0
    info = image_dict[asin]
    thumb = info.get("thumbnail_score",0)
    thumb_purchase = info.get("thumbnail_purchase",0)
    main_score = thumb*0.6 + thumb_purchase*0.4
    detail_visual = info.get("detail_visual_score",0)
    detail_risk = info.get("detail_risk",0)
    detail_score = detail_visual*0.6 + detail_risk*0.4
    aplus_mean = info.get("aplus_mean_score",0)
    aplus_trust = info.get("aplus_trust",0)
    aplus_score = aplus_mean*0.5 + aplus_trust*0.5
    total = main_score*0.40 + detail_score*0.40 + aplus_score*0.20
    return round(total,2)

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
            asin = f"Unknown_{idx+1}"
        if idx % 10 == 0:
            print(f"  评论进度 {idx+1}/{len(review_data)}")

        stars_br = rev.get("starsBreakdown") or {}
        avg_stars = (stars_br.get("5star",0)*5 + stars_br.get("4star",0)*4 +
                     stars_br.get("3star",0)*3 + stars_br.get("2star",0)*2 + stars_br.get("1star",0)*1)
        ai_summary = rev.get("aiReviewsSummary") or {}
        keywords = ai_summary.get("keywords", [])
        pos_mentions = sum(kw.get("customersMentionedCount",{}).get("total",0) for kw in keywords if kw.get("sentiment")=="positive")
        neg_mentions = sum(kw.get("customersMentionedCount",{}).get("total",0) for kw in keywords if kw.get("sentiment")=="negative")
        total_mentions = pos_mentions + neg_mentions
        ai_pos_ratio = pos_mentions/total_mentions if total_mentions>0 else 0.5

        all_reviews = []
        for r in rev.get("productPageReviews", []):
            desc = r.get("reviewDescription","")
            rating = r.get("ratingScore")
            if desc and rating:
                all_reviews.append((desc, rating))
        for r in rev.get("productPageReviewsFromOtherCountries", []):
            desc = r.get("reviewDescription","")
            rating = r.get("ratingScore")
            if desc and rating:
                all_reviews.append((desc, rating))

        sentiment_scores = []
        if sentiment_pipeline:
            for text, _ in all_reviews:
                if len(text.strip())<3:
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
                            score = probs.get('positive',0) - probs.get('negative',0)
                        else:
                            score_map = {'LABEL_0':-1, 'LABEL_1':0, 'LABEL_2':1}
                            score = sum(probs.get(label,0)*score_map[label] for label in score_map)
                        sentiment_scores.append(score)
                except:
                    pass
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        if sentiment_scores:
            positive_ratio = sum(1 for s in sentiment_scores if s>0.3) / len(sentiment_scores)
            neutral_ratio = sum(1 for s in sentiment_scores if -0.3<=s<=0.3) / len(sentiment_scores)
            negative_ratio = sum(1 for s in sentiment_scores if s<-0.3) / len(sentiment_scores)
        else:
            positive_ratio = neutral_ratio = negative_ratio = 0

        total_comments = len(all_reviews)
        stars_norm = avg_stars/5 if avg_stars else 0.5
        sent_norm = (avg_sentiment+1)/2 if avg_sentiment is not None else 0.5
        pos_norm = positive_ratio if positive_ratio is not None else 0.5
        count_norm = min(math.log(total_comments+1)/math.log(1001),1.0) if total_comments else 0.0
        ai_norm = ai_pos_ratio if ai_pos_ratio is not None else 0.5
        w = [0.30,0.30,0.20,0.10,0.10]
        final_score = (stars_norm*w[0] + sent_norm*w[1] + pos_norm*w[2] + count_norm*w[3] + ai_norm*w[4])*100
        results.append({
            "ASIN": asin,
            "Avg_Stars": round(avg_stars,2),
            "Total_Reviews_Count": total_comments,
            "Avg_Sentiment_Score": round(avg_sentiment,3),
            "Positive%": round(positive_ratio*100,1),
            "Neutral%": round(neutral_ratio*100,1),
            "Negative%": round(negative_ratio*100,1),
            "Conversion_Score": round(final_score,2)
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
            image_rows.append({"asin":asin, "image_type":"high_resolution", "image_index":idx, "image_url":url})
        aplus = product.get("aPlusContent")
        if aplus:
            def extract_urls(obj):
                urls=[]
                if isinstance(obj,dict):
                    for k,v in obj.items():
                        if k=="url" and isinstance(v,str):
                            urls.append(v)
                        else:
                            urls.extend(extract_urls(v))
                elif isinstance(obj,list):
                    for item in obj:
                        urls.extend(extract_urls(item))
                return urls
            aplus_urls = extract_urls(aplus)
            for idx, url in enumerate(aplus_urls):
                image_rows.append({"asin":asin, "image_type":"a_plus", "image_index":idx, "image_url":url})
    for product in search_list:
        asin = product.get("asin")
        thumb = product.get("thumbnailImage")
        if thumb:
            image_rows.append({"asin":asin, "image_type":"thumbnail", "image_index":0, "image_url":thumb})
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
    prices = [p.get("price",{}).get("value") for p in search_list if p.get("price")]
    search_results = []
    for idx, item in enumerate(search_list):
        asin = item.get("asin")
        print(f"  搜索 ASIN {idx+1}/{len(search_list)}: {asin}")
        title = item.get("title")
        price = item.get("price",{}).get("value")
        stars = item.get("stars")
        reviews = item.get("reviewsCount")
        position = item.get("categoryPageData",{}).get("productPosition")
        title_psych = text2score.get(title, 0.0) if title else 0.0
        title_score_val = title_psych*0.8 + min(len(title)/100,1.0)*20 if title else 0
        if title:
            has_num = any(c.isdigit() for c in title)
            has_unit = any(u in title.lower() for u in ['cm','mm','kg','g','ml','l','w','h'])
            title_score_val += (5 if has_num else 0) + (5 if has_unit else 0)
        title_score_val = round(min(title_score_val,100),2)
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
        if stars is None: stars=0
        if reviews is None: reviews=0
        rating = stars/5
        review_norm = min(math.log(reviews+1)/math.log(100000),1) if reviews>0 else 0
        trust = rating*0.6 + review_norm*0.4
        trust_score_val = round(trust*100,2)
        if position is not None and position>0:
            pos_score = 1/math.log(position+2)
        else:
            pos_score = 0
        pos_score = round(pos_score*100,2)
        search_score = (title_score_val*0.25 + thumb_score*0.30 + price_score*0.15 +
                        trust_score_val*0.20 + pos_score*0.10)
        search_score = round(search_score,2)
        search_results.append({
            "asin": asin,
            "title_score": title_score_val,
            "thumbnail_score": thumb_score,
            "price_score": price_score,
            "trust_score": trust_score_val,
            "position_score": pos_score,
            "search_score": search_score,
            "stars": stars,
            "reviews": reviews
        })
        if (idx+1) % 5 == 0 or idx == len(search_list)-1:
            print(f"  搜索进度 {idx+1}/{len(search_list)}")
    search_df = pd.DataFrame(search_results)
    search_df["search_rank"] = search_df["search_score"].rank(ascending=False, method="min")
    print(f"✅ 搜索页评分完成，耗时 {time.time()-t_search:.2f} 秒")
    progress_bar.progress(70, "搜索页评分完成")

    # Listing评分
    print("📄 开始详情页评分...")
    t_list = time.time()
    listing_results = []
    for idx, listing in enumerate(listing_list):
        asin = listing.get("originalAsin")
        if not asin:
            continue
        print(f"  详情 ASIN {idx+1}/{len(listing_list)}: {asin}")
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
        weights = {"features":0.30, "attributes":0.25, "important":0.05, "aplus":0.10, "video":0.05, "image":0.25}
        base = (feat_s*weights["features"] + attr_s*weights["attributes"] +
                imp_s*weights["important"] + aplus_s*weights["aplus"] +
                vid_s*weights["video"] + img_s*weights["image"])
        brand_bonus = 10 if brand_s>0 else 0
        search_row = search_df[search_df["asin"]==asin]
        if not search_row.empty:
            stars = search_row.iloc[0].get("stars", 0)
            reviews = search_row.iloc[0].get("reviews", 0)
        else:
            stars = reviews = 0
        trust_bonus = 0
        if stars>=4.5: trust_bonus += 4
        elif stars>=4.2: trust_bonus += 2
        if reviews>1000: trust_bonus += 3
        elif reviews>500: trust_bonus += 1.5
        elif reviews>100: trust_bonus += 0.5
        final = min(base + brand_bonus + trust_bonus, 100)
        listing_results.append({
            "asin": asin,
            "bullet_score": feat_s,
            "attributes_score": attr_s,
            "important_score": imp_s,
            "aplus_score": aplus_s,
            "video_score": vid_s,
            "image_score": img_s,
            "brand_bonus": brand_bonus,
            "trust_bonus": round(trust_bonus,2),
            "listing_score": round(final,2)
        })
        if (idx+1) % 5 == 0 or idx == len(listing_list)-1:
            print(f"  详情进度 {idx+1}/{len(listing_list)}")
    listing_df = pd.DataFrame(listing_results)
    listing_df["listing_rank"] = listing_df["listing_score"].rank(ascending=False, method="min")
    print(f"✅ 详情页评分完成，耗时 {time.time()-t_list:.2f} 秒")
    progress_bar.progress(85, "Listing评分完成")

    # 评论分析
    print("📝 开始评论分析...")
    t_rev = time.time()
    sentiment_pipeline = load_sentiment_pipeline()
    review_df = analyze_reviews(review_list, sentiment_pipeline)
    print(f"✅ 评论分析完成，耗时 {time.time()-t_rev:.2f} 秒")
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
    print(f"✅ 全部分析完成！总耗时 {time.time()-total_start:.2f} 秒")
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
    st.dataframe(final_df, width='stretch')

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
        fig, ax = plt.subplots(figsize=(8,4))
        final_df[["search_score","Detail_Conversion","Total_Score"]].boxplot(ax=ax)
        st.pyplot(fig)

    st.subheader("🔍 单品对比分析")
    asin_list = final_df["asin"].tolist()
    selected_asin = st.selectbox("选择或输入 ASIN", asin_list)

    if selected_asin:
        row = full_df[full_df["asin"]==selected_asin].iloc[0]
        numeric_cols = ["search_score","Detail_Conversion","Total_Score","listing_score","Conversion_Score"]
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
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True)
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
else:
    st.info("请上传原始爬虫 JSON 文件开始分析")