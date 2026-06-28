#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
差评挖掘机 - Amazon Gift Cards 评论分析
一键运行：python bad_review_miner.py
"""
import os
import re
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# ==================== 配置区 ====================
# 请在此处填入你的百炼 API Key
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your-api-key-here")
CSV_PATH = "amazon_reviews.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== API 调用函数 ====================
def call_qwen(messages, model="qwen-max", max_retries=3):
    """调用百炼 API"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": {"messages": messages},
        "parameters": {"result_format": "message", "max_tokens": 2000}
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            data = resp.json()
            if "output" in data and "choices" in data["output"]:
                return data["output"]["choices"][0]["message"]["content"]
            if "output" in data and "text" in data["output"]:
                return data["output"]["text"]
            print(f"API 返回异常: {data}")
        except Exception as e:
            print(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    return None


# ==================== 数据清洗 ====================
def clean_text(text):
    """清洗评论文本"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_clean_data(path):
    """加载并清洗数据"""
    print("=" * 50)
    print("【步骤 1】加载并清洗数据")
    print("=" * 50)
    df = pd.read_csv(path)
    print(f"原始数据量: {len(df)} 条")
    
    # 清洗文本
    df['text_clean'] = df['text'].apply(clean_text)
    df['title_clean'] = df['title'].apply(clean_text)
    
    # 组合标题和正文
    df['full_text'] = df['title_clean'].fillna('') + '. ' + df['text_clean']
    df['full_text'] = df['full_text'].str.replace(r'^\.\s*', '', regex=True)
    
    # 去除空评论
    df = df[df['full_text'].str.len() > 5].copy()
    
    print(f"清洗后数据量: {len(df)} 条")
    print(f"字段: {df.columns.tolist()}")
    print(f"评分分布:\n{df['rating'].value_counts().sort_index()}")
    return df


# ==================== 情感分类 ====================
def classify_sentiment_by_rating(rating):
    """基于评分做初步情感分类"""
    if rating <= 2:
        return "负面"
    elif rating == 3:
        return "中性"
    else:
        return "正面"


def batch_ai_classify(df, batch_size=15):
    """用 AI 对评论做批量分类和关键词提取"""
    print("\n" + "=" * 50)
    print("【步骤 2】AI 批量分类与关键词提取")
    print("=" * 50)
    
    # 先基于评分做初步分类
    df['sentiment'] = df['rating'].apply(classify_sentiment_by_rating)
    
    # 只对负面和中性评论做 AI 深度分析（节省 API 调用）
    need_ai = df[df['sentiment'].isin(['负面', '中性'])].copy()
    print(f"需要 AI 深度分析的评论数: {len(need_ai)} 条（负面+中性）")
    
    ai_results = {}
    
    # 分批处理
    total = len(need_ai)
    for i in range(0, total, batch_size):
        batch = need_ai.iloc[i:i+batch_size]
        batch_texts = []
        idx_map = {}
        for j, (idx, row) in enumerate(batch.iterrows()):
            batch_texts.append(f"{j+1}. {row['full_text'][:300]}")
            idx_map[j+1] = idx
        
        prompt = f"""你是一个电商评论情感分析专家。请对以下 {len(batch_texts)} 条评论进行分析：
1. 判断情感：正面 / 负面 / 中性
2. 提取 1-3 个关键词（如：包装破损、余额不对、物流慢等）
3. 如果负面，简要说明核心槽点（10字以内）

请严格按以下格式输出，每行一条：
1. [情感] | [关键词1, 关键词2] | [槽点简述或无]
2. [情感] | [关键词1, 关键词2] | [槽点简述或无]
...

评论列表：
""" + "\n".join(batch_texts)

        messages = [
            {"role": "system", "content": "你是一个专业的电商评论分析助手，只输出指定格式的结果，不做额外解释。"},
            {"role": "user", "content": prompt}
        ]
        
        print(f"  处理批次 {i//batch_size + 1}/{(total-1)//batch_size + 1} ({i+1}-{min(i+batch_size, total)})")
        result = call_qwen(messages)
        
        if result:
            # 解析结果
            for line in result.strip().split('\n'):
                line = line.strip()
                match = re.match(r'(\d+)\.\s*\[?([^\]|]+)\]?\s*\|\s*\[?([^\]|]*)\]?\s*(?:\|\s*(.*))?', line)
                if match:
                    num = int(match.group(1))
                    sentiment = match.group(2).strip()
                    keywords = match.group(3).strip() if match.group(3) else ""
                    complaint = match.group(4).strip() if match.group(4) else ""
                    if num in idx_map:
                        orig_idx = idx_map[num]
                        ai_results[orig_idx] = {
                            'ai_sentiment': sentiment,
                            'keywords': keywords,
                            'complaint': complaint
                        }
        
        time.sleep(0.5)  # 避免速率限制
    
    # 合并结果
    df['ai_sentiment'] = df.index.map(lambda x: ai_results.get(x, {}).get('ai_sentiment', ''))
    df['keywords'] = df.index.map(lambda x: ai_results.get(x, {}).get('keywords', ''))
    df['complaint'] = df.index.map(lambda x: ai_results.get(x, {}).get('complaint', ''))
    
    # 用 AI 结果覆盖初步分类
    df.loc[df['ai_sentiment'] != '', 'sentiment'] = df.loc[df['ai_sentiment'] != '', 'ai_sentiment']
    
    print(f"\nAI 分析完成！最终情感分布:")
    print(df['sentiment'].value_counts())
    return df


# ==================== 负面归因分析 ====================
def analyze_negative_reviews(df):
    """对负面评论进行深度归因"""
    print("\n" + "=" * 50)
    print("【步骤 3】负面评论深度归因")
    print("=" * 50)
    
    negative = df[df['sentiment'] == '负面'].copy()
    print(f"负面评论总数: {len(negative)} 条")
    
    if len(negative) == 0:
        print("没有负面评论，跳过归因分析")
        return None
    
    # 选取有具体文本的负面评论
    neg_texts = negative['full_text'].dropna().tolist()
    
    # 分批给 AI 做归因（每批 20 条）
    all_complaints = []
    batch_size = 20
    for i in range(0, len(neg_texts), batch_size):
        batch = neg_texts[i:i+batch_size]
        batch_str = "\n".join([f"- {t[:400]}" for t in batch])
        
        prompt = f"""你是一个产品分析师。请阅读以下负面评论，归纳核心槽点。
对每条评论，输出格式：槽点简述（如"余额不对"、"包装破损"、"到账延迟"等，尽量简短）

负面评论：
{batch_str}

请列出所有槽点（每条评论一个）："""
        
        messages = [
            {"role": "system", "content": "你是一个专业的产品分析师，擅长从用户反馈中提炼问题。"},
            {"role": "user", "content": prompt}
        ]
        
        print(f"  分析负面批次 {i//batch_size + 1}/{(len(neg_texts)-1)//batch_size + 1}")
        result = call_qwen(messages)
        if result:
            for line in result.strip().split('\n'):
                line = line.strip('- ').strip()
                if line and len(line) < 50:
                    all_complaints.append(line)
        time.sleep(0.5)
    
    # 统计槽点频率
    complaint_counter = Counter(all_complaints)
    top_complaints = complaint_counter.most_common(10)
    
    print(f"\nTop 10 槽点:")
    for complaint, count in top_complaints:
        pct = count / len(negative) * 100
        print(f"  - {complaint}: {count}次 ({pct:.1f}%)")
    
    return top_complaints


# ==================== 可视化 ====================
def generate_visualizations(df, top_complaints):
    """生成可视化图表"""
    print("\n" + "=" * 50)
    print("【步骤 4】生成可视化图表")
    print("=" * 50)
    
    # 1. 评分分布柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 评分分布
    rating_counts = df['rating'].value_counts().sort_index()
    axes[0, 0].bar(rating_counts.index, rating_counts.values, color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60'])
    axes[0, 0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Star Rating')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(rating_counts.values):
        axes[0, 0].text(rating_counts.index[i], v + 5, str(v), ha='center', fontsize=10)
    
    # 情感分布饼图
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'负面': '#e74c3c', '中性': '#f39c12', '正面': '#27ae60'}
    pie_colors = [colors.get(s, '#95a5a6') for s in sentiment_counts.index]
    axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   colors=pie_colors, startangle=90)
    axes[0, 1].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    
    # 槽点分布柱状图（Top 8）
    if top_complaints:
        complaints, counts = zip(*top_complaints[:8])
        axes[1, 0].barh(range(len(complaints)), counts, color='#e74c3c')
        axes[1, 0].set_yticks(range(len(complaints)))
        axes[1, 0].set_yticklabels(complaints)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title('Top Complaints', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Count')
    else:
        axes[1, 0].text(0.5, 0.5, 'No complaints data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Top Complaints', fontsize=14, fontweight='bold')
    
    # 词云
    all_text = ' '.join(df['full_text'].dropna().astype(str))
    stopwords = set(STOPWORDS)
    stopwords.update(['gift', 'card', 'amazon', 'it', 'the', 'and', 'to', 'a', 'is', 'this', 'for'])
    
    if len(all_text) > 100:
        wc = WordCloud(width=800, height=400, background_color='white', 
                       stopwords=stopwords, max_words=100, colormap='viridis').generate(all_text)
        axes[1, 1].imshow(wc, interpolation='bilinear')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Word Cloud (All Reviews)', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient text for word cloud', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "analysis_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {chart_path}")
    
    # 负面评论词云
    neg_text = ' '.join(df[df['sentiment'] == '负面']['full_text'].dropna().astype(str))
    if len(neg_text) > 50:
        plt.figure(figsize=(10, 5))
        wc_neg = WordCloud(width=800, height=400, background_color='white',
                           stopwords=stopwords, max_words=80, colormap='Reds').generate(neg_text)
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud (Negative Reviews)', fontsize=14, fontweight='bold')
        neg_chart_path = os.path.join(OUTPUT_DIR, "negative_wordcloud.png")
        plt.savefig(neg_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"负面词云已保存: {neg_chart_path}")


# ==================== 生成报告 ====================
def generate_report(df, top_complaints):
    """调用百炼生成产品改进建议书"""
    print("\n" + "=" * 50)
    print("【步骤 5】生成产品改进建议书")
    print("=" * 50)
    
    # 统计信息
    total = len(df)
    neg_count = len(df[df['sentiment'] == '负面'])
    neu_count = len(df[df['sentiment'] == '中性'])
    pos_count = len(df[df['sentiment'] == '正面'])
    avg_rating = df['rating'].mean()
    
    # 选取代表性负面评论
    neg_samples = df[df['sentiment'] == '负面']['full_text'].dropna().head(15).tolist()
    neg_samples_str = "\n".join([f"- {t[:300]}" for t in neg_samples])
    
    # 槽点汇总
    complaints_str = ""
    if top_complaints:
        for c, count in top_complaints[:5]:
            pct = count / neg_count * 100 if neg_count > 0 else 0
            complaints_str += f"- {c}: {count}次 (约{pct:.1f}%)\n"
    
    prompt = f"""你是一位资深电商产品顾问。请基于以下 Amazon Gift Cards 产品的评论分析数据，撰写一份专业的《产品改进建议书》。

## 分析数据
- 总评论数: {total} 条
- 平均评分: {avg_rating:.2f} / 5.0
- 正面评价: {pos_count} 条 ({pos_count/total*100:.1f}%)
- 中性评价: {neu_count} 条 ({neu_count/total*100:.1f}%)
- 负面评价: {neg_count} 条 ({neg_count/total*100:.1f}%)

## 主要槽点 Top 5
{complaints_str}

## 代表性负面评论
{neg_samples_str}

## 输出要求
请按以下结构撰写报告（用中文，专业且 actionable）：

# Amazon Gift Cards 产品改进建议书

## 一、用户画像分析
（分析评论用户的特征、使用场景、核心诉求）

## 二、主要槽点 Top 3
（归纳最严重的 3 个问题，每个问题给出：现象描述 + 影响范围 + 严重程度）

## 三、竞品优势对比
（从评论中提炼用户期望但未被满足的点，对比行业最佳实践）

## 四、具体改进建议
（针对每个槽点给出 1-2 条可落地的改进方案）

## 五、优先级排序
（按影响程度和实施难度给出改进优先级）
"""
    
    messages = [
        {"role": "system", "content": "你是一位拥有10年经验的电商产品顾问，擅长数据驱动的用户洞察和产品优化。报告要专业、数据翔实、建议可落地。"},
        {"role": "user", "content": prompt}
    ]
    
    report = call_qwen(messages, max_retries=3)
    if report:
        report_path = os.path.join(OUTPUT_DIR, "product_improvement_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存: {report_path}")
        return report
    else:
        print("报告生成失败")
        return None


# ==================== 主函数 ====================
def main():
    print("\n" + "=" * 60)
    print("  差评挖掘机 v1.0 - Amazon Gift Cards 评论分析")
    print("  Powered by 阿里云百炼 (qwen-max)")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_and_clean_data(CSV_PATH)
    
    # 2. AI 批量分类
    df = batch_ai_classify(df)
    
    # 3. 负面归因
    top_complaints = analyze_negative_reviews(df)
    
    # 4. 可视化
    generate_visualizations(df, top_complaints)
    
    # 5. 生成报告
    report = generate_report(df, top_complaints)
    
    # 保存分析结果
    result_path = os.path.join(OUTPUT_DIR, "analyzed_reviews.csv")
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"\n分析结果已保存: {result_path}")
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("  分析完成！")
    print("=" * 60)
    print(f"输出文件目录: {os.path.abspath(OUTPUT_DIR)}")
    print("  - analyzed_reviews.csv     : 带分类标签的完整数据")
    print("  - analysis_charts.png      : 综合分析图表")
    print("  - negative_wordcloud.png   : 负面评论词云")
    print("  - product_improvement_report.md : 产品改进建议书")


if __name__ == "__main__":
    main()
