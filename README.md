# 差评挖掘机 v1.0：基于 Amazon 真实评论的电商产品体验洞察 Agent

## 项目简介

面向跨境电商（如 Amazon）运营中的 VOC（Voice of Customer，客户之声）分析场景。面对成千上万条非结构化的用户评论，传统人工阅读效率低下且主观性强。本工具基于阿里云百炼大模型（qwen-max），自动完成评论清洗、情感分类、负面归因、可视化分析和产品改进建议书生成，实现**一键运行，自动出结果**。

## 核心功能

1. **数据清洗**：自动去除 HTML 标签、空值、短评论
2. **AI 情感分类**：调用百炼 API 将评论分为正面/负面/中性，提取关键词
3. **负面归因分析**：针对差评深入归纳核心槽点（如余额不对、卡无法使用等）
4. **可视化**：评分分布、情感分布、槽点分布柱状图、词云图
5. **智能报告**：调用 qwen-max 生成包含用户画像、Top 3 槽点、竞品对比、改进建议的《产品改进建议书》

## 技术栈

- **AI 模型**：阿里云百炼（qwen-max）
- **数据处理**：Python + Pandas
- **可视化**：Matplotlib + WordCloud
- **数据源**：Amazon Reviews 2023（HuggingFace 开源数据集）

## 一键运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key（二选一）
export DASHSCOPE_API_KEY="your-api-key"
# 或直接修改 bad_review_miner.py 中的 API_KEY 变量

# 3. 运行
python bad_review_miner.py
```

## 输出文件

运行完成后，在 `output/` 目录生成：

| 文件 | 说明 |
|------|------|
| `analyzed_reviews.csv` | 带 AI 分类标签的完整数据 |
| `analysis_charts.png` | 综合分析图表（评分分布、情感分布、槽点、词云） |
| `negative_wordcloud.png` | 负面评论词云 |
| `product_improvement_report.md` | 产品改进建议书 |
| `complaints.txt` | Top 10 槽点统计 |

## 项目结构

```
bad-review-miner/
├── bad_review_miner.py          # 主脚本（一键运行）
├── amazon_reviews.csv          # 原始数据（2000条Amazon评论）
├── requirements.txt            # Python 依赖
├── .gitignore
├── README.md
└── output/                     # 输出目录
    ├── analysis_charts.png
    ├── negative_wordcloud.png
    ├── analyzed_reviews.csv
    ├── complaints.txt
    └── product_improvement_report.md
```

## 分析结果示例

- **数据量**：1995 条真实评论
- **平均评分**：4.46 / 5.0
- **情感分布**：正面 94.1%、负面 4.4%、中性 1.5%
- **Top 3 槽点**：余额不对、卡无法使用、使用麻烦且限制多

## License

MIT
