# -Swiggy-Reviews-Sentiment-Analysis
# 🍔 Swiggy Sentiment Analyzer

**Author:** Jaishree  
**Last Updated:** June 2025

A lightweight Python project to analyze Swiggy food delivery reviews using Natural Language Processing (NLP). This tool classifies reviews into positive, negative, or neutral sentiments and offers visual insights into various customer satisfaction metrics.

---

## 📌 Overview

This project performs sentiment analysis on customer reviews using **TextBlob**, and helps Swiggy (or similar platforms) uncover customer satisfaction patterns through:

- Sentiment classification
- Restaurant and city-level performance analysis
- Delivery time and order value impact
- Interactive data visualizations
- Exporting results for reporting

---

## 🎯 Key Features

- ✅ **Sentiment Analysis**: Positive, Negative, Neutral tagging using TextBlob  
- 🧠 **NLP Powered**: Uses polarity & subjectivity scoring  
- 📊 **Visual Dashboards**: 9 types of analytical charts  
- 🏙️ **City-Level Insight**: Track satisfaction by geography  
- 🍟 **Restaurant Comparison**: Know your best & worst performers  
- 🚚 **Delivery Impact**: Does speed affect sentiment?  
- 💰 **Spending Patterns**: Explore how order value links to feedback  
- 📁 **Export Support**: Outputs to CSV and TXT summary  

---

## 📈 Visualization Types

1. **Sentiment Distribution** – Pie chart of review categories  
2. **Rating Distribution** – Bar chart of 1 to 5-star ratings  
3. **Sentiment vs Rating** – Scatter plot for correlation  
4. **Delivery Time Analysis** – Box plot of delivery times  
5. **Restaurant Performance** – Bar chart of top eateries  
6. **City-wise Analysis** – Sentiment heatmap or bar plot  
7. **Order Value Distribution** – Spending patterns by sentiment  
8. **Polarity Histogram** – How extreme is customer feedback?  
9. **Subjectivity Chart** – Objective vs subjective review detection  

---

## 🛠 Technologies Used

- **Python 3.7+**
- `pandas` for data handling  
- `numpy` for computation  
- `matplotlib` + `seaborn` for plotting  
- `textblob` for sentiment analysis  

---

## 🧪 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/swiggy-sentiment-analysis.git
cd swiggy-sentiment-analysis

# (Optional) Create virtual environment
python -m venv env
source env/bin/activate  # Or use env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
