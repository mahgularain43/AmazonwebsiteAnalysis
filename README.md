# 🛍️ Amazon Product Data Analysis & Recommendation System

This project is a **Streamlit web app** for analyzing **Amazon product datasets**.  
It combines **data preprocessing, visualization, sentiment analysis, machine learning, and product recommendations** into one tool.  

---

## 🚀 Features

### 1. **Data Preprocessing & Cleaning**
- Handles missing values, invalid ratings, and inconsistent price formats.
- Converts prices (₹), ratings, and review counts into numeric formats.
- Adds derived features such as `category` and `brand`.

### 2. **Exploratory Data Analysis (EDA)**
- 📊 Visualizes price distributions (discount vs actual prices).
- ⭐ Shows rating distributions and correlations with prices.
- 🔍 Review count analysis: impact on product popularity.
- 🏷️ Product listing frequency by category and subcategory.

### 3. **Machine Learning: Sales Prediction**
- Predicts **sales** using `discount_price`, `actual_price`, `ratings`, and `reviews`.
- Built with **Random Forest Regressor**.
- Evaluation metrics: **Mean Squared Error (MSE)** and **R² Score**.
- Plots **actual vs predicted sales** and top product comparisons.

### 4. **Image Analysis**
- Downloads product images from URLs.
- Converts to grayscale with OpenCV.
- Uses **KMeans clustering** to detect dominant colors.

### 5. **Keyword Analysis**
- Cleans product titles and descriptions.
- Generates **word clouds**.
- Finds most common keywords across categories.

### 6. **Competitive & Brand Analysis**
- Average ratings, total reviews, discount percentages by brand.
- Top-20 brands visualization.
- Subcategory-wise product comparisons.

### 7. **Sentiment Analysis**
- Cleans and tokenizes review text.
- Uses **TextBlob** for polarity scoring.
- Categorizes reviews into **positive, neutral, negative**.
- Plots sentiment distribution and word clouds.

### 8. **Product Recommendation System**
- Built with **TF-IDF + Cosine Similarity**.
- Suggests similar products based on review text.
- Input: Product ASIN → Output: Top 5 similar products.

---

## 🛠 Tech Stack
- **Frontend/UI**: Streamlit
- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (RandomForest, KMeans, TF-IDF)
- **Deep Learning**: TensorFlow (MobileNetV2 for image features)
- **NLP**: NLTK, TextBlob
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Image Processing**: OpenCV, PIL

---

## 📂 Project Structure
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
├── sample_data.csv # Example dataset (Amazon products)
└── README.md # Project documentation


## ▶️ Run Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/amazon-data-analysis.git
   cd amazon-data-analysis


## 📊 Sample Visualizations

Actual vs Predicted Sales

Distribution of Product Ratings

Word Cloud of Reviews

Top Brands by Ratings and Discounts

Sentiment Distribution

## 📌 Future Enhancements

Deploy online with Streamlit Cloud / HuggingFace Spaces.

Enhance recommendation engine using transformer-based embeddings (BERT/SBERT).

Add dashboard filters for category-wise exploration.

Expand image analysis with deep learning feature extraction.
