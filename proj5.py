import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Set dashboard style
st.set_page_config(layout="wide")
sns.set_theme(style="whitegrid")

# Load model & encoders
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load dataset
df = pd.read_csv('C:\\Users\\LAVANYA\\Desktop\\Project5\\chatgpt_reviews - chatgpt_reviews.csv')
df['review'] = df['review'].astype(str).fillna('')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype('Int64')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['rating'])

# Predict sentiment
X = tfidf.transform(df['review'])
df['sentiment'] = le.inverse_transform(model.predict(X))

# Sidebar navigation
st.sidebar.title("ChatGPT Review Sentiment Dashboard")
question = st.sidebar.selectbox("Choose an Analysis Type", [
    "1. Overall Sentiment Distribution",
    "2. Sentiment vs Rating",
    "3. Keywords per Sentiment (WordCloud)",
    "4. Sentiment Trends Over Time",
    "5. Verified vs Non-Verified Sentiment",
    "6. Review Length vs Sentiment",
    "7. Sentiment by Location",
    "8. Sentiment by Platform",
    "9. Sentiment by Version",
    "10. Negative Feedback Themes"
])

# Function to display matplotlib charts in Streamlit
def display_chart(fig):
    st.pyplot(fig, clear_figure=True)
# Question 1: Overall Sentiment Distribution (Pie Chart)
if question == "1. Overall Sentiment Distribution":
    st.header("1️⃣ Overall Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           startangle=140, colors=['#66BB6A', '#FFEE58', '#EF5350'])
    ax.axis('equal')
    st.pyplot(fig) 

# Question 2: Sentiment vs Rating (Stacked Bar Chart)
elif question == "2. Sentiment vs Rating":
    st.header("2️⃣ Sentiment Distribution by Rating")
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
    rating_sentiment.plot(kind='barh', stacked=True, colormap='Accent')
    st.pyplot(plt)

# Question 3: Keywords per Sentiment (WordCloud)
elif question == "3. Keywords per Sentiment (WordCloud)":
    st.header("3️⃣ Keywords Associated with Each Sentiment")
    for sentiment in df['sentiment'].unique():
        st.subheader(f"{sentiment.capitalize()} Reviews")
        text = " ".join(df[df['sentiment'] == sentiment]['review'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array())

# Question 4: Sentiment Trends Over Time (Line Plot)
elif question == "4. Sentiment Trends Over Time":
    st.header("4️⃣ Sentiment Trends Over Time")
    df['month'] = df['date'].dt.to_period('M').astype(str)
    trend = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend, x='month', y='count', hue='sentiment', marker='o', ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    display_chart(fig)

# Question 5: Verified vs Non-Verified Sentiment
elif question == "5. Verified vs Non-Verified Sentiment":
    st.header("5️⃣ Sentiment by Verified Purchase")
    verified = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index') * 100
    verified.plot(kind='bar', colormap='coolwarm')
    plt.ylabel("Percentage")
    st.pyplot(plt)

# Question 6: Review Length vs Sentiment
elif question == "6. Review Length vs Sentiment":
    st.header("6️⃣ Review Length vs Sentiment")
    df['review_length'] = df['review'].apply(len)
    fig, ax = plt.subplots()
    sns.boxplot(x='sentiment', y='review_length', data=df, palette='Set2')
    plt.ylabel("Review Length (characters)")
    display_chart(fig)

# Question 7: Sentiment by Location
elif question == "7. Sentiment by Location":
    st.header("7️⃣ Sentiment by Location (Top 10)")
    top_locations = df['location'].value_counts().head(10).index
    location_data = pd.crosstab(df[df['location'].isin(top_locations)]['location'], df['sentiment'], normalize='index')*100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(location_data, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.ylabel("Location")
    display_chart(fig)

# Question 8: Sentiment by Platform
elif question == "8. Sentiment by Platform":
    st.header("8️⃣ Sentiment by Platform (Web vs Mobile)")
    platform_data = pd.crosstab(df['platform'], df['sentiment'], normalize='index')*100
    platform_data.plot(kind='bar', stacked=True, colormap='Pastel1')
    plt.ylabel("Percentage")
    st.pyplot(plt)

# Question 9: Sentiment by Version
elif question == "9. Sentiment by Version":
    st.header("9️⃣ Sentiment by ChatGPT Version")
    version_data = pd.crosstab(df['version'], df['sentiment'], normalize='index')*100
    version_data.plot(kind='bar', colormap='viridis')
    plt.xticks(rotation=45)
    plt.ylabel("Percentage")
    st.pyplot(plt)

# Question 10: Negative Feedback Themes
elif question == "10. Negative Feedback Themes":
    st.header("🔟 Common Themes in Negative Reviews")
    neg_reviews = df[df['sentiment'].str.lower() == 'negative']['review']

    if neg_reviews.empty:
        st.warning("⚠️ No negative reviews found to analyze themes.")
    else:
        count_vec = CountVectorizer(stop_words='english', max_features=15)
        neg_counts = count_vec.fit_transform(neg_reviews)
        neg_freq = pd.Series(neg_counts.toarray().sum(axis=0),
                             index=count_vec.get_feature_names_out()).sort_values(ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x=neg_freq.values, y=neg_freq.index, palette='Reds_r')
        plt.xlabel("Frequency")
        plt.title("Top Keywords in Negative Reviews")
        st.pyplot(fig)
