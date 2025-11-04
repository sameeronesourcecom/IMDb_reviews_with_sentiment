# IMDb Movie Review Sentiment Analysis

Analyzed 50,000 IMDb reviews using Python, TextBlob, and WordCloud.

import kagglehub

01. # Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print("Path to dataset files:", path)

02. import pandas as pd
# Replace with actual path if needed
df = pd.read_csv(path + "/IMDB Dataset.csv")
df.head()

03. print("Total reviews:", len(df))
print("Missing values:", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Remove duplicates and nulls
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

04. pip install textblob

05. from textblob import TextBlob

# Add sentiment polarity column
df['Sentiment_Score'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)

06. import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(df['Sentiment_Score'], bins=50, color='skyblue', edgecolor='black')
plt.title('Sentiment Score Distribution')
plt.xlabel('Polarity')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.show()

07. pip install wordcloud

08. from wordcloud import WordCloud

text = " ".join(df['review'].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in IMDb Reviews')
plt.show()

09. df.to_csv("IMDb_reviews_with_sentiment.csv", index=False)

10. from google.colab import files
files.download("IMDb_reviews_with_sentiment.csv")





## Tools Used
- Python
- Pandas
- TextBlob
- Matplotlib
- WordCloud
- Google Colab

## Files
- `IMDb_reviews_with_sentiment.csv`: Cleaned reviews with sentiment scores
- `wordcloud.png`: Word cloud of common review words
- `sentiment_chart.png`: Sentiment score distribution

## Author
Sameer Basha Shaik
