import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from helper import fetch_stats, create_wordcloud, most_busy_users, most_common_words, emoji_helper
from helper import most_busy_users_sentiment, create_sentiment_wordcloud, add_sentiment_column
from helper import monthly_timeline, daily_timeline , month_activity_map , week_activity_map, activity_heatmap
from preprocessor import preprocess
from fpdf import FPDF
import tempfile
import os

# Set up the Streamlit app title
st.title("WhatsApp Chat Analyzer with Sentiment")

# Upload and preprocess the data
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = uploaded_file.read().decode("utf-8")
    df = preprocess(data)
    df = add_sentiment_column(df)

    # Show statistics
    st.header("Overall Statistics")
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.selectbox("Show analysis for", user_list)

    # PDF generation function
    def generate_pdf(selected_user, df, stats):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="WhatsApp Chat Analysis Report", ln=True, align='C')
        pdf.ln(10)
        
        # Add basic statistics
        pdf.cell(200, 10, txt=f"Selected User: {selected_user}", ln=True)
        pdf.cell(200, 10, txt=f"Messages: {stats['messages']}, Words: {stats['words']}, Media: {stats['media']}, Links: {stats['links']}", ln=True)
        pdf.ln(10)

        # Monthly timeline example (saves plot to image, then adds it to PDF)
        with tempfile.TemporaryDirectory() as tmpdirname:
            fig, ax = plt.subplots()
            ax.plot(df['time'], df['message'], label='Messages')
            plt.savefig(os.path.join(tmpdirname, "timeline.png"))
            plt.close(fig)
            pdf.image(os.path.join(tmpdirname, "timeline.png"), x=10, w=180)
        
        return pdf.output(dest='S').encode('latin1')

    if st.button("Show Analysis"):
        # Fetch Stats
        num_messages, words, media_messages, links = fetch_stats(selected_user, df)
        
        # Add additional sentiment statistics
        num_positive = df[df['sentiment'] == 'Positive'].shape[0]
        num_neutral = df[df['sentiment'] == 'Neutral'].shape[0]
        num_negative = df[df['sentiment'] == 'Negative'].shape[0]
        
        st.write(f"Messages: {num_messages}, Words: {words}, Media: {media_messages}, Links: {links}")
        st.write(f"Positive messages: {num_positive}, Neutral messages: {num_neutral}, Negative messages: {num_negative}")
        
        # Monthly and Daily Timelines
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Message Count by Sentiment")
            monthly_df_positive = monthly_timeline(selected_user, df, sentiment_type="Positive")
            monthly_df_neutral = monthly_timeline(selected_user, df, sentiment_type="Neutral")
            monthly_df_negative = monthly_timeline(selected_user, df, sentiment_type="Negative")

            fig, ax = plt.subplots()
            ax.plot(monthly_df_positive['time'], monthly_df_positive['message'], color='blue', label="Positive")
            ax.plot(monthly_df_neutral['time'], monthly_df_neutral['message'], color='black', label="Neutral")
            ax.plot(monthly_df_negative['time'], monthly_df_negative['message'], color='red', label="Negative")
            ax.set_title("Monthly Message Count by Sentiment")
            ax.set_xlabel("Month-Year")
            ax.set_ylabel("Number of Messages")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Daily Message Count by Sentiment")
            daily_df_positive = daily_timeline(selected_user, df, sentiment_type="Positive")
            daily_df_neutral = daily_timeline(selected_user, df, sentiment_type="Neutral")
            daily_df_negative = daily_timeline(selected_user, df, sentiment_type="Negative")

            fig, ax = plt.subplots()
            ax.plot(daily_df_positive['only_date'], daily_df_positive['message'], color='blue', label="Positive")
            ax.plot(daily_df_neutral['only_date'], daily_df_neutral['message'], color='black', label="Neutral")
            ax.plot(daily_df_negative['only_date'], daily_df_negative['message'], color='red', label="Negative")
            ax.set_title("Daily Message Count by Sentiment")
            st.pyplot(fig)

        # Wordclouds for each sentiment
        st.subheader("Wordclouds by Sentiment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Positive")
            wordcloud_positive = create_sentiment_wordcloud(selected_user, df, "Positive")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_positive, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.subheader("Neutral")
            wordcloud_neutral = create_sentiment_wordcloud(selected_user, df, "Neutral")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neutral, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with col3:
            st.subheader("Negative")
            wordcloud_negative = create_sentiment_wordcloud(selected_user, df, "Negative")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_negative, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # Weekly Activity Heatmap
        st.subheader("Weekly Activity Heatmap")
        activity_heatmap_df = activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        sns.heatmap(activity_heatmap_df, cmap=cmap, linewidths=0.3, annot=False, ax=ax)
        ax.set_title("Weekly Activity Heatmap")
        st.pyplot(fig)

        # Sentiment Distribution Pie Chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

        # PDF Download
        if st.button("Download Analysis Report"):
            pdf_data = generate_pdf(selected_user, df, {
                "messages": num_messages,
                "words": words,
                "media": media_messages,
                "links": links,
            })
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="chat_analysis_report.pdf",
                mime="application/pdf"
            )
