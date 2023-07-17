import streamlit as st
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Embedding Function
sentence_transformer_ef = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Store
vectordb = Chroma(collection_name='books', persist_directory='db', embedding_function=sentence_transformer_ef)

# Function to load the DataFrame
def load_dataframe(filename):
    df = pd.read_csv(filename)  # Load from CSV file
    return df

# Load dataframe
df = load_dataframe('books_df.csv')

def recommend_books(user_text):
    books = []
    results = vectordb.similarity_search(user_text, k=5) 
    result_ids = [doc.metadata['id'] for doc in results]
    for id in result_ids:
        book = {
            "title": df.iloc[id]['title'],
            "author": df.iloc[id]['author'],
            "description": df.iloc[id]['description'],
            "image_url": df.iloc[id]['coverImg']
        }
        books.append(book)
    return books

# Streamlit app
def main():
    # Set page title
    st.set_page_config(page_title="Book Recommendation System", page_icon=":books:")

    # Custom CSS styling
    st.markdown(
        """
        <style>
        .title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #FF4B4B;
        }
        .author {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .description {
            margin-bottom: 10px;
        }
        .book-info {
            display: flex;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        .book-image {
            max-width: 100px;
            margin-right: 20px;
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add title and description
    st.title('Book Recommendation')
    st.markdown('<div class="title">Enter some text related to books, and we will recommend 5 books for you.</div>', unsafe_allow_html=True)

    # Add text input
    user_input = st.text_area('Enter your text here:', height=100)

    # Add recommendation button
    if st.button('Recommend'):
        # Call recommend_books function
        if user_input.strip() == '':
            st.error('Please enter some text related to books.')
        else:    
            recommended_books = recommend_books(user_input)

            # Display recommended books
            st.markdown('## Recommended Books:')
            for book in recommended_books:
                st.markdown('<div class="book-info"><img class="book-image" src="' + book["image_url"] + '"><div><div class="title">' + book["title"] + '</div><div class="author">' + book["author"] + '</div><div class="description">' + book["description"] + '</div></div></div>', unsafe_allow_html=True)
                st.markdown('---')

# Run the app
if __name__ == '__main__':
    main()