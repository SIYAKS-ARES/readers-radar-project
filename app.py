from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

def normalize_tag_name(tag_name):
    if pd.isnull(tag_name):
        return ''
    return tag_name.lower().replace(r'[^a-zA-Z0-9\s]', '')

def filter_books_by_tag(book_tags, tags, selected_tag):
    selected_tag_normalized = normalize_tag_name(selected_tag)
    selected_tag_id = tags.loc[tags['tag_name_normalized'] == selected_tag_normalized, 'tag_id'].values
    if len(selected_tag_id) == 0:
        print("Invalid tag. Please select a valid tag.")
        return pd.DataFrame()
    filtered_books = book_tags.merge(tags, on='tag_id')
    filtered_books = filtered_books[filtered_books['tag_name_normalized'] == selected_tag_normalized]
    return filtered_books

def recommend_book(filtered_books, books):
    if filtered_books.empty:
        print("No books were found associated with the selected tag.")
        return ""
    random_book_id = np.random.choice(filtered_books['goodreads_book_id'])
    recommended_book_title = books.loc[books['book_id'] == random_book_id, 'title'].values[0]
    return recommended_book_title

tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/tags.csv')
book_tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/book_tags.csv')
books = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/books.csv')

tags['tag_name_normalized'] = tags['tag_name'].apply(normalize_tag_name)


@app.route('/')
def index():
    return 'Welcome to the Book Recommendation Web App!'

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        top_tags = tags['tag_name'].value_counts().head(10).index.tolist()
        print("Top tags:", top_tags)
        return render_template('recommend_form.html', top_tags=top_tags)
    elif request.method == 'POST':
        selected_tag = request.form['tag']
        print("Selected tag:", selected_tag)
        filtered_books = filter_books_by_tag(book_tags, tags, selected_tag)
        recommended_book = recommend_book(filtered_books, books)
        print("Recommended book:", recommended_book)
        return render_template('result.html', recommended_book=recommended_book)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
