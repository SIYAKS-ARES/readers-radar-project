from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np

# For selecting kernel on py files; press cmd + shift + P then choose "Select Interpreter"

app = Flask(__name__)

def normalize_tag_name(tag_name):
    if pd.isnull(tag_name):
        return ''
    return tag_name.lower().replace(r'[^a-zA-Z0-9\s]', '')

tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/tags.csv')
book_tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/book_tags.csv')
books = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/books.csv')

tags['tag_name_normalized'] = tags['tag_name'].apply(normalize_tag_name)

valid_tag_ids = tags['tag_id'].unique()
book_tags = book_tags[book_tags['tag_id'].isin(valid_tag_ids)]

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        top_tag_ids = book_tags['tag_id'].value_counts().head(10).index
        top_tag_names = tags[tags['tag_id'].isin(top_tag_ids)]['tag_name']
        print("Top tags:", top_tag_names)  # Corrected line
        return render_template('recommend_form.html', top_tags=top_tag_names)
    elif request.method == 'POST':
        selected_tag = request.form['tag']
        print("Selected tag:", selected_tag)
        filtered_books = filter_books_by_tag(book_tags, tags, selected_tag)
        if filtered_books.empty:
            flash('No books were found associated with the selected tag.', 'error')
        else:
            recommended_book = recommend_book(filtered_books, books)
            flash(f'The recommended book is: {recommended_book}', 'success')
        return render_template('result.html', recommended_book=recommended_book)

'''def recommend():
    if request.method == 'GET':
        top_tag_ids = book_tags['tag_id'].value_counts().head(10).index
        top_tag_names = tags[tags['tag_id'].isin(top_tag_ids)]['tag_name']
        print("Top tags:", top_tag_names)
        return render_template('recommend_form.html', top_tags=top_tag_names)
    elif request.method == 'POST':
        selected_tag = request.form['tag']
        print("Selected tag:", selected_tag)
        filtered_books = filter_books_by_tag(book_tags, tags, selected_tag)
        if filtered_books.empty:
            flash('No books were found associated with the selected tag.', 'error')
        else:
            recommended_book = recommend_book(filtered_books, books)
            flash(f'The recommended book is: {recommended_book}', 'success')
        return render_template('result.html', recommended_book=recommended_book)'''
'''def recommend():
    if request.method == 'GET':
        top_tag_ids = book_tags['tag_id'].value_counts().head(10).index
        top_tag_names = tags[tags['tag_id'].isin(top_tag_ids)]['tag_name']
        print("Top tags:", top_tag_names)
        return render_template('recommend_form.html', top_tags=top_tag_names)
    elif request.method == 'POST':
        selected_tag = request.form['tag']
        print("Selected tag:", selected_tag)
        filtered_books = filter_books_by_tag(book_tags, tags, selected_tag)
        recommended_book = recommend_book(filtered_books, books)
        print("Recommended book:", recommended_book)
        return render_template('result.html', recommended_book=recommended_book)'''

if __name__ == '__main__':
    app.run(debug=True, port=3000)
