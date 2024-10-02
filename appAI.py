import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request

app = Flask(__name__)

# Load data with consistent data types
tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/tags.csv', dtype={'tag_id': int, 'tag_name': str})
book_tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/book_tags.csv', dtype={'goodreads_book_id': int, 'tag_id': int})
books = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/books.csv', dtype={'book_id': int, 'title': str, 'authors': str, 'original_publication_year': int})
ratings = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/ratings.csv', dtype={'user_id': int, 'book_id': int, 'rating': float})

# Normalize tag names
def normalize_tag_name(tag_name):
    if pd.isnull(tag_name):
        return ''
    return tag_name.lower().replace(r'[^a-zA-Z0-9\s]', '')

tags['tag_name_normalized'] = tags['tag_name'].apply(normalize_tag_name)

# Filter valid tags
valid_tag_ids = tags['tag_id'].unique()
book_tags = book_tags[book_tags['tag_id'].isin(valid_tag_ids)]

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_size)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = user_embedding * item_embedding
        x = self.fc(x)
        return x

# Determine the upper limit for user_id and book_id
max_user_id = ratings['user_id'].max()
max_book_id = ratings['book_id'].max()

print(f"Max user_id: {max_user_id}")
print(f"Max book_id: {max_book_id}")

# Initialize the model, loss function, and optimizer
model = RecommenderNet(max_user_id, max_book_id)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# önceki sonuclara bak ve karşılaştır.

# Prepare the data for training
user_ids = torch.tensor(ratings['user_id'].values, dtype=torch.long)
item_ids = torch.tensor(ratings['book_id'].values, dtype=torch.long)
ratings_values = torch.tensor(ratings['rating'].values, dtype=torch.float)

# Train the neural network
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(user_ids, item_ids).squeeze()
    loss = criterion(outputs, ratings_values)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        top_books = books.head(10)
        return render_template('recommend_form.html', books=top_books)
    elif request.method == 'POST':
        user_books = request.form.getlist('user_books')
        user_books = [int(book) for book in user_books]
        recommended_book = recommend_book(user_books, books, model)
        return render_template('result.html', recommended_book=recommended_book)

def recommend_book(user_books, books, model):
    if len(user_books) == 0:
        return {}

    # Dummy user ID within the range
    user_tensor = torch.tensor([1] * len(user_books), dtype=torch.long)
    item_tensor = torch.tensor(user_books, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor).squeeze()
    top_item_index = predictions.argmax().item()
    top_item = user_books[top_item_index]
    recommended_book = books.loc[books['book_id'] == top_item].to_dict(orient='records')[0]
    return recommended_book

if __name__ == '__main__':
    app.run(debug=True, port=3000)
