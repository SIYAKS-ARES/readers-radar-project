'''from flask import Flask, render_template, request
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


ratings = pd.read('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/ratings.csv')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

trainset = data.build_full_trainset()
svd_model = SVD()
svd_model.fit(trainset)

user_item_matrix = ratings.pivot(index='user_id', columns='book_id', values='rating')
user_item_matrix.fillna(0, inplace=True)


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = user_embedding * item_embedding
        x = self.fc(x)
        return x

num_users = ratings['user_id'].nunique()
num_items = ratings['book_id'].nunique()
model = RecommenderNet(num_users, num_items)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

user_ids = torch.tensor(ratings['user_id'].values, dtype=torch.long)
item_ids = torch.tensor(ratings['book_id'].values, dtype=torch.long)
ratings_values = torch.tensor(ratings['rating'].values, dtype=torch.float)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(user_ids, item_ids).squeeze()
    loss = criterion(outputs, ratings_values)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
'''