# %%
import pandas as pd

r = pd.read_csv( 'ratings.csv' )
tr = pd.read_csv( 'to_read.csv' )
b = pd.read_csv( 'books.csv' )
t = pd.read_csv( 'tags.csv' )
bt = pd.read_csv( 'book_tags.csv')
# %%
r.head()
# %%
len(r)
# %%
r.rating.hist( bins = 5 )
# %%
tr.head()
# %%
len(tr)
# %%
len(tr.book_id.unique())
# %%
len(tr.user_id.unique())
# %%
b.head()
# %%
len(b)
# %%
r.user_id.max()
# %%
r.book_id.max()
# %%
assert( len( r.user_id.unique()) == r.user_id.max())
assert( len( r.book_id.unique()) == r.book_id.max())
# %%
reviews_per_book = r.groupby( 'book_id' ).book_id.apply( lambda x: len( x ))
reviews_per_book.describe()
# %%
reviews_per_book.sort_values().head( 10 )
# %% [markdown]
# %%
reviews_per_user = r.groupby( 'user_id' ).user_id.apply( lambda x: len( x ))
reviews_per_user.describe()
# %%
reviews_per_user.sort_values().head( 10 )
# %%
import matplotlib.pyplot as plt

t.head()
t[t['tag_name'].str.contains("fiction")] = "Fiction"
t[t['tag_name'].str.contains("football")] = "Football"
t[t['tag_name'].str.contains("horror")] = "Horror"
t[t['tag_name'].str.contains("crime")] = "Crime"
t[t['tag_name'].str.contains("detective")] = "Detective"
t[t['tag_name'].str.contains("manga")] = "Manga"
t[t['tag_name'].str.contains("read")] = "To Read"
t[t['tag_name'].str.contains("american")] = "American"
t[t['tag_name'].str.contains("children")] = "Children"
t[t['tag_name'].str.contains("chinese")] = "Chinese"
t[t['tag_name'].str.contains("church")] = "Church"
t[t['tag_name'].str.contains("ancient")] = "Ancient History"
t[t['tag_name'].str.contains("alien")] = "Alien"
t[t['tag_name'].str.contains("apocalyptic")] = "Apocalyptic"
t[t['tag_name'].str.contains("greek")] = "Greek"
t[t['tag_name'].str.contains("romance")] = "Romance"
t[t['tag_name'].str.contains("thriller")] = "Thrilling Thrillers"

selected_tags = ['Fiction', 'Football', 'Horror', 'Crime', 'Detective', 'Manga', 'To Read', 'American',
                 'Children', 'Chinese', 'Church', 'Ancient History', 'Alien', 'Apocalyptic', 'Greek',
                 'Romance', 'Thrilling Thrillers']
tag_frequencies = t[t['tag_name'].isin(selected_tags)]['tag_name'].value_counts()
plt.figure(figsize=(12, 6))
tag_frequencies.plot(kind='bar', color='skyblue')
plt.xlabel('Tag')
plt.ylabel('Frekans')
plt.title('Seçilen Tag\'lerin Frekansları')
plt.xticks(rotation=45)
plt.show()
# %%
len(t)
# %%
bt.head()
# %%
len(bt)
# %%
bt = bt.merge( t, on = 'tag_id' )
# %%
print(b.columns)
# %%
bt = bt.merge(t, on='tag_id')
bt = bt.merge(b[['book_id', 'title']], left_on='goodreads_book_id', right_on='book_id')
bt.loc[bt['count'] < 0, 'count'] = 0
bt.drop(columns=['goodreads_book_id'], inplace=True)
bt.sample(10, weights='count')
# %%
bt['count'].describe()
# %%
bt.loc[ bt['count'] < 0, 'count'] = 0
# %%
bt.sample( 10, weights = 'count')
#%%
import matplotlib.pyplot as plt

plt.hist(r['rating'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.show()
#%%
avg_ratings_by_author = b.groupby('authors')['average_rating'].mean().sort_values(ascending=False).head(10)
avg_ratings_by_author.plot(kind='barh')
plt.xlabel('Average Rating')
plt.title('Average Rankings by Author')
plt.show()
#%%
user_review_counts = r.groupby('user_id')['book_id'].count().sort_values(ascending=False).head(10)
user_review_counts.plot(kind='barh')
plt.xlabel('Number of Reviews')
plt.title('Number of Reviews by User')
plt.show()
#%%
reviews_by_year = b.groupby('original_publication_year')['id'].count().sort_index()
reviews_by_year.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews by Book Year')
plt.show()
#%%
import matplotlib.pyplot as plt

# Assuming the column name is 'tag_name_tag', adjust it if needed
tag_counts = bt.groupby('tag_name_tag').tag_name_tag.count().sort_values(ascending=False)
top_tags = tag_counts.head(10)

# Plot the bar chart
top_tags.plot(kind='barh')
plt.xlabel('Number of Tags')
plt.title('Top 10 Most Popular Tags')
plt.show()

#%%
import matplotlib.pyplot as plt

top_rated_books = b.sort_values(by='ratings_count', ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_rated_books['title'], top_rated_books['ratings_count'], color='skyblue')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Rated Books')
plt.gca().invert_yaxis()
plt.show()
# %%
bt = bt.merge(t[['tag_id', 'tag_name']], on='tag_id', suffixes=('_tag_app', '_tag'))
bt.drop(columns=['book_id'], inplace=True)
bt.loc[bt['count'] < 0, 'count'] = 0
tag_counts = bt.groupby('tag_name').tag_name.count().sort_values(ascending=False)
tag_counts.head(20)
# %%
tag_counts = bt.groupby( 'tag_name' ).tag_name.count().sort_values( ascending = False )
tag_counts.head( 20 )
#%%
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'b' containing the necessary data
reviews_by_year = b.groupby('original_publication_year')['id'].count().sort_index()

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(reviews_by_year.index, reviews_by_year.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews by Original Publication Year')
plt.show()

#%%
import pandas as pd
import numpy as np

tags = pd.read_csv('tags.csv')
selected_tag = np.random.choice(tags['tag_name'], 1)[0]
print(f"Random Selected Label: {selected_tag}")
#%%
import pandas as pd

book_tags = pd.read_csv('book_tags.csv')
book_tags.drop('goodreads_book_id', axis=1, inplace=True)
print(book_tags.head())
# %%
import pandas as pd

tags = pd.read_csv('tags.csv')
book_tags = pd.read_csv('book_tags.csv')
top_tags = book_tags.groupby('tag_id').count().sort_values(by='count', ascending=False).head(10)
top_tag_names = tags[tags['tag_id'].isin(top_tags.index)]['tag_name']
print("Top 10 Most Used Tags:")
print(top_tag_names)
selected_tag = input("Please select a tag: ")
selected_tag_id = tags[tags['tag_name'] == selected_tag]['tag_id'].values
if not selected_tag_id:
    print("Invalid label. Please select an existing tag.")
else:
    selected_tag_id = selected_tag_id[0]
    filtered_books = book_tags[book_tags['tag_id'] == selected_tag_id]
    if not filtered_books.empty:
        random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
        books = pd.read_csv('books.csv')
        selected_book = books[books['book_id'] == random_book_id][['title']]
        print("\nSelected Book:")
        print(selected_book)
    else:
        print("No books were found associated with the selected tag.")
# %%
