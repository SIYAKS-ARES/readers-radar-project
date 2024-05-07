# %%
import pandas as pd

r = pd.read_csv( 'ratings.csv' )
tr = pd.read_csv( 'to_read.csv' )
b = pd.read_csv( 'books.csv' )
t = pd.read_csv( 'tags.csv' )
bt = pd.read_csv( 'book_tags.csv')

# %%
'''This cell imports the necessary libraries and reads five CSV files into
Pandas DataFrames: ratings.csv, to_read.csv, books.csv, tags.csv, and book_tags.csv.'''

# %%
r.head()

# %%
len(r)

# %%
'''This cell calculates and prints the total number of rows in the ratings.csv DataFrame using the len() function.'''

# %%
r.rating.hist( bins = 5 )

# %%
'''This cell generates a histogram of the 'rating' column in the ratings.csv DataFrame, dividing the ratings into five bins.'''

# %%
tr.head()

# %%
len(tr)

# %%
len(tr.book_id.unique())

# %%
'''This cell calculates and prints the total number of unique book IDs in the to_read.csv DataFrame.'''

# %%
len(tr.user_id.unique())

# %%
b.head()

# %%
len(b)

# %%
r.user_id.max()

# %%
'''This cell calculates and prints the maximum user ID in the ratings.csv DataFrame.'''

# %%
r.book_id.max()

# %%
assert( len( r.user_id.unique()) == r.user_id.max())
assert( len( r.book_id.unique()) == r.book_id.max())

# %%
'''This cell uses assertions to check if the number of unique user
IDs and book IDs in the ratings.csv DataFrame matches the maximum user ID and book ID, respectively.'''

# %%
reviews_per_book = r.groupby( 'book_id' ).book_id.apply( lambda x: len( x ))
reviews_per_book.describe()

# %%
'''This cell calculates and displays descriptive statistics for the number of reviews per book in the ratings.csv DataFrame.'''

# %%
reviews_per_book.sort_values().head( 10 )

# %%
'''This cell displays the ten books with the lowest number of reviews, sorted in ascending order.'''

# %%
reviews_per_user = r.groupby( 'user_id' ).user_id.apply( lambda x: len( x ))
reviews_per_user.describe()

# %%
'''This cell calculates and displays descriptive statistics for the number of reviews per user in the ratings.csv DataFrame.'''

# %%
reviews_per_user.sort_values().head( 10 )

# %%
import matplotlib.pyplot as plt

t.head()

# %%
'''This cell imports the matplotlib.pyplot library and displays the first few rows of the tags.csv DataFrame.'''

# %%
import matplotlib.pyplot as plt

r = pd.read_csv('ratings.csv')
tr = pd.read_csv('to_read.csv')
b = pd.read_csv('books.csv')
t = pd.read_csv('tags.csv')
bt = pd.read_csv('book_tags.csv')

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
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Added for random tag selection

# Read CSV files
r = pd.read_csv('ratings.csv')
tr = pd.read_csv('to_read.csv')
b = pd.read_csv('books.csv')
t = pd.read_csv('tags.csv')
bt = pd.read_csv('book_tags.csv')

# Exploratory Data Analysis (EDA)

# Print basic information about DataFrames (consider head() for first few rows)
print("Information about ratings DataFrame (r):")
print(r.info())

print("\nInformation about to_read DataFrame (tr):")
print(tr.info())

print("\nInformation about books DataFrame (b):")
print(b.info())

print("\nInformation about tags DataFrame (t):")
print(t.info())

print("\nInformation about book_tags DataFrame (bt):")
print(bt.info())

# Descriptive statistics for ratings, reviews per book/user, etc.
# (Consider using describe(), value_counts(), groupby() for these)
print("\nDescriptive statistics for ratings column in 'ratings.csv':")
print(r['rating'].describe())

print("\nNumber of reviews per book:")
reviews_per_book = r.groupby('book_id')['book_id'].apply(len)
print(reviews_per_book.describe())

print("\nNumber of reviews per user:")
reviews_per_user = r.groupby('user_id')['user_id'].apply(len)
print(reviews_per_user.describe())

# Data Cleaning and Merging

# Ensure non-negative counts (consider handling missing values as well)
bt.loc[bt['count'] < 0, 'count'] = 0

# Merge DataFrames based on common columns
bt = bt.merge(t[['tag_id', 'tag_name']], on='tag_id')
bt = bt.merge(b[['book_id', 'title']], left_on='goodreads_book_id', right_on='book_id')
bt.drop(columns=['goodreads_book_id'], inplace=True)

# Visualization

# Histograms, bar charts, horizontal bar charts (using Matplotlib)
plt.hist(r['rating'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution in ratings.csv')
plt.show()

avg_ratings_by_author = b.groupby('authors')['average_rating'].mean().sort_values(ascending=False).head(10)
avg_ratings_by_author.plot(kind='barh')
plt.xlabel('Average Rating')
plt.title('Average Ratings by Author (Top 10)')
plt.show()

user_review_counts = r.groupby('user_id')['book_id'].count().sort_values(ascending=False).head(10)
user_review_counts.plot(kind='barh')
plt.xlabel('Number of Reviews')
plt.title('Number of Reviews by User (Top 10)')
plt.show()

reviews_by_year = b.groupby('original_publication_year')['id'].count().sort_index()
plt.figure(figsize=(10, 6))  # Set figure size for better readability
plt.bar(reviews_by_year.index, reviews_by_year.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews by Book Publication Year')
plt.show()

top_tags = bt.groupby('tag_name').tag_name.count().sort_values(ascending=False).head(10)
top_tags.plot(kind='barh')
plt.xlabel('Number of Tags')
plt.title('Top 10 Most Popular Tags')
plt.show()

top_rated_books = b.sort_values(by='ratings_count', ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_rated_books['title'], top_rated_books['ratings_count'], color='skyblue')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Rated Books')
plt.gca().invert_
'''
# %%
'''I tried to gather the names of the tags under one roof here. I had quite a few trials, but as I mentioned before, there are 35k tags and it’s impossible to reorganize them all from scratch. This was one of the reasons why this project was very challenging.
Afterwards, I sorted them from the most to the least books and visualized it'''

# %%
len(t)

# %%
bt.head()

# %%
len(bt)

# %%
bt = bt.merge( t, on = 'tag_id' )

# %%
'''The code bt = bt.merge(t, on='tag_id') merges two DataFrames (bt and t) based on the common column 'tag_id',
combining information about book tags and general tag details into a single DataFrame (bt).'''

# %%
print(b.columns)

# %%
bt = bt.merge(t, on='tag_id')
bt = bt.merge(b[['book_id', 'title']], left_on='goodreads_book_id', right_on='book_id')
bt.loc[bt['count'] < 0, 'count'] = 0
bt.drop(columns=['goodreads_book_id'], inplace=True)
bt.sample(10, weights='count')

# %%
'''Merging Tags and Book Data:

Combine information about book tags, general tags, and book titles.
Data Cleaning:

Ensure non-negative counts and remove redundant columns.
Sampling Data:

Take a sample of 10 rows, considering count values for weighting.'''

# %%
bt['count'].describe()

# %%
bt.loc[ bt['count'] < 0, 'count'] = 0

# %%
bt.sample( 10, weights = 'count')

# %%
import matplotlib.pyplot as plt

plt.hist(r['rating'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.show()

# %%
'''Histogram Creation:

The code generates a histogram for the 'rating' column in DataFrame 'r'.
Bins and Edge Color:

The histogram is divided into 5 bins, and the edges are outlined in black.
Labels and Title:

X and Y-axis labels, as well as a title, are added for clarity.
Display:

The resulting histogram is displayed using Matplotlib.'''

# %%
avg_ratings_by_author = b.groupby('authors')['average_rating'].mean().sort_values(ascending=False).head(10)
avg_ratings_by_author.plot(kind='barh')
plt.xlabel('Average Rating')
plt.title('Average Rankings by Author')
plt.show()

# %%
'''Average Ratings by Author:

Calculate the mean of 'average_rating' for each author in DataFrame 'b'.
Top 10 Authors:

Select the top 10 authors with the highest average ratings.
Horizontal Bar Chart:

Plot a horizontal bar chart to visualize the average ratings for each author.
Labels and Title:

Add labels for the x-axis and y-axis, along with a title for the chart.
Display:

Display the bar chart using Matplotlib.'''

#%%
user_review_counts = r.groupby('user_id')['book_id'].count().sort_values(ascending=False).head(10)
user_review_counts.plot(kind='barh')
plt.xlabel('Number of Reviews')
plt.title('Number of Reviews by User')
plt.show()

#%%
'''User Review Counts:

Count the number of reviews each user ('user_id') has given in DataFrame 'r'.
Top 10 Users:

Identify the top 10 users with the most reviews.
Horizontal Bar Chart:

Create a horizontal bar chart to visualize the number of reviews for each user.
Labels and Title:

Include labels for the x-axis and y-axis, along with a title for the chart.
Display:

Show the horizontal bar chart using Matplotlib.'''

# %%
reviews_by_year = b.groupby('original_publication_year')['id'].count().sort_index()
reviews_by_year.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews by Book Year')
plt.show()

# %%
'''Reviews by Publication Year:

Count the number of reviews for each original publication year in DataFrame 'b'.
Bar Chart:

Create a bar chart to illustrate the distribution of reviews based on the original publication year.
Labels and Title:

Add labels for the x-axis and y-axis, along with a title for the chart.
Display:

Show the bar chart using Matplotlib.'''

# %%
##################And the reason for my various attempts at these visualizations 
##################is to better introduce the data frame and to show its complexity.

# %%
tag_counts = bt.groupby('tag_id')['count'].sum().sort_values(ascending=False)
top_tags = tag_counts.head(10)
top_tags.plot(kind='barh')
plt.xlabel('Number of Tags')
plt.title('Top 10 Most Popular Tags')
plt.show()

# %%
import matplotlib.pyplot as plt

top_rated_books = b.sort_values(by='ratings_count', ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_rated_books['title'], top_rated_books['ratings_count'], color='skyblue')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Rated Books')
plt.gca().invert_yaxis()
plt.show()

# %%
'''Top-Rated Books:

Select the top 10 books with the highest ratings count from DataFrame 'b'.
Horizontal Bar Chart:

Create a horizontal bar chart to visualize the ratings count for each top-rated book.
Figure Size and Color:

Set the figure size to 10x6 inches and use sky-blue color for the bars.
Labels and Title:

Add labels for the x-axis and y-axis, along with a title for the chart.
Invert Y-Axis:

Invert the y-axis to display the highest-rated book at the top of the chart.
Display:

Show the horizontal bar chart using Matplotlib.'''

# %%
bt = bt.merge(t[['tag_id', 'tag_name']], on='tag_id', suffixes=('_tag_app', '_tag'))
bt.drop(columns=['book_id'], inplace=True)
bt.loc[bt['count'] < 0, 'count'] = 0
tag_counts = bt.groupby('tag_name').tag_name.count().sort_values(ascending=False)
tag_counts.head(20)

# %%
'''
In this cell:

Tag information from DataFrame t is merged into the existing DataFrame bt based on 'tag_id'.
The 'book_id' column is dropped for simplification.
Negative counts in the 'count' column are set to 0.
Tag counts are calculated by grouping on 'tag_name'.
The top 20 tags by count are displayed'''

# %%
tag_counts = bt.groupby( 'tag_name' ).tag_name.count().sort_values( ascending = False )
tag_counts.head( 20 )

# %%
'''import matplotlib.pyplot as plt

reviews_by_year = b.groupby('original_publication_year')['id'].count().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(reviews_by_year.index, reviews_by_year.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews by Original Publication Year')
plt.show()
'''

# %%
import pandas as pd
import numpy as np

tags = pd.read_csv('tags.csv')
selected_tag = np.random.choice(tags['tag_name'], 1)[0]
print(f"Random Selected Label: {selected_tag}")

# %%
import pandas as pd

book_tags = pd.read_csv('book_tags.csv')
book_tags.drop('goodreads_book_id', axis=1, inplace=True)
print(book_tags.head())

# %%
'''
In this cell:

The 'book_tags.csv' file is read into a Pandas DataFrame named book_tags.
The 'goodreads_book_id' column is dropped for simplification.
The first few rows of the modified DataFrame are printed for inspection.'''

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
        print("Selected Book:")
        print(selected_book)
    else:
        print("No books were found associated with the selected tag.")

# %%
'''And lastly

In this cell:

Tags and book tags information are read into Pandas DataFrames (tags and book_tags).
The top 10 most used tags are identified based on their counts in the book_tags DataFrame.
User input is requested to select a tag.
If the selected tag is not valid, an error message is displayed.
If the tag is valid, a random book associated with the selected tag is displayed.
If no books are found for the selected tag, a corresponding message is printed.

I choose Horror tag and it's recommend me this book: 1978  Dead Ever After (Sookie Stackhouse, #13)'''


