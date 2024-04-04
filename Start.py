#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df1= pd.read_csv("books.csv")
bt= pd.read_csv("book_tags.csv")
df3= pd.read_csv("ratings.csv")
df4= pd.read_csv("to_read.csv")
df5= pd.read_csv("tags.csv")
# %%
df1.head()
bt.head()
df3.head()
df4.head()
df5.head()
# %%
b =  df1.iloc[21]
b[0]

# %%
df5[df5["tag_name"] == "fiction"].value_counts()

# %%
fiction_tags = df5[df5['tag_name'].str.contains("fiction")]
fiction_tags["tag_name"] = "fiction"
fiction_tags.head()

# %%
df5[df5['tag_name'].str.contains("fiction")] = "Fiction"
df5[df5['tag_name'].str.contains("football")] = "Football"
df5[df5['tag_name'].str.contains("horror")] = "Horror"
df5[df5['tag_name'].str.contains("crime")] = "Crime"
df5[df5['tag_name'].str.contains("detective")] = "Detective"
df5[df5['tag_name'].str.contains("manga")] = "Manga"
df5[df5['tag_name'].str.contains("read")] = "To Read"
df5[df5['tag_name'].str.contains("american")] = "American"
df5[df5['tag_name'].str.contains("children")] = "Children"
df5[df5['tag_name'].str.contains("chinese")] = "Chinese"
df5[df5['tag_name'].str.contains("church")] = "Church"
df5[df5['tag_name'].str.contains("ancient")] = "Ancient History"
df5[df5['tag_name'].str.contains("alien")] = "Alien"
df5[df5['tag_name'].str.contains("apocalyptic")] = "Apocalyptic"
df5[df5['tag_name'].str.contains("greek")] = "Greek"
df5[df5['tag_name'].str.contains("romance")] = "Romance"
df5[df5['tag_name'].str.contains("thriller")] = "Thrilling Thrillers"

df5["tag_name"].value_counts().head(10)

# %%
df5[df5["tag_name"] == "Fiction"].head()
df5[df5["tag_name"] == "football"].head()

# %%
df5["tag_name"].value_counts().head().plot()
# %
# %%
'''import random

fiction_rows = df5[df5["tag_name"] == "Fiction"]

random_index = fiction_rows["tag_id"].random.random()
r_int = int(random_index)'''



# %%
for i in range(0,10000):
    int(df1["book_id"][i])

# %%
r_int = int(random_index)
# %%


# %%
print(book_column["title"].values[0])

# %%
if random_index in df1["work_ratings_count"]:
    print(df1.loc[random_index, "title"])
else:
    print("No book found with the given ID.")

# %%
df1["title"].head()
# %%
df1.index
# %%
print(random_index)
# %%
df1["ratings_1"]
# %%
df5["Fiction"].head()
# %%
'''import random

fiction_rows = df5[df5["tag_name"] == "Fiction"]

if not fiction_rows.empty:
    random_index = random.choice(fiction_rows["tag_id"].values)
    book_row = df1[df1["book_id"] == random_index]

    if not book_row.empty:
        print(book_row[10])
    else:
        print("No book found with the given ID.")
else:
    print("No fiction tags found.")'''


# %%
bt = bt.merge( b[[ 'goodreads_book_id', 'title']], on = 'goodreads_book_id' )