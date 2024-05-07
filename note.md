import pandas as pd

tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/tags.csv')
book_tags = pd.read_csv('/Users/siyaksares/Developer/GitHub/ReadersRadarProject/book_tags.csv')

tags['tag_id'] = pd.to_numeric(tags['tag_id'], errors='coerce')
book_tags['tag_id'] = pd.to_numeric(book_tags['tag_id'], errors='coerce')

tags.dropna(subset=['tag_id'], inplace=True)
book_tags.dropna(subset=['tag_id'], inplace=True)

top_tags = book_tags.groupby('tag_id').count().sort_values(by='count', ascending=False).head(10)
top_tag_names = tags[tags['tag_id'].isin(top_tags.index)]['tag_name']

print("Top 10 Most Used Tags:")
print(top_tag_names)

'''
selected_tag = input("Please select a tag: ")  # Strip whitespace from the input
print("Selected tag:", selected_tag)

if not selected_tag:
    print("No tag selected. Please enter a tag.")
elif selected_tag not in top_tag_names.values:
    print("Invalid label. Please select a tag from the top 10 most used tags.")
else:
    selected_tag_id = tags[tags['tag_name'] == selected_tag]['tag_id'].values[0]
    filtered_books = book_tags[book_tags['tag_id'] == selected_tag_id]
    if not filtered_books.empty:
        random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
        books = pd.read_csv('books.csv')
        selected_book = books[books['book_id'] == random_book_id][['title']]
        print("Selected Book:")
        print(selected_book)
    else:
        print("No books were found associated with the selected tag.")'''

'''
selected_tag = input("Please select a tag: ")  # Strip whitespace from the input
print("Selected tag:", selected_tag)

if not selected_tag:
    print("No tag selected. Please enter a tag.")
elif selected_tag not in top_tag_names.values:
    print("Invalid label. Please select a tag from the top 10 most used tags.")
else:
    selected_tag_id = tags[tags['tag_name'] == selected_tag]['tag_id'].values[0]
    filtered_books = book_tags[book_tags['tag_id'] == selected_tag_id]
    if not filtered_books.empty:
        random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
        books = pd.read_csv('books.csv')
        selected_book = books[books['book_id'] == random_book_id][['title']]
        print("Selected Book:")
        print(selected_book)
    else:
        print("No books were found associated with the selected tag.")
'''

'''
selected_tag_lower = selected_tag.lower()  # Convert selected tag to lowercase

if not selected_tag_lower:
    print("No tag selected. Please enter a tag.")
elif selected_tag_lower not in top_tag_names.str.lower().values:  # Convert tag names to lowercase for comparison
    print("Invalid label. Please select a tag from the top 10 most used tags.")
else:
    filtered_books = book_tags[book_tags['tag_name'].str.lower().str.contains(selected_tag_lower)]
    if not filtered_books.empty:
        random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
        books = pd.read_csv('books.csv')
        selected_book = books[books['book_id'] == random_book_id][['title']]
        print("Selected Book:")
        print(selected_book)
    else:
        print("No books were found associated with the selected tag.")'''

'''
selected_tag = input("Please select a tag: ")  # Strip whitespace from the input
print("Selected tag:", selected_tag)

if not selected_tag:
    print("No tag selected. Please enter a tag.")
elif selected_tag_lower not in top_tag_names.str.lower().values:  # Convert tag names to lowercase for comparison
    print("Invalid label. Please select a tag from the top 10 most used tags.")
else:
    selected_tag_lower = selected_tag.lower()  # Convert selected tag to lowercase
    selected_tag_id = tags[tags['tag_name'].str.lower() == selected_tag_lower]['tag_id'].values[0]
    filtered_books = book_tags[book_tags['tag_id'] == selected_tag_id]
    if not filtered_books.empty:
        random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
        books = pd.read_csv('books.csv')
        selected_book = books[books['book_id'] == random_book_id][['title']]
        print("Selected Book:")
        print(selected_book)
    else:
        print("No books were found associated with the selected tag.")'''

'''
selected_tag = input("Please select a tag: ")  # Strip whitespace from the input
print("Selected tag:", selected_tag)

if not selected_tag:
    print("No tag selected. Please enter a tag.")
elif selected_tag_lower not in top_tag_names.str.lower().values:  # Convert tag names to lowercase for comparison
    print("Invalid label. Please select a tag from the top 10 most used tags.")
else:
    selected_tag_lower = selected_tag.lower()  # Convert selected tag to lowercase
    selected_tag_id_series = tags[tags['tag_name'].str.lower() == selected_tag_lower]['tag_id']
    if not selected_tag_id_series.empty:
        selected_tag_id = selected_tag_id_series.values[0]
        filtered_books = book_tags[book_tags['tag_id'] == selected_tag_id]
        if not filtered_books.empty:
            random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
            books = pd.read_csv('books.csv')
            selected_book = books[books['book_id'] == random_book_id][['title']]
            print("Selected Book:")
            print(selected_book)
        else:
            print("No books were found associated with the selected tag.")
    else:
        print("Invalid tag. Please select a valid tag.")'''

selected_tag = input("Please select a tag: ").strip()  # Strip whitespace from the input
print("Selected tag:", selected_tag)

if not selected_tag:
    print("No tag selected. Please enter a tag.")
else:
    selected_tag_lower = selected_tag.lower()  # Convert selected tag to lowercase
    top_tag_names_lower = top_tag_names.str.lower()  # Convert top tag names to lowercase for comparison
    if selected_tag_lower not in top_tag_names_lower.values:
        print("Invalid label. Please select a tag from the top 10 most used tags.")
    else:
        selected_tag_id_series = tags[tags['tag_name'].str.lower() == selected_tag_lower]['tag_id']
        if not selected_tag_id_series.empty:
            selected_tag_id = selected_tag_id_series.values[0]
            filtered_books = book_tags[book_tags['tag_id'] == selected_tag_id]
            if not filtered_books.empty:
                random_book_id = filtered_books.sample(1)['goodreads_book_id'].values[0]
                books = pd.read_csv('books.csv')
                selected_book = books[books['book_id'] == random_book_id][['title']]
                print("Selected Book:")
                print(selected_book)
            else:
                print("No books were found associated with the selected tag.")
        else:
            print("Invalid tag. Please select a valid tag.")

