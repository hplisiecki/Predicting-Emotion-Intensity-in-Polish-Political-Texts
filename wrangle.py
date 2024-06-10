import os
import pickle
import pandas as pd
dirs = ['journalists', 'ngos', 'politicians']


post_dataframe = pd.DataFrame()
for dir in dirs:
    for file in os.listdir(dir):
        if file.endswith('.pkl'):
            if file in os.path.join(dir, 'with_cuts'):
                with open(os.path.join(dir, 'with_cuts', file), 'rb') as f:
                    posts = pickle.load(f)
            else:
                with open(os.path.join(dir, file), 'rb') as f:
                    posts = pickle.load(f)
            posts = [post for post in posts if 'text' in post.keys()]
            texts = [post['text'] for post in posts]
            ids = [post['post_id'] for post in posts]
            urls = [post['post_url'] for post in posts]
            times = [post['time'] for post in posts]
            likes = [post['likes'] for post in posts]
            comments = [post['comments'] for post in posts]
            shares = [post['shares'] for post in posts]
            user = [post['username'] for post in posts]
            user_type = [dir for post in posts]
            # concat
            df = pd.DataFrame({'text': texts
                                , 'id': ids
                                , 'url': urls
                                , 'time': times
                                , 'likes': likes
                                , 'comments': comments
                                , 'shares': shares
                                , 'user': user,
                                'user_type': user_type})
            post_dataframe = pd.concat([post_dataframe, df], axis = 0)


# save
post_dataframe.to_csv(r'data\facebook_posts.csv', index = False)


# sample 500 posts
sample = post_dataframe.sample(500)
# split into sentences on ., !, ?
sample['sentences'] = sample['text'].str.split(r'[.!?]')

# flatten
sample = sample.explode('sentences')
# replace \n
sample['sentences'] = sample['sentences'].apply(lambda x: x.replace('\n', ' '))
# strip whitespace
sample['sentences'] = sample['sentences'].apply(lambda x: x.strip())
# sample 500 longest
sample = sample.sort_values('sentences', ascending = False).head(500)
# reset index
sample = sample.reset_index(drop = True)

# save
sample.to_csv(r'data\facebook_posts_sample.csv', index = False)