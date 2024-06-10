import pandas as pd
import os
import pickle
dirs = ['journalists', 'ngos', 'politicians']

# create different dataframes for each user type
df_j = pd.DataFrame()
df_n = pd.DataFrame()
df_p = pd.DataFrame()

breaking = False
for dir in dirs:
    for file in os.listdir(dir):
        if file.endswith('.pkl'):
            if file in os.path.join(dir, 'with_cuts'):
                with open(os.path.join(dir, 'with_cuts', file), 'rb') as f:
                    posts = pickle.load(f)
            else:
                with open(os.path.join(dir, file), 'rb') as f:
                    posts = pickle.load(f)
            breaking = True
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
            if dir == 'journalists':
                df_j = pd.concat([df_j, df], axis = 0)
            elif dir == 'ngos':
                df_n = pd.concat([df_n, df], axis = 0)
            else:
                df_p = pd.concat([df_p, df], axis = 0)

# save
df_j.to_csv(r'data\facebook_posts_j.csv', index = False)
df_n.to_csv(r'data\facebook_posts_n.csv', index = False)
df_p.to_csv(r'data\facebook_posts_p.csv', index = False)