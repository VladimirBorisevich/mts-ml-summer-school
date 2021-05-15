import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import pickle
import logging
import os


def make_logger(name, format):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler_format = logging.Formatter(format)
    handler.setFormatter(handler_format)
    logger.addHandler(handler)
    return logger


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_data():
    path = 'data'
    sample = pd.read_csv(os.path.join(path, "sample_submission.csv"))
    interactions = pd.read_csv(os.path.join(path, "interactions.csv"))
    items = pd.read_csv(os.path.join(path, "items.csv"))
    users = pd.read_csv(os.path.join(path, "users.csv"))

    interactions['start_date'] = pd.to_datetime(interactions['start_date'])
    users.fillna(0, inplace=True)
    return sample, interactions, items, users


def merge_data(interactions, items, sample):
    user_ind = np.array([])
    for i in sample.Id:
        user_ind = np.append(user_ind, np.where(interactions.user_id == i)[0])
    merged = (interactions.iloc[user_ind].merge(items, left_on='item_id', right_on='id'))
    return merged


def separate_users(users, merged, n):
    user_list_less = []
    user_list_more = []
    for i, user in enumerate(users):
        if merged.loc[merged.user_id == user].shape[0] < n:
            user_list_less.append(user)
        else:
            user_list_more.append(user)
    return user_list_less, user_list_more


class Recommender:
    path = 'data'

    def __init__(self):
        self.author_genre, self.author_most_pop, \
            self.author_most_pop_male, self.author_most_pop_female = self._load_data()

    def _load_data(self):
        author_genre = pd.read_csv(os.path.join(self.path, 'author_genre.csv'), index_col=0)
        author_most_pop = load_obj(os.path.join(self.path, 'author_most_pop.pkl'))
        author_most_pop_male = load_obj(os.path.join(self.path, 'author_most_pop_male.pkl'))
        author_most_pop_female = load_obj(os.path.join(self.path, 'author_most_pop_female.pkl'))
        return author_genre, author_most_pop, author_most_pop_male, author_most_pop_female

    def add_item_to_pred(self, items, user_items, pred):
        for item in items:
            if (item not in pred) and (item not in user_items):
                pred.append(item)
                break
        return pred

    def find_users_attr(self, user, data):
        counter_genre = Counter()
        counter_author = Counter()
        row = data.loc[data.user_id == user].sort_values(by='start_date')

        for i in row.genres.to_list():
            counter_genre.update(i.split(','))

        for i in row.authors.to_list():
            counter_author.update(str(i).split(','))

        top1_genre = counter_genre.most_common(2)[0][0]
        top2_genre = counter_genre.most_common(2)[1][0]

        top1_author = counter_author.most_common(1)[0][0]
        if len(counter_author) == sum(counter_author.values()):
            top1_author = str(row.iloc[-1].authors).split(',')[0]

        last1_author = str(row.iloc[-1].authors).split(',')[0]

        if len(counter_author) == 1:
            last2_author = str(row.iloc[-1].authors).split(',')[0]
            return top1_genre, top2_genre, top1_author, last1_author, last2_author

        last2_author = str(row.iloc[-2].authors).split(',')[0]

        return top1_genre, top2_genre, top1_author, last1_author, last2_author

    def recommend_n_popular(self, n, user, user_activity, pred):
        user_items = user_activity.item_id.values
        gender = users.loc[users.user_id == user].sex.values
        prediction = []

        if len(gender) == 0:
            for item in all_top_15:
                if len(prediction) == n:
                    break
                if item not in user_items and item not in pred:
                    prediction.append(item)

        elif gender == 0:
            for item in male_top_15:
                if len(prediction) == n:
                    break
                if item not in user_items and item not in pred:
                    prediction.append(item)

        elif gender == 1:
            for item in female_top_15:
                if len(prediction) == n:
                    break
                if item not in user_items and item not in pred:
                    prediction.append(item)

        return pred + prediction

    def _recommend_with_low_activity(self, user):
        recommends = self.recommend_n_popular(10, user, merged.loc[merged.user_id == user], [])
        recs_in_line = " ".join([str(i) for i in recommends])
        sample.loc[sample.Id == user, 'Predicted'] = recs_in_line

    def _recommend_wth_high_activity(self, user):
        pred = []
        gender = users.loc[users.user_id == user].sex.values
        top1_genre, top2_genre, top1_author, last1_author, last2_author = self.find_users_attr(user, int_item)
        user_activity = merged.loc[merged.user_id == user]
        user_items = user_activity.item_id.to_list()

        if top1_genre == 'nan':
            top1_genre = str(top2_genre)
        if top2_genre == 'nan':
            top2_genre = str(top1_genre)

        if len(gender) == 0:
            most_pop = self.author_most_pop
        elif gender == 0:
            most_pop = self.author_most_pop_male
        elif gender == 1:
            most_pop = self.author_most_pop_female

        authors_items = most_pop[last1_author]
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        pred = self.add_item_to_pred(authors_items, user_items, pred)

        authors_items = most_pop[last2_author]
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        pred = self.add_item_to_pred(authors_items, user_items, pred)

        authors_items = most_pop[last1_author]
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        authors_items = most_pop[last2_author]
        pred = self.add_item_to_pred(authors_items, user_items, pred)
        pred = self.add_item_to_pred(authors_items, user_items, pred)

        # authors_items = most_pop[top1_author]
        # pred = add_item_to_pred(authors_items,user_items,pred)
        # pred = add_item_to_pred(authors_items,user_items,pred)

        # authors_items = most_pop[last1_author]
        # pred = add_item_to_pred(authors_items,user_items,pred)
        # pred = add_item_to_pred(authors_items,user_items,pred)

        # authors_items = most_pop[last2_author]
        # pred = add_item_to_pred(authors_items,user_items,pred)

        if len(pred) != 10:
            second_popular = user_activity.groupby('authors')['authors'].count(). \
                sort_values(ascending=False).index.to_list()[1].split(',')[0]
            authors_items = most_pop[second_popular]
            pred = self.add_item_to_pred(authors_items, user_items, pred)

        if len(pred) != 10:
            second_popular = user_activity.groupby('authors')['authors'].count(). \
                sort_values(ascending=False).index.to_list()[1].split(',')[0]
            authors_items = most_pop[second_popular]
            pred = self.add_item_to_pred(authors_items, user_items, pred)

        if len(pred) != 10:
            authors_items = most_pop[last1_author]
            pred = self.add_item_to_pred(authors_items, user_items, pred)

        if len(pred) != 10:
            authors_items = most_pop[last2_author]
            pred = self.add_item_to_pred(authors_items, user_items, pred)

        if len(pred) != 10:
            pred.append(all_top_15[0])

        answ = " ".join([str(i) for i in pred])
        sample.loc[sample.Id == user, 'Predicted'] = answ

    def recommend_to_user(self, users):
        logger.warning('separating data')
        user_list_less, user_list_more = separate_users(users, merged, 12)
        logger.warning('data is separated')
        logger.warning('making predictions')
        for user in tqdm(users):
            if user in user_list_less:
                self._recommend_with_low_activity(user)
            else:
                self._recommend_wth_high_activity(user)
        logger.warning('done')


logger = make_logger('recommender', '%(name)s - %(levelname)s - %(message)s - %(asctime)s')

logger.warning('loading data')
sample, interactions, items, users = load_data()
logger.warning('data is loaded\n')

logger.warning('merging data')
merged = merge_data(interactions, items, sample)
logger.warning('data is merged')

int_item = (interactions.merge(items, left_on='item_id', right_on='id'))
int_user = interactions.merge(users, on='user_id')
int_user = int_user.merge(items, left_on='item_id', right_on='id')
int_item.authors.fillna('nan', inplace=True)

male_top_15 = int_user.loc[int_user.start_date > '2019-12-01'].loc[int_user.sex == 0]. \
    groupby('item_id')['item_id'].count().sort_values(ascending=False).head(15).index.to_list()
female_top_15 = int_user.loc[int_user.start_date > '2019-12-01'].loc[int_user.sex == 1]. \
    groupby('item_id')['item_id'].count().sort_values(ascending=False).head(15).index.to_list()
all_top_15 = interactions.loc[interactions.start_date > '2019-12-01']. \
    groupby('item_id')['item_id'].count().sort_values(ascending=False).head(15).index.to_list()

model = Recommender()
model.recommend_to_user(sample.Id.values)
