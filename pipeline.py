import luigi
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from collections import Counter


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class CountGenresAuthors(luigi.Task):
    counter_genre_path = 'counter_genre.pkl'
    counter_author_path = 'counter_author.pkl'

    task_complete = False

    def requires(self):
        pass

    # def output(self):
    #    return [luigi.LocalTarget(self.counter_genre_path),
    #            luigi.LocalTarget(self.counter_author_path)]

    def complete(self):
        return self.task_complete

    def count_genres_authors(self, data):
        counter_genre = Counter()
        counter_author = Counter()
        for i in tqdm(range(data.shape[0])):
            genres = str(data.iloc[i].genres).split(',')
            authors = str(data.iloc[i].authors).split(',')
            counter_genre.update(genres)
            counter_author.update(authors)
        return counter_genre, counter_author

    def run(self):
        items = pd.read_csv("items.csv")
        counter_genre, counter_author = self.count_genres_authors(items)
        with open(self.counter_genre_path, 'wb') as f1:
            pickle.dump(counter_genre, f1, pickle.HIGHEST_PROTOCOL)
        with open(self.counter_author_path, 'wb') as f2:
            pickle.dump(counter_author, f2, pickle.HIGHEST_PROTOCOL)
        self.task_complete = True


class MakeAuthorGenre(luigi.Task):
    filename = 'author_genre.csv'

    def requires(self):
        return CountGenresAuthors()

    def output(self):
        return luigi.LocalTarget(self.filename)

    def make_author_genre(self, counter_genre, counter_author, data):
        author_genre = pd.DataFrame(0, index=list(counter_author.keys()), columns=list(counter_genre.keys()))
        for i in tqdm(range(data.shape[0])):
            genres = str(data.iloc[i].genres).split(',')
            authors = str(data.iloc[i].authors).split(',')
            for genre in genres:
                for author in authors:
                    author_genre.loc[author, genre] += 1
        return author_genre

    def run(self):
        sample = pd.read_csv("sample_submission.csv")
        interactions = pd.read_csv("interactions.csv")
        items = pd.read_csv("items.csv")
        users = pd.read_csv("users.csv")

        interactions['start_date'] = pd.to_datetime(interactions['start_date'])
        users.fillna(0, inplace=True)
        user_ind = np.array([])

        for i in sample.Id:
            user_ind = np.append(user_ind, np.where(interactions.user_id == i)[0])

        merged = (interactions.iloc[user_ind].merge(items, left_on='item_id', right_on='id'))
        counter_genre = load_obj('counter_genre.pkl')
        counter_author = load_obj('counter_author.pkl')
        author_genre = self.make_author_genre(counter_genre, counter_author, merged)
        with open(self.filename, 'w') as f:
            author_genre.to_csv(f)


class MakeAuthorMostPop(luigi.Task):
    filename = 'author_most_pop.pkl'

    def requires(self):
        return MakeAuthorGenre()

    def output(self):
        return luigi.LocalTarget(self.filename)

    def make_author_most_pop(self, data, gender=None):
        author_most_pop = dict()
        counter_author = load_obj('counter_author.pkl')
        if gender is None:
            for author in tqdm(list(counter_author.keys())):
                author_most_pop[author] = data.loc[data.authors == author]. \
                    groupby('item_id')['item_id']. \
                    count().sort_values(ascending=False).index.to_list()
            return author_most_pop

        for author in tqdm(list(counter_author.keys())):
            author_most_pop[author] = data.loc[(data.sex == gender) & (data.authors == author)]. \
                groupby('item_id')['item_id']. \
                count().sort_values(ascending=False).index.to_list()

        return author_most_pop

    def run(self):
        interactions = pd.read_csv("interactions.csv")
        items = pd.read_csv("items.csv")
        int_item = (interactions.merge(items, left_on='item_id', right_on='id'))

        author_most_pop = self.make_author_most_pop(int_item)
        with open(self.filename, 'w') as f:
            pickle.dump(author_most_pop, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    luigi.run()
