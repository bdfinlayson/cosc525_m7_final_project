import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from constants.klass_mappings import klass_mappings
from constants.transliterations import transliterations
from keras.utils import to_categorical

class DataManager:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    X_val: pd.DataFrame
    y_val: pd.DataFrame
    X_train_sample: pd.DataFrame
    y_train_sample: pd.DataFrame

    def __init__(self):
        pass

    def load_data(self, use_preprocessed_data=False):
        root, dimensions = ('./data/preprocessing', '_24x24') if use_preprocessed_data else './data', ''
        self.X_test = pd.read_csv(f'{root}/test_x{dimensions}.csv')
        self.y_test = pd.read_csv('./data/test_y.csv')

        self.X_train = pd.read_csv(f'{root}/train_x{dimensions}.csv')
        self.y_train = pd.read_csv(f'./data/train_y.csv')

        self.generate_character_distribution_chart()

        # build validation dataset from training dataset
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        # take a small random sample for tuning purposes
        self.X_train_sample = self.X_train.sample(n=15000, random_state=6)
        self.y_train_sample = self.y_train.loc[self.X_train_sample.index]

        # ensure y datasets are same shape
        self.y_train = to_categorical(self.y_train, num_classes=len(klass_mappings))
        self.y_train_sample = to_categorical(self.y_train_sample, num_classes=len(klass_mappings))
        self.y_val = to_categorical(self.y_val, num_classes=len(klass_mappings))
        self.y_test = to_categorical(self.y_test, num_classes=len(klass_mappings))

    def training_data(self):
        return self.X_train, self.y_train

    def testing_data(self):
        return self.X_test, self.y_test

    def validation_data(self):
        return self.X_val, self.y_val

    def tuning_data(self):
        return self.X_train_sample, self.y_train_sample

    def generate_character_distribution_chart(self):
        plt.rcParams.update({'font.size': 22})
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor']='white'
        plt.rcParams['savefig.facecolor']='white'
        mappings_df = self.y_train.groupby('1')['1'].count().to_frame()
        mappings_df = mappings_df.rename(columns={'1': 'count'})
        mappings_df['klass'] = mappings_df.index
        mappings_df['character'] = mappings_df['klass'].apply(lambda x: klass_mappings[x])
        mappings_df['transliteration'] = mappings_df['klass'].apply(lambda x: transliterations[x])
        mappings_df['klass_transliteration'] = mappings_df['klass'].apply(lambda x: f'{klass_mappings[x]}  ({transliterations[x]}, {mappings_df["count"][x]}) ')
        mappings_df['proportion'] = mappings_df['klass'].apply(lambda x: mappings_df['count'][x] / mappings_df['count'][1])
        fig = mappings_df[['klass_transliteration', 'count']].plot(kind='bar', xlabel=f'Characters', ylabel='Count', title=f"Persian Character Distribution\nTotal Characters: {self.X_train.shape[0]}", x='klass_transliteration', figsize=(30,10)).get_figure()
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.35)
        fig.savefig('./images/character_distribution.png')

    def show_character_at_index(self, idx: int):
        plt.imshow(self.X_train.iloc[idx].values.reshape((64, 64)), cmap="gray_r")
