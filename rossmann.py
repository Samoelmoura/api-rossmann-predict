import pickle
import pandas as pd
import inflection
import numpy as np
import datetime

# path_repo = r'D:\My Drive\Pessoal\Projetos\rossmann_sales_predict\src'
path_repo = ''

class Rossmann(object):
    def __init__(self):
        # constant_competition_distance
        self.constant_competition_distance = pickle.load(open(path_repo + r'features/constant_competition_distance.pkl', 'rb'))

        # month_map
        self.month_map = pickle.load(open(path_repo + r'features/month_map.pkl', 'rb'))

        # cols_filtering
        self.cols_filtering = pickle.load(open(path_repo + r'features/cols_filtering.pkl', 'rb'))

        # rs_competition_distance
        self.rs_competition_distance = pickle.load(open(path_repo + r'features/rs_competition_distance.pkl', 'rb'))

        # rs_competition_time_month
        self.rs_competition_time_month = pickle.load(open(path_repo + r'features/rs_competition_time_month.pkl', 'rb'))

        # mm_promo2_time_week
        self.mm_promo2_time_week = pickle.load(open(path_repo + r'features/mm_promo2_time_week.pkl', 'rb'))

        # map_state_holiday
        self.map_state_holiday = pickle.load(open(path_repo + r'features/map_state_holiday.pkl', 'rb'))

        # map_store_type
        self.map_store_type = pickle.load(open(path_repo + r'features/map_store_type.pkl', 'rb'))

        # map_assortment
        self.map_assortment = pickle.load(open(path_repo + r'features/map_assortment.pkl', 'rb'))

        # map_year
        self.map_year = pickle.load(open(path_repo + r'features/map_year.pkl', 'rb'))

        # day_of_week_cicle
        self.day_of_week_cicle = pickle.load(open(path_repo + r'features/day_of_week_cicle.pkl', 'rb'))

        # week_cicle
        self.week_cicle = pickle.load(open(path_repo + r'features/week_cicle.pkl', 'rb'))

        # year_quarters_cicle
        self.year_quarters_cicle = pickle.load(open(path_repo + r'features/year_quarters_cicle.pkl', 'rb'))

        # cols_feature_selection
        self.cols_feature_selection = pickle.load(open(path_repo + r'features/cols_feature_selection.pkl', 'rb'))

        # xgb_fit
        self.model = pickle.load(open(path_repo + r'models/xgb_fit.pkl', 'rb'))


    def data_cleaning(self, df_raw):
        df1 = df_raw.copy()
        snake_case = lambda x: inflection.underscore(x)

        cols_old = df1.columns.to_list()
        cols_new = list(map(snake_case, cols_old))

        # rename
        df1.columns = cols_new

        # changing date datatype
        df1['date'] = pd.to_datetime(df1['date'])

        # competition_distance
        df1.loc[df1['competition_distance'].isna(), 'competition_distance'] = self.constant_competition_distance

        # competition_open_since_month
        df1.loc[df1['competition_open_since_month'].isna(), 'competition_open_since_month'] = df1.loc[df1['competition_open_since_month'].isna(), 'date'].dt.month

        # competition_open_since_year
        df1.loc[df1['competition_open_since_year'].isna(), 'competition_open_since_year'] = df1.loc[df1['competition_open_since_year'].isna(), 'date'].dt.year

        # promo2_since_week
        df1.loc[df1['promo2_since_week'].isna(), 'promo2_since_week'] = df1.loc[df1['promo2_since_week'].isna(), 'date'].dt.week

        # promo2_since_year
        df1.loc[df1['promo2_since_year'].isna(), 'promo2_since_year'] = df1.loc[df1['promo2_since_year'].isna(), 'date'].dt.year

        # promo_interval
        df1['promo_interval'].fillna(0, inplace=True)
        df1['month_map'] = df1['date'].dt.month.map(self.month_map)
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
        df1.drop(['promo_interval', 'month_map'], axis=1, inplace=True)

        # change datatypes
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int')

        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int')

        # return df
        return df1
    

    def feature_engineering(self, df2):
        # week
        df2['week'] = df2['date'].dt.isocalendar().week

        # year
        df2['year'] = df2['date'].dt.year

        # quarters
        df2['year_quarters'] = df2['date'].dt.month.apply(lambda x: 1 if x <= 3 else (2 if x <= 6 else (3 if x <= 9 else 4)))

        # weekends
        df2['weekends'] = df2['date'].dt.day_name().apply(lambda x: 0 if x not in ['Friday', 'Saturday'] else 1)

        # last_week_of_month
        df2['last_week_of_month'] = df2['date'].dt.day.apply(lambda x: 0 if x <= 23 else 1)

        # competition_since
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(day=1, month=x['competition_open_since_month'], year=x['competition_open_since_year']), axis=1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days).astype('int')
        df2.drop(['competition_since', 'competition_open_since_month', 'competition_open_since_year'], axis=1, inplace=True)

        # promo_since
        df2['promo2_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo2_since'] = df2['promo2_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo2_time_week'] = ((df2['date'] - df2['promo2_since']) / 7).apply(lambda x: x.days).astype(int)
        df2.drop(['promo2_since', 'promo2_since_year', 'promo2_since_week'], axis=1, inplace=True)

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x=='a' else 'extra' if x=='b' else 'extended')

        # state_holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x=='a' else 'easter_holiday' if x=='b' else 'christmas' if x=='c' else 'regular_day')

        # day_of_week
        df2['day_of_week'] = df2['date'].dt.day_name()

        # return df
        return df2


    def data_filtering(self, df4):
        # columns filtering
        try:
            cols_filtering = ['open', 'customers']
            df4 = df4.drop(cols_filtering, axis=1)
        except:
            cols_filtering = ['open', 'id']
            df4 = df4.drop(cols_filtering, axis=1)

        # return df
        return df4


    def data_preparation(self, df5):
        # 'competition_distance'
        df5['competition_distance'] = self.rs_competition_distance.transform(df5[['competition_distance']].values)

        # 'competition_time_month'
        df5['competition_time_month'] = self.rs_competition_distance.transform(df5[['competition_time_month']].values)

        # 'promo2_time_week'
        df5['promo2_time_week'] = self.mm_promo2_time_week.transform(df5[['promo2_time_week']].values)

        # 'state_holiday'
        df5['state_holiday'] = df5['state_holiday'].map(self.map_state_holiday)

        # 'store_type'
        df5['store_type'] = df5['store_type'].map(self.map_store_type)

        # 'assortment'
        df5['assortment'] = df5['assortment'].map(self.map_assortment)

        # 'year'
        df5['year'] = df5['year'].map(self.map_year)

        # 'day_of_week'
        df5['day_of_week'] = df5['date'].dt.dayofweek
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/self.day_of_week_cicle)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/self.day_of_week_cicle)))
        df5.drop('day_of_week', axis=1, inplace=True)

        # 'week'
        df5['week_sin'] = df5['week'].apply(lambda x: np.sin(x*(2*np.pi/self.week_cicle)))
        df5['week_cos'] = df5['week'].apply(lambda x: np.cos(x*(2*np.pi/self.week_cicle)))
        df5.drop('week', axis=1, inplace=True)

        # 'year_quarters'
        df5['year_quarters_sin'] = df5['year_quarters'].apply(lambda x: np.sin(x*(2*np.pi/self.year_quarters_cicle)))
        df5['year_quarters_cos'] = df5['year_quarters'].apply(lambda x: np.cos(x*(2*np.pi/self.year_quarters_cicle)))
        df5.drop('year_quarters', axis=1, inplace=True)

        # feature selection
        df5.drop(self.cols_feature_selection, axis=1, inplace=True)

        # return df
        return df5

    def get_predictions(self, df7, df_raw):
        # removing especific features
        X = df7.drop(['date', 'store'], axis=1).values

        # predicting
        y_hat = self.model.predict(X)

        # dataframe
        df = df_raw.copy()
        df['predictions'] = np.expm1(y_hat)
        
        # return df
        return df
        