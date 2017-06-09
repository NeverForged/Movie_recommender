import logging
import numpy as np
import pandas as pd
import pyspark as ps
from pyspark.ml.recommendation import ALS


class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        # ...
        # First Crap Model
        self.spark = (ps.sql.SparkSession.builder
             .master('local[4]')
             .appName('Recommender')
             .getOrCreate())

        self.sc = self.spark.sparkContext

        self.als = ALS(itemCol='movie',
                  userCol='user',
                  ratingCol='rating',
                  nonnegative=True,
                  regParam=0.1,
                  rank=10)



    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        # ...
        # Try in case we get a spark df and not the pandas df you promised us
        spark_ratings = self.df_to_spark(ratings)

        # actual Fit call
        self.recommender = self.als.fit(spark_ratings)

        self.logger.debug("finishing fit")
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        # requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        trans_requests = self.recommender.transform(self.df_to_spark(requests))

        self.logger.debug("finishing predict")
        trans_requests = trans_requests.toPandas()
        trans_requests = trans_requests.drop('actualrating', inplace=False, axis=1)
        return trans_requests

    def df_to_spark(self, df):
        '''
        Converts pandas to spark...and checks to make sure it needs to
        '''
        try:
            return self.spark.createDataFrame(df)
        except Exception as e:
            print e
            return df

if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
