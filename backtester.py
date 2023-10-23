# TODO: Need to add lead/lag for data

import json
import logging
import pandas as pd
from fredapi import Fred
from datetime import datetime, date

class BacktestParametersJson:
  BACKTEST_NAME = "backtest_name"
  START_DATE = "start_date"
  END_DATE = "end_date"
  FRED_API_KEY = "fred_api_key"
  DATAPATH_FILE = "datapath_file"
  FRED_DATASETS = "fred_datasets"
  class FredDatasetsJson:
    SERIES_ID = "series_id"
    LEAD_LAG = "lead_lag"
  MODEL_FILE = "model_file"
  MODEL_CLASS_NAME = "model_class_name"
  MODEL_PARAMS = "model_params"
  PRICES_DATAPATH_FILE = "prices_datapath_file"
  PRICES_DATASETS = "prices_datasets"


class FredData:
  def __init__(self, api_key_file: str, series_id: str, start_date: str, end_date: str, lead_lag: int = 0):
    self.api_key_file = api_key_file
    self.series_id = series_id
    self.start_date = start_date
    self.end_date = end_date
    self.lead_lag = lead_lag

    self.api_key = None
    self.df = None
    self.load_api_key()
    # self.load_data()

  def load_api_key(self):
    with open(self.api_key_file) as f:
      self.api_key = f.read()
    if self.api_key is None:
      logging.error("API key not found.")

  def load_data(self):
    fred = Fred(api_key=self.api_key)
    self.df = fred.get_series(self.series_id, observation_start=self.start_date, observation_end=self.end_date)
    # self.df.shift(-1*self.lead_lag) # Need to shift the other way # TODO

  def load_data_from_files(self, datafile: str) -> bool:
    try:
      data = pd.read_csv(datafile)
    except FileNotFoundError as e:
      return False
    # TODO: Bug fix here...
    # start_year, start_week = datetime.strptime(self.start_date, "%Y-%m-%d").isocalendar()[:2] # Make sure to use lower case m
    # end_year, end_week = datetime.strptime(self.end_date, "%Y-%m-%d").isocalendar()[:2]
    # self.df = data[(data["Year"] >= start_year) & (data["Year"] <= end_year) & (data["Week"] >= start_week) & (data["Week"] <= end_week)]
    self.df = data
    # self.df.shift(-1*self.lead_lag) # Need to shift the other way # TODO
    # TODO: ... to here
    return True


class Backtester:

  def __init__(self, parameter_file: str = "backtest_parameters.json"):
    self.parameter_file: str = parameter_file

    # Parameters from parameter file
    self.backtest_name: str = None
    self.start_date: str = None
    self.end_date: str = None
    self.fred_api_key: str = None
    self.datapath_file: str = None
    self.fred_datasets: FredData = None
    self.model_file: str = None
    self.model_class_name: str = None
    self.model_params: dict = None

    self.dfs: dict(str, pd.DataFrame) = {}
    self.merged_df: pd.DataFrame = None
    self.prices_dfs: dict(str, pd.DataFrame) = {}
    self.merged_prices_df: pd.DataFrame = None
    self.model = None
    self.load_parameters()
    self.load_data()
    self.load_model_class()
    self.merge_datasets()
    self.merge_prices()

  def load_parameters(self):
    if self.parameter_file is None:
      logging.info("No parameter file found.")
      return
    logging.info("Loading parameters from file: {}".format(self.parameter_file) + " ...")

    with open(self.parameter_file) as f:
      param = json.load(f)
      self.backtest_name = param[BacktestParametersJson.BACKTEST_NAME]
      logging.info("Backtest name: {}".format(self.backtest_name))

      def validate_date(date_text: str) -> date:
        try:
          d = date.fromisoformat(date_text)
          return d
        except ValueError:
          raise ValueError("Incorrect data format, should be YYYY-MM-DD and within correct date ranges")

      self.start_date = param[BacktestParametersJson.START_DATE]
      try:
        s = validate_date(self.start_date)
        logging.info("Start date: {}".format(self.start_date))
      except ValueError as e:
        logging.error("Invalid start date: {}".format(e))
        raise e

      self.end_date = param[BacktestParametersJson.END_DATE]
      try:
        e = validate_date(self.end_date)
        logging.info("End date: {}".format(self.end_date))
      except ValueError as e:
        logging.error("Invalid end date: {}".format(e))
        raise e

      if s > e:
        logging.error("Start date is after end date.")
        raise ValueError("Start date is after end date.")
      elif s == e:
        logging.warning("Start date is the same as end date.")

      self.datapath_file = param[BacktestParametersJson.DATAPATH_FILE]
      logging.info("Data path: {}".format(self.datapath_file))
      self.fred_api_key = param[BacktestParametersJson.FRED_API_KEY]
      if self.fred_api_key is not None:
        logging.info("FRED API key found.")
      self.fred_datasets = param[BacktestParametersJson.FRED_DATASETS]
      logging.info("FRED datasets: {}".format(self.fred_datasets))
      self.model_file = param[BacktestParametersJson.MODEL_FILE]
      logging.info("Model file: {}".format(self.model_file))
      self.model_class_name = param[BacktestParametersJson.MODEL_CLASS_NAME]
      logging.info("Model class name: {}".format(self.model_class_name))
      self.model_params = param[BacktestParametersJson.MODEL_PARAMS]
      logging.info("Model params: {}".format(self.model_params))
      self.prices_datapath_file = param[BacktestParametersJson.PRICES_DATAPATH_FILE]
      logging.info("Prices data path: {}".format(self.prices_datapath_file))
      self.prices_datasets = param[BacktestParametersJson.PRICES_DATASETS]
      logging.info("Prices datasets: {}".format(self.prices_datasets))


  def load_data(self):
    for dataset in self.fred_datasets:
      series_id = dataset[BacktestParametersJson.FredDatasetsJson.SERIES_ID]
      lead_lag = dataset[BacktestParametersJson.FredDatasetsJson.LEAD_LAG]
      fred_data = FredData(self.fred_api_key, series_id, self.start_date, self.end_date, lead_lag)
      # fred_data.load_data()
      success = fred_data.load_data_from_files(self.datapath_file + series_id + ".csv")
      if not success:
        logging.error("File could not be found for: {}".format(series_id))
        continue
      self.dfs[series_id] = fred_data.df

    for dataset in self.prices_datasets:
      df = pd.read_csv(self.prices_datapath_file + dataset)
      self.prices_dfs[dataset] = df


  def load_model_class(self):
    import importlib.util as import_util
    try:
      # spec = import_util.spec_from_file_location(self.model_class_name, self.model_file)
      spec = import_util.spec_from_file_location(self.model_class_name, self.model_file)
      model_module = import_util.module_from_spec(spec)
      spec.loader.exec_module(model_module)
      self.model = getattr(model_module, self.model_class_name)(**self.model_params)
      logging.info("Loaded model: {}".format(self.model_class_name))
    except ImportError as e:
      logging.error("Error importing model: {}".format(e))
      raise e


  def merge_datasets(self, start_date: str = None, end_date: str = None):
    # Make one big df according to the year and week dates
    if self.start_date is None or self.end_date is None:
      logging.error("Start or end date not found.")
      return

    # Generate all the dates between start and end date by week
    start_year, start_week = datetime.strptime(self.start_date, "%Y-%m-%d").isocalendar()[:2] # Make sure to use lower case m
    end_year, end_week = datetime.strptime(self.end_date, "%Y-%m-%d").isocalendar()[:2]
    dates_list = []
    for year in range(start_year, end_year + 1):
      for week in range(1, 53):
        if year == start_year and week < start_week:
          continue
        if year == end_year and week > end_week:
          continue
        dates_list.append((year, week))

    df = pd.DataFrame(dates_list, columns=["Year", "Week"])

    # Merge all the datasets
    for dataset in self.dfs:
      df = df.merge(self.dfs[dataset][["Year", "Week", "Value"]], on=["Year", "Week"], how='left')
      df.rename(columns={"Value": dataset}, inplace=True)

    # df.to_csv("test_x.csv", index=True) # TODO: Remove
    self.merged_df = df


  def merge_prices(self):
    # Make one big df according to the year and week dates
    if self.start_date is None or self.end_date is None:
      logging.error("Start or end date not found.")
      return

    # Generate all the dates between start and end date by week
    start_year, start_week = datetime.strptime(self.start_date, "%Y-%m-%d").isocalendar()[:2]
    end_year, end_week = datetime.strptime(self.end_date, "%Y-%m-%d").isocalendar()[:2]
    dates_list = []
    for year in range(start_year, end_year + 1):
      for week in range(1, 53):
        if year == start_year and week < start_week:
          continue
        if year == end_year and week > end_week:
          continue
        dates_list.append((year, week))

    df = pd.DataFrame(dates_list, columns=["Year", "Week"])

    # Merge all the datasets
    for dataset in self.prices_dfs:
      df = df.merge(self.prices_dfs[dataset][["Year", "Week", "Price"]], on=["Year", "Week"], how='left')
      df.rename(columns={"Price": dataset}, inplace=True)

    # df.to_csv("test_y.csv", index=True) # TODO: Remove
    self.merge_prices_df = df


  def run(self, output_file: str = "output.csv"):
    # Make one big df according to the year and week dates
    assert self.merged_df is not None
    assert len(self.merged_df.columns) == len(self.fred_datasets) + 2
    logging.info("Running inference on merged datasets...")

    output_df_list = []
    total_len = len(self.merged_df)
    for i, row in self.merged_df.iterrows():
      if i % int(0.1*total_len) == 0:
        logging.info("Inference progress: {:.2f}%".format(i/total_len*100))
      try:
        year = row["Year"]
        week = row["Week"]
        output = self.model.run_inference(row)
        error = self.model.calculate_error(row, self.merge_prices_df.iloc[i])
        output_df_list.append(pd.DataFrame({"year": year, "week": week, **output, **error}, index=[i]))
      except Exception as e:
        logging.error("Error running inference: {}".format(e))
        output_df = pd.concat(output_df_list, axis=1, ignore_index=True)
        output_df.to_csv(output_file, index=False)
        raise e

    output_df = pd.concat(output_df_list, ignore_index=True)
    output_df.to_csv(output_file, index=False)
    logging.info("Backtest complete. Output saved to: {}".format(output_file))


if __name__ == "__main__":
  logging.basicConfig(filename='log.txt', level=logging.INFO)

  open('log.txt', 'w').close() # Clear logging file
  logging.info("Running file: " + __file__)
  backtester = Backtester("backtest_parameters.json")

  backtester.run("output_file.csv")

  # datapath = "fred_data/"
  # if not os.path.exists(datapath):
  #   logging.warning("Data path does not exist.")
  #   raise Exception("Data path does not exist.")

  # for file in os.listdir(datapath):
  #   backtester.load_from_file(datapath + file)
  #   logging.info("Loaded file: {}".format(file))