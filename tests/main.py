import pandas as pd
from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = [7, 5]
import seaborn as sns; sns.set_theme()
import os
import glob
from statsmodels.tsa.stattools import adfuller, kpss


def get_files(fn):
    home = os.path.expanduser('~')
    path = f'{home}/Downloads/project_mining_2022/dataset/'
    files = glob.glob(path + fn + '/*.csv')
    return files


def to_date(char):
    return char[:4] + '-' + char[4:6] + '-' + char[6:8]


def parse_file(files, filenames):
    list_of_pandas = []
    for i, f in enumerate(files):
        # if file empty
        if os.stat(f).st_size != 0:
            # create df
            temp = pd.read_csv(f, header=0, nrows=288, skip_blank_lines=False)
            # create date column
            temp['Date'] = [to_date(filenames[i]) for _ in range(288)]
            # append df 
            list_of_pandas += [temp]
    # cast all together
    df = pd.concat(list_of_pandas, ignore_index=True)
    df.index = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    df.drop(columns=['Date', 'Time'], inplace=True)
    return df


# get all files from each set
demand_files = get_files('demand')
source_files = get_files('sources')

# list filenames to use on timestamps
filenames = [file[-12: -4] for file in demand_files]

# get data from files
demands = parse_file(demand_files, filenames)
sources = parse_file(source_files, filenames)

# move values and drop temp cols
sources['Natural gas'].fillna(sources['Natural Gas'], inplace=True)
sources['Large hydro'].fillna(sources['Large Hydro'], inplace=True)
sources.drop(columns=['Natural Gas', 'Large Hydro'], inplace=True)

# null values in demands are probably due to power cut
# so, linear interpolation will be good estimate
demands.interpolate(method='linear', axis=0, inplace=True)
sources.interpolate(method='linear', axis=0, inplace=True)

# demand graph 
demands['Current demand'].plot(xlabel='Date', ylabel='GWh', title = 'Power Demand Per 5 minutes for 3 years')
pyplot.xticks(rotation=30)
pyplot.show()

# Rolling mean and std for demand with a window of a day
demands_rolmean = demands['Current demand'].rolling(window=289).mean()
demands_rolstd = demands['Current demand'].rolling(window=289).std()
pyplot.title('Rolling Mean & Std')
pyplot.xlabel('Date')
pyplot.ylabel('GWh')
pyplot.plot(demands['Current demand'], color='blue', label='Original')
pyplot.plot(demands_rolmean, color='red', label='Rolling Mean')
pyplot.plot(demands_rolstd, color='green', label='Rolling Std')
pyplot.xticks(rotation=30)
pyplot.legend(loc='best')
pyplot.show()


# calc mean, std, min, max per year
means = demands.groupby(by=demands.index.year).mean()['Current demand']
stds = demands.groupby(by=demands.index.year).std()['Current demand']
mins = demands.groupby(by=demands.index.year).min()['Current demand']
maxs = demands.groupby(by=demands.index.year).max()['Current demand']

basic_stats_year = pd.concat([means, stds, mins, maxs], axis=1, keys=['Mean', 'Std', 'Min', 'Max'])
basic_stats_year.plot(xlabel='Year', ylabel='GWh', title='Basic Stats Per Year', marker='o')
pyplot.xticks(demands.index.year.unique().values)
pyplot.show()

# calc mean, std, min, max per month
means = demands.groupby(by=[demands.index.year, demands.index.month]).mean()['Current demand']
stds = demands.groupby(by=[demands.index.year, demands.index.month]).std()['Current demand']
mins = demands.groupby(by=[demands.index.year, demands.index.month]).min()['Current demand']
maxs = demands.groupby(by=[demands.index.year, demands.index.month]).max()['Current demand']

basic_stats_year_month = pd.concat([means, stds, mins, maxs], axis=1, keys=['Mean', 'Std', 'Min', 'Max'])
basic_stats_year_month.plot(xlabel='Year-Month', ylabel='GWh', title='Basic Stats Per Month', marker='o')
pyplot.show()

# Mean demand per hour
average_hour_per_year = demands.groupby(by=[demands.index.year, demands.index.hour]).mean()
pyplot.title('Mean Demand Per Hour')
pyplot.xlabel('Hour')
pyplot.ylabel('GWh')
pyplot.plot(average_hour_per_year[:24]['Current demand'].values, label='2019')
pyplot.plot(average_hour_per_year[24:48]['Current demand'].values, label='2020')
pyplot.plot(average_hour_per_year[48:]['Current demand'].values, label='2021')
pyplot.legend(loc='best')
pyplot.xticks(demands.index.hour.unique().values)
pyplot.show()

# Seasonality
average_month_per_year = demands.groupby(by=[demands.index.year, demands.index.month]).mean()
pyplot.title('Seasonality of Power Demand')
pyplot.xlabel('Month')
pyplot.ylabel('GWh')
pyplot.plot(demands.index.month.unique().values, average_month_per_year[:12]['Current demand'].values, label='2019')
pyplot.plot(demands.index.month.unique().values, average_month_per_year[12:24]['Current demand'].values, label='2020')
pyplot.plot(demands.index.month.unique().values, average_month_per_year[24:]['Current demand'].values, label='2021')
pyplot.xticks(demands.index.month.unique().values)
pyplot.legend(loc='best')
pyplot.show()


######################################## check for stationarity ##################################################

# ADF Test
result = adfuller(demands['Current demand'].values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(demands['Current demand'].values, regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# 5 mins interval supply plot
pyplot.rcParams['figure.figsize'] = [8, 6]
sources.plot(subplots=True)
pyplot.show()


# Mean power supply per day per source
sources_day = sources.groupby(by=[sources.index.year, sources.index.day]).mean()
sources_day.plot(xlabel='Year, Day', ylabel='GWh', title='Mean Power supply per day for 3 years')
pyplot.show()

# Mean power supply per hour per source
sources_hour = sources.groupby(by=[sources.index.year, sources.index.hour]).mean()
sources_hour.plot(xlabel='Year, Hour', ylabel='GWh', title='Mean Power supply per hour for 3 years')
pyplot.show()