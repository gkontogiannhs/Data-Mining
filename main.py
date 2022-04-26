import pandas as pd
from matplotlib import pyplot
import os
import glob

def get_files(fn):
    home = os.path.expanduser('~')
    path = f'{home}/Downloads/project_mining_2022/dataset/'
    # current_path = os.getcwd()
    # path = f'{current_path}/dateset/'
    files = glob.glob(path + fn + '/*.csv')
    return files


def to_date(char):
    return char[:4] + '-' + char[4:6] + '-' + char[6:8]

# for each file create a pandas df
# concatenate them all together
# change index to datetime
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

def main():

    demand_files = get_files('demand')
    source_files = get_files('sources')
    
    filenames = [file[-12: -4] for file in demand_files]

    demands = parse_file(demand_files, filenames)

    sources = parse_file(source_files, filenames)

    sources['Natural gas'].fillna(sources['Natural Gas'], inplace=True)

    sources['Large hydro'].fillna(sources['Large Hydro'], inplace=True)

    sources.drop(columns=['Natural Gas', 'Large Hydro'], inplace=True)

    # plot all data. Measurements every 5 mins
    demands['Current demand'].plot()
    pyplot.show()
    
    # plot every hour
    demands.loc[:, 'Current demand'][0::12].plot()
    pyplot.show()

    # plot per day
    demands.loc[:, 'Current demand'][0::288].plot()
    pyplot.show()

    # plot all data. Measurements every 5 mins
    sources.plot()
    pyplot.show()

    # plot every hour
    sources[0::12].plot()
    pyplot.show()

    # plot per day
    sources[0::288].plot()
    pyplot.show()

main()
