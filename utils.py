# import libraries
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def df_load_azdias():
    """
    This function is to load Udacity_AZDIAS_052018.csv file into dataframe

    INPUT

    OUTPUT
    Dataframe for data file
    """
    azdias = pd.DataFrame()
    # load in the data sets
    azdias = pd.read_csv('data/Udacity_AZDIAS_052018.csv', sep=';')
    return azdias


def df_load_customers():
    """
    This function is to load Udacity_CUSTOMERS_052018.csv file into dataframe

    INPUT

    OUTPUT
    Dataframe for data file
    """
    customers = pd.DataFrame()
    # load in the data sets
    customers = pd.read_csv('data/Udacity_CUSTOMERS_052018.csv', sep=';')
    # drop additional columns in customers
    customers = customers.drop(['PRODUCT_GROUP', 'CUSTOMER_GROUP', 'ONLINE_PURCHASE'], axis=1)
    return customers


def df_load_mailout_train():
    """
    This function is to load Udacity_MAILOUT_052018_TRAIN.csv file into dataframe

    INPUT

    OUTPUT
    Dataframe for data file
    """
    mailout_train = pd.DataFrame()
    # load in the data sets
    mailout_train = pd.read_csv('data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';')
    return mailout_train


def df_load_mailout_test():
    """
    This function is to load Udacity_MAILOUT_052018_TEST.csv file into dataframe

    INPUT

    OUTPUT
    Dataframe for data file
    """
    mailout_test = pd.DataFrame()
    # load in the data sets
    mailout_test = pd.read_csv('data/Udacity_MAILOUT_052018_TEST.csv', sep=';')
    return mailout_test


def clear_X_XX(df):
    """
    This function is to convert CAMEO_DEUG_2015, CAMEO_DEUG_2015 with X or XX to NaN

    INPUT
    df - dataframe

    OUTPUT

    """
    # Replace X and XX with nan
    df[['CAMEO_DEUG_2015', 'CAMEO_INTL_2015', 'CAMEO_DEU_2015']] = df[
        ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015', 'CAMEO_DEU_2015']].replace(['X', 'XX'], np.nan)


# Function to create a consolidated Attribute_Summary from both Attributes - Values file and Data Set file and save to file
def attribute_summary_file_generation(azdias):
    """
    This function is to prepare the attribute file for further processing:
        Combine the attributes from azdias data set and attributs value files to a new dataframe (attr_summary)
        Indicate if the data soucred from azdias (field: in_data), or from attributs value files (field:in_info)
        Output attr_summary dataframe to csv file for further processing
        Remove attribute not in data set
        Output attr_summary dataframe to csv file for further processing

    After check, a new field (new_value) will be added to keep the final values of unknown

    INPUT
    azdias - A dataframe of data set azdias

    OUTPUT
    Two data files:
        attr_summary.csv
        attr_data.csv

    """
    # Attribute data from AZDIAS - data set file, customers have the same with additional 3 columns
    attr_data = azdias.columns

    # data set columns data type
    attr_data_type = pd.DataFrame(azdias.dtypes, columns=['dtype'])
    attr_data_type.reset_index(inplace=True)
    attr_data_type = attr_data_type.rename(columns={'index': 'Attribute'})

    # Read attribute description file
    DIAS_attr_raw = pd.read_excel('data/DIAS Attributes - Values 2017.xlsx', sep=',', header=1)
    DIAS_attr_raw = pd.DataFrame(DIAS_attr_raw).drop(['Unnamed: 0'], axis=1)

    # Select attributes from attribute description file
    DIAS_attr = DIAS_attr_raw.loc[DIAS_attr_raw.Attribute.notnull()]

    # Attribute data from Attributes - Values file, combine with attributes from Data Set
    attr_info = DIAS_attr['Attribute'].unique()
    attr_all = np.unique(attr_data.tolist() + attr_info.tolist())
    attr_summary = pd.DataFrame(attr_all, columns=['Attribute'])

    # Consolidate attribute information and save to file
    attr_summary['in_info'] = attr_summary['Attribute'].apply(lambda s: s in attr_info)
    attr_summary['in_data'] = attr_summary['Attribute'].apply(lambda s: s in attr_data)
    attr_summary = pd.merge(attr_summary, attr_data_type, how="left", on=['Attribute'])
    attr_summary = pd.merge(attr_summary, DIAS_attr, how="left", on=['Attribute'])

    # Save attribute summary to file
    attr_summary.to_csv('attr_summary.csv')

    # Keep fields in data set for further processing
    attr_data = attr_summary[attr_summary.in_data == True][['Attribute', 'Description', 'dtype']]
    attr_data.to_csv('attr_data.csv')


def convert_EINGEFUEGT_AM(df):
    """
    This function is to convert EINGEFUEGT_AM to year

    INPUT
    df - dataframe

    OUTPUT

    """
    df['EINGEFUEGT_AM'] = pd.to_datetime(df['EINGEFUEGT_AM'], format='%Y%m%d %H:%M:%S')
    df['EINGEFUEGT_AM'] = df['EINGEFUEGT_AM'].dt.year


def unknown_to_nan_replace(df):
    """
    This function is to read data (unknown values) from attr_summary_1.xlsx, field new_value
    and to convert unknown value to NaN for each attributes

    INPUT
    df - dataframe

    OUTPUT

    """
    # read in attribute_summary file
    attr_summary = pd.read_excel('attr_summary_1.xlsx').drop(['id'], axis=1)

    # list all attributes in Data Set with the unknown data in new_values
    attr_unknown_df = attr_summary[attr_summary.in_data == True]
    attr_unknown_df = attr_unknown_df[attr_unknown_df.new_value.notnull()][['Attribute', 'new_value']]

    for col in attr_unknown_df.Attribute.to_list():
        # convert unknown values for field to list of int
        lst = attr_unknown_df[attr_unknown_df.Attribute == col].new_value.astype(str).to_list()[0].split(',')
        lst = [int(j.strip()) for j in lst]
        # convert unknown value to nan
        df[col] = df[col].apply(lambda x: np.nan if pd.isnull(x) else (np.nan if int(x) in lst else x))


def convert_OST_WEST_KZ(df):
    """
    This function is to convert string values to number

    INPUT
    df - dataframe

    OUTPUT

    """
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map({'O': 0, 'W': 1})


def convert_TITEL_KZ(df):
    """
    This function is to convert values in TITEL_KZ to person has title (1) or not (0)

    INPUT
    df - dataframe

    OUTPUT
    """
    df['TITEL_KZ'] = df['TITEL_KZ'].apply(lambda x: 0 if pd.isnull(x) else (0 if int(x) == 0 else 1))


def split_CAMEO_INTL_2015(df):
    """
    This function is to split values in CAMEO_INTL_2015 to
    CAMEO_INTL_2015_hh (10,20,30,40,50 mapped to Households)
    CAMEO_INTL_2015_fm (1,2,3,4,5 mapped to family)

    INPUT
    df - dataframe

    OUTPUT
    df - dataframe replaced by two new features

    """
    cameo_intl_flds = df['CAMEO_INTL_2015'].copy()
    cameo_intl_flds = cameo_intl_flds.replace([np.NaN], 0).astype(float)

    cameo_intl_df = pd.DataFrame()

    cameo_intl_df['CAMEO_INTL_2015_hh'] = cameo_intl_flds.apply(lambda x: math.floor(x / 10))
    cameo_intl_df['CAMEO_INTL_2015_fm'] = cameo_intl_flds.apply(lambda x: x % 10)

    df = pd.concat([df, cameo_intl_df], axis=1)
    df.drop(columns=['CAMEO_INTL_2015'], inplace=True)
    return df


# All attributes to be dropped
fld_drop_list = ['CAMEO_DEU_2015',
                 'D19_LETZTER_KAUF_BRANCHE',
                 'GFK_URLAUBERTYP',
                 'GEBAEUDETYP',
                 'ALTER_KIND4',
                 'ALTER_KIND3',
                 'D19_VERSI_ONLINE_DATUM',
                 'D19_TELKO_ONLINE_DATUM',
                 'D19_BANKEN_LOKAL',
                 'D19_BANKEN_OFFLINE_DATUM',
                 'ALTER_KIND2',
                 'D19_VERSI_OFFLINE_DATUM',
                 'D19_TELKO_ANZ_12',
                 'D19_DIGIT_SERV',
                 'D19_BIO_OEKO',
                 'D19_TIERARTIKEL',
                 'D19_NAHRUNGSERGAENZUNG',
                 'D19_GARTEN',
                 'D19_LEBENSMITTEL',
                 'D19_WEIN_FEINKOST',
                 'D19_BANKEN_ANZ_12',
                 'D19_ENERGIE',
                 'D19_TELKO_ANZ_24',
                 'D19_BANKEN_REST',
                 'D19_VERSI_ANZ_12',
                 'D19_TELKO_OFFLINE_DATUM',
                 'D19_BILDUNG',
                 'ALTER_KIND1',
                 'D19_BEKLEIDUNG_GEH',
                 'D19_RATGEBER',
                 'D19_SAMMELARTIKEL',
                 'D19_BANKEN_ANZ_24',
                 'D19_FREIZEIT',
                 'D19_BANKEN_GROSS',
                 'D19_VERSI_ANZ_24',
                 'D19_SCHUHE',
                 'D19_HANDWERK',
                 'D19_TELKO_REST',
                 'D19_DROGERIEARTIKEL',
                 'D19_KINDERARTIKEL',
                 'D19_KOSMETIK',
                 'D19_REISEN',
                 'D19_VERSAND_REST',
                 'D19_BANKEN_DIREKT',
                 'D19_BANKEN_ONLINE_DATUM',
                 'D19_TELKO_MOBILE',
                 'D19_HAUS_DEKO',
                 'D19_BEKLEIDUNG_REST',
                 'D19_BANKEN_DATUM',
                 'AGER_TYP',
                 'D19_TELKO_DATUM',
                 'D19_VERSI_DATUM',
                 'D19_VERSICHERUNGEN',
                 'D19_VERSAND_ANZ_12',
                 'D19_VERSAND_OFFLINE_DATUM',
                 'D19_TECHNIK',
                 'D19_BUCH_CD',
                 'D19_VOLLSORTIMENT',
                 'D19_GESAMT_ANZ_12',
                 'KK_KUNDENTYP',
                 'D19_VERSAND_ANZ_24',
                 'D19_GESAMT_OFFLINE_DATUM',
                 'D19_SONSTIGE',
                 'D19_GESAMT_ANZ_24',
                 'D19_VERSAND_ONLINE_DATUM',
                 'D19_GESAMT_ONLINE_DATUM',
                 'D19_VERSAND_DATUM',
                 'D19_GESAMT_DATUM']


def drop_columns_per_list(df, fld_drop_list):
    """
    This function is to drop the columns according to provided list of attributes

    INPUT
    df - dataframe
    fld_drop_list - provided list of attributes

    OUTPUT

    """
    for col in fld_drop_list:
        try:
            df.drop(columns=[col], inplace=True)
        except:
            pass


def plot_distribution(df, fld_list):
    """
    This function is to plot the distribution of missing data of attributes, sorted by percentage

    INPUT
    df - dataframe

    OUTPUT

    """
    for fld in fld_list:
        sns.set_style("darkgrid")
        sns.set_style("ticks")
        plt.figure(figsize=(14, 6))
        plt.title("Histogram of {0}".format(fld))
        sns.histplot(df[fld].values, kde=False)
        plt.ylabel('Frequency', fontsize=12)
        plt.show()


def remove_rows_missing_values(df, pct):
    """
    This function is to remove rows according to provided percentage of null value

    INPUT
    df - dataframe

    OUTPUT

    """
    df.drop(df[df.isnull().sum(axis=1) / df.shape[1] > pct / 100].index, axis=0, inplace=True)


def df_to_float(df):
    """
    This function is to change all columns to float

    INPUT
    df - dataframe

    OUTPUT
    df - dataframe processed
    """
    df = df.apply(pd.to_numeric, errors='ignore', downcast='float')
    return df


# Column list with outlier values to be moved
columns_with_outlier = ['ALTER_HH', 'ALTERSKATEGORIE_FEIN', 'ANZ_HAUSHALTE_AKTIV', 'ANZ_KINDER', 'ANZ_HH_TITEL',
                        'ANZ_PERSONEN', 'ANZ_STATISTISCHE_HAUSHALTE', 'GEBURTSJAHR', 'KBA13_ANZAHL_PKW',
                        'MIN_GEBAEUDEJAHR', 'VERDICHTUNGSRAUM']


def outlier_to_nan(df, columns_with_outlier):
    """
    This function is to convert outlier values to NaN

    INPUT
    df - dataframe
    columns_with_outlier - column list with outlier values to be moved

    OUTPUT

    """
    # Outlier matrix
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    for col in columns_with_outlier:
        # Remove rows beyond the IQR
        df[col] = np.where((df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3[col] + 1.5 * IQR[col])), np.nan,
                           df[col])


# Columns of categorical fields to be encoded by OneHotEncoder
onehot_list = ['CAMEO_DEUG_2015', 'CJT_GESAMTTYP', 'FINANZTYP', 'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN',
               'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'ZABEOTYP']


def one_hot_encoding(df, onehot_list):
    """
    This function is to encode categorical columns by OneHotEncoder

    INPUT
    df - dataframe
    onehot_list - column list of categorical fields to be encoded

    OUTPUT
    df - dataframe processed
    """
    df = pd.get_dummies(df, columns=onehot_list)
    return df


def df_fillna(df):
    """
    This function is to fill missing data with mode values

    INPUT
    df - dataframe

    OUTPUT

    """
    df.fillna(df.mode().iloc[0], inplace=True)


def feature_scaling(df):
    """
    This function is to standardize the column values

    INPUT
    df - dataframe

    OUTPUT
    df - dataframe processed
    """
    cols = df.columns
    scaler = StandardScaler()
    data_arr = scaler.fit_transform(df)
    df = pd.DataFrame(data_arr, columns=cols)
    return df
