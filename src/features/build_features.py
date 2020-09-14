
import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal


def get_doubling_time_via_regression(in_array: pd.Series) -> float:
    """
    Use a linear regression to approximate the doubling rate

    Args:
        in_array: input array on which regression is applied

    Returns:
        Doubling rate calculated on the input array
    """

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


def savgol_filter(df_input: pd.DataFrame, filter_on: str= 'confirmed', window: int=5, degree: int=1) -> pd.DataFrame:

    """

    Args:
        df_input: pandas DataFrame on which the function is applied
        filter_on: name of the column used
        window:
        degree:

    Returns:
        Input DataFrame with the result added as a new column
    """
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)

        parameters:
        ----------
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result

        Returns:
        ----------
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''


    df_result=df_input

    filter_in=df_input[filter_on].fillna(0) # attention with the neutral element here

    result=signal.savgol_filter(np.array(filter_in),
                           window, # window size used for filtering
                           degree)
    df_result[str(filter_on + '_filtered')]=result
    return df_result

def rolling_reg(df_input,col='confirmed'):
    ''' Rolling Regression to approximate the doubling time'

        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str
            defines the used column
        Returns:
        ----------
        result: pd.DataFrame
    '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)



    return result




def calc_filtered_data(df_input: pd.DataFrame, filter_on: str='confirmed') -> pd.DataFrame:
    """
    Calculate savgol filter and return merged data frame

    Args:
        df_input: pandas DataFrame on which the function is applied
        filter_on: name of the column used

    Returns:
        Input DataFrame with the result added as a new column
    """


    must_contain= {'state', 'country', filter_on}
    assert must_contain.issubset(set(df_input.columns)), ' Error in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country'])\
        .apply(savgol_filter).reset_index()


    df_output=pd.merge(df_output,pd_filtered_result[['index',str(filter_on+'_filtered')]],on=['index'],how='left')

    return df_output.copy()





def calc_doubling_rate(df_input: pd.DataFrame, filter_on: str='confirmed') -> pd.DataFrame:
    """
    Calculate approximated doubling rate and return merged data frame

    Args:
        df_input: pandas DataFrame on which the function is applied
        filter_on: name of the column used

    Returns:
        Input DataFrame with the result added as a new column
    """

    must_contain= {'state', 'country', filter_on}
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_dr_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_dr_result=pd_dr_result.rename(columns={filter_on: filter_on+'_DR',
                             'level_2': 'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_dr_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])


    return df_output


if __name__ == '__main__':
    test_data_reg=np.array([2,4,6])
    result=get_doubling_time_via_regression(test_data_reg)
    print('the test slope is: '+str(result))

    pd_JH_data=pd.read_csv('../../data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).reset_index().copy()

    #test_structure=pd_JH_data[((pd_JH_data['country']=='US')|
    #                  (pd_JH_data['country']=='Germany'))]

    pd_result_large=calc_filtered_data(pd_JH_data)
    pd_result_large=calc_doubling_rate(pd_result_large)
    pd_result_large=calc_doubling_rate(pd_result_large, 'confirmed_filtered')


    mask= pd_result_large['confirmed'] > 100
    pd_result_large['confirmed_filtered_DR']=pd_result_large['confirmed_filtered_DR'].where(mask, other=np.NaN)
    pd_result_large.to_csv('../../data/processed/COVID_final_set.csv', sep=';', index=False)
    print(pd_result_large[pd_result_large['country'] == 'Germany'].tail())
