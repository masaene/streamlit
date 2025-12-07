#!/usr/bin/env python3

import streamlit as st
import scipy
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd
import random
import altair as alt
from matplotlib import pyplot as plt
from glob import glob

type_mark_list = {
  'Grass': ':material/grass:',
  'Fire': ':material/mode_heat:',
  'Water': ':material/water_drop:',
  'Bug': ':material/bug_report:',
  'Poison': ':material/Skull:',
  'Electric': ':material/electric_bolt:',
  'Ice': ':material/ac_unit:',
}

def type1_pill_conv(sel):
  if sel in type_mark_list:
    mark = type_mark_list[sel]
    return f'{sel}{mark}'
  else:
    return sel

def make_stock_df():
  df = pd.read_csv('./csv/stock/TSE_8058_1D.csv')
  return df

def make_pokemon_df():
  df = pd.read_csv('./csv/pokemon/Pokemon.csv')
  return df

def make_measurement_df():
  l = sorted(glob('./csv/measurement_data/**/*.csv', recursive=True))
  tmp = list()
  df_list = list()
  for v in l:
    df = pd.read_csv(v)
    df_list.append(df)
    uni = df['Function']
    tmp.append(uni)
  func_list_df = pd.concat(tmp).drop_duplicates()

  d = dict()
  for v in func_list_df:
    d[v] = list()

  for df in df_list:
    for func_name in d.keys():
      if func_name in df['Function'].values:
        value = df.set_index('Function').loc[func_name]['Max']
        d[func_name].append(value)
      else:
        d[func_name].append(0)
  df = pd.DataFrame(d)
  return df

def tab_stock(df):
  date = df['time'].apply(lambda x:datetime.utcfromtimestamp(x))
  #df.insert(loc=1, column='date', value=date)
  weekday = df['time'].apply(lambda x:datetime.fromtimestamp(x).weekday())
  #df.insert(loc=2, column='weekday', value=weekday)
  #close_diff = df['close'].diff(periods=1)
  close_diff = df['close'].pct_change(periods=1)
  diff_in_day = df['open'] - df['close']

  df = pd.DataFrame({
    'time':date,
    'weekday':weekday,
    'close_diff':close_diff,
    'diff_in_day':diff_in_day,
    'open':df['open'],
    'close':df['close'],
    })

  #st.dataframe(df)
  #st.line_chart(df['close_diff'])

  """
  ret = df.groupby('weekday')['close_diff'].std()
  st.write(ret)
  ret = df.groupby('weekday')['diff_in_day'].std()
  st.write(ret)
  ret = df.groupby('weekday')['close_diff'].mean()
  st.write(ret)

  groupby_df = df[df['close_diff'] > 0.04]
  st.dataframe(groupby_df)
  """

  l = list()
  d = {'mon':0,'tue':1,'wed':2,'thu':3,'fri':4}
  for k,v in d.items():
    one_wd = df[df['weekday'] == v]['close_diff'].reset_index(drop=True)
    #one_wd.columns = [k]
    one_wd.rename(k,inplace=True)
    l.append(one_wd)

  df = pd.concat(l,axis=1)
  st.dataframe(df)
  st.line_chart(df, y=d.keys())

def tab_measurement(df):
  #selected = st.pills('function', df.columns, key='function_pills', selection_mode='multi')
  selected = st.multiselect('function', df.columns, key='function_pills')
  agg = st.toggle('aggregation?')
  if agg == True:
    mean = df.mean().reset_index()
    mean.columns = ['Function', 'mean']
    st.dataframe(mean)
    st.bar_chart(mean, x='Function', y='mean')

  st.dataframe(df)
  if selected:
    st.line_chart(df[selected])
    st.bar_chart(df[selected], stack=False)

def tab_graph(df):

  type_list = df['Type 1'].unique()
  pill = st.pills('Type 1', type_list, key='graph_pills', selection_mode='single', format_func=type1_pill_conv)
  grass_std = stats.tstd(df[df['Type 1'] == pill]['HP'],ddof=0)
  grass_mean = scipy.mean(df[df['Type 1'] == pill]['HP'])
  std = stats.tstd(df['HP'],ddof=0)
  mean = scipy.mean(df['HP'])

  st.write(std)

  x_min = mean - (4*std)
  x_max = mean + (4*std)

  x_data = np.arange(x_min, x_max, 0.1)
  #y_data = [stats.norm.pdf(i, loc=mean, scale=std) for i in range(x_min,x_max)]
  y_data = stats.norm.pdf(x_data, loc=mean, scale=std)
  std_mean = pd.DataFrame(
  {
  'x':x_data,
  'y':y_data
  },
  )

  ylim = max(y_data) * 1.1
  
  with st.container(border=True):
    fig, ax1 = plt.subplots()
    ax1.plot(std_mean['x'], std_mean['y'])
    ax2 = ax1.twinx()
    grass_pdf = stats.norm.pdf(grass_mean, loc=mean, scale=std)
    ax2.bar(grass_mean, grass_pdf, color='red')
    ax1.set_ylim([0,ylim])
    ax2.set_ylim([0,ylim])
    st.pyplot(fig)
    #st.line_chart(std_mean, x='x', y='y')
  

def tab_table(df):
 
  type_list = df['Type 1'].unique()

  filtered_df = df

  pill = st.pills('Type 1', type_list, key='type_pills', selection_mode='multi', format_func=type1_pill_conv)
  #pill = st.multiselect('Type 1', type_list)
  filtered_df = filtered_df[filtered_df['Type 1'].isin(pill)]

  mean_button = st.toggle('mean')
  if mean_button == True:
    filtered_df = filtered_df.groupby('Type 1')['HP'].mean().reset_index()

  st.dataframe(filtered_df, hide_index=True, selection_mode=['single-row', 'single-column'])
  if mean_button == True:
    st.bar_chart(filtered_df,x='Type 1',y='HP',color=['#87ceeb'])


if __name__ == '__main__':
  if 'pokemon_df' not in st.session_state:
    st.session_state['pokemon_df'] = make_pokemon_df()
  pokemon_df = st.session_state['pokemon_df']

  if 'measurement_df' not in st.session_state:
    st.session_state['measurement_df'] = make_measurement_df()
  measurement_df = st.session_state['measurement_df']

  if 'stock_df' not in st.session_state:
    st.session_state['stock_df'] = make_stock_df()
  stock_df = st.session_state['stock_df']

  st.set_page_config(layout='wide')

  tab_id_stock, tab_id_measurement, tab_id_pokemon_table, tab_id_pokemon_graph = st.tabs(['Stock', 'Measurement', 'Talbe(Pokemon)', 'Graph(Pokemon)'])

  with tab_id_stock:
    st.header('stock')
    tab_stock(stock_df)

  with tab_id_measurement:
    st.header('measuremet result')
    tab_measurement(measurement_df)

  with tab_id_pokemon_graph:
    st.header('table')
    tab_table(pokemon_df)

  with tab_id_pokemon_table:
    st.header('graph')
    tab_graph(pokemon_df)



