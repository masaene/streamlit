#!/usr/bin/env python3

import streamlit as st
import scipy
from scipy import stats
import numpy as np
import pandas as pd
import random
import altair as alt

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

def make_dataset():
  df = pd.read_csv('./csv/pokemon/Pokemon.csv')
  return df

def tab_graph(df):
  std = stats.tstd(df[df['Type 1'] == 'Grass']['HP'])
  mean = scipy.mean(df[df['Type 1'] == 'Grass']['HP'])
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
  
  with st.container(border=True):
    st.line_chart(std_mean, x='x', y='y')
  

def tab_table(df):
 
  type_list = df['Type 1'].unique()

  filtered_df = df

  pill = st.pills('Type 1', type_list, selection_mode='multi', format_func=type1_pill_conv)
  #pill = st.multiselect('Type 1', type_list)
  filtered_df = filtered_df[filtered_df['Type 1'].isin(pill)]

  mean_button = st.toggle('mean')
  if mean_button == True:
    filtered_df = filtered_df.groupby('Type 1')['HP'].mean().reset_index()

  st.dataframe(filtered_df, hide_index=True, selection_mode=['single-row', 'single-column'])
  if mean_button == True:
    st.bar_chart(filtered_df,x='Type 1',y='HP',color=['#87ceeb'])


if __name__ == '__main__':
  if 'dataset' not in st.session_state:
    st.session_state['dataset'] = make_dataset()
  df = st.session_state['dataset']

  tab1, tab2 = st.tabs(['table_tab', 'grach_tab'])

  with tab1:
    st.header('table')
    tab_table(df)

  with tab2:
    st.header('graph')
    tab_graph(df)



