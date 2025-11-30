#!/usr/bin/env python3

import streamlit as st
import numpy as np
import pandas as pd
import random
import altair as alt

color_list=['blue','red','yellow','green','black','white']
cam_list=['front', 'rear', 'left', 'right']
bin_type_list=['video-1','video-2']

def make_dataset():
  df = pd.DataFrame()

  df['no'] = range(1,501,1)
  df['color'] = random.choices(color_list,k=500)
  df['cam'] = random.choices(cam_list,k=500)
  df['bin_type'] = random.choices(bin_type_list,k=500)
  df['duration'] = np.random.randn(500)
  df['size'] = np.random.randn(500)

  return df

if __name__ == '__main__':
  st.markdown('markdown test :material/search:')
  st.success('success test :material/Search:')
  st.warning('success test :material/Open_With:')

  #selected_columns = st.multiselect('please select', df.columns[1:])
  #st.line_chart(df.set_index('x')[selected_columns])


  if 'dataset' not in st.session_state:
    st.session_state['dataset'] = make_dataset()
  df = st.session_state['dataset']
  #chart = (alt.Chart(df).mark_line().encode(x='no', y='duration'))
  #st.altair_chart(chart)

  st.line_chart(df, x='no', y=['duration','size'])


  l = df['color'].unique()
  l = ['all', 'yellow', 'black']
  sel = st.selectbox('camera', l)

  if sel != 'all':
    filtered_df = df[df['color'] == sel]
  else:
    filtered_df = df

  #name = st.text_input('please input')
  #if (name) and (name != 'all'):
  #  df = df[df['cam'].str.contains(name)]

  pill = st.pills('pill', color_list, default=color_list, selection_mode='multi')
  filtered_df = filtered_df[filtered_df['color'].isin(pill)]

  seg = st.segmented_control('seg', bin_type_list, default=bin_type_list, selection_mode='multi')
  filtered_df = filtered_df[filtered_df['bin_type'].isin(seg)]

  mul = st.multiselect('multiselect', cam_list, default=cam_list)
  filtered_df = filtered_df[filtered_df['cam'].isin(mul)]

  event = st.dataframe(filtered_df, hide_index=True, selection_mode=['single-row', 'single-column'])



