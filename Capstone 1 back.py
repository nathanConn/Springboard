import pandas as pd
import numpy as np
import scipy as sp
import plotly as py
import plotly.graph_objs as go
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

#import training set
df = pd.read_csv('/Users/Nathan/Springboard/Capstone 1/dota2.training.data.csv')

#print column names
for col in df.columns:
    print(col)

#load in heroes JSON file and convert to df
j = json.load((open('/Users/Nathan/Springboard/Capstone 1/hero_names.json')))
heroes_df = pd.read_json('/Users/Nathan/Springboard/Capstone 1/hero_names.json')
heroes_df.head()

#transpose heroes df shape so that each row represents a single hero
heroes_transposed = heroes_df.transpose()
heroes_transposed.head()

#move hero id to column index 0
heroes = heroes_transposed.set_index('id')
heroes.head()

#placeholder definition as variations of this will be used often
all_heroes = df[['r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']]

#calculate radiant team hero frequency
r_heroes = df[['r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero']]
r_hero_freq = r_heroes.apply(pd.value_counts).sum(axis=1,skipna = True)
r_hero_freq.head()

#radiant team hero frequency with win
r_heroes_with_win = df[['r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero','radiant_win']]
r_heroes_win_only = r_heroes_with_win[r_heroes_with_win['radiant_win'] > 0]
r_heroes_win_only = r_heroes_win_only.drop(columns='radiant_win').apply(pd.value_counts).sum(axis=1,skipna = True)
r_heroes_win_only.head()

#radiant team hero win rate
r_heroes_win_per = r_heroes_win_only.divide(r_hero_freq)

#calculate dire team hero frequency
d_heroes = df[['d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']]
d_hero_freq = d_heroes.apply(pd.value_counts).sum(axis=1,skipna = True)
d_hero_freq.head()

#dire team hero frequency with win
d_heroes_with_win = df[['d1_hero','d2_hero','d3_hero','d4_hero','d5_hero','radiant_win']]
d_heroes_win_only = d_heroes_with_win[d_heroes_with_win['radiant_win'] == 0]
d_heroes_win_only = d_heroes_win_only.drop(columns='radiant_win').apply(pd.value_counts).sum(axis=1,skipna = True)
d_heroes_win_only.head()

#dire team hero win rate
d_heroes_win_per = d_heroes_win_only.divide(d_hero_freq)
d_heroes_win_per.head()
r_heroes_win_per.head()

#average radiant and dire team hero win rates for all hero win rate
all_heroes_win_per = ((d_heroes_win_per + r_heroes_win_per)/2)*100
all_heroes_win_per.sort_values(ascending=False).head()

#radiant team win rate
df['radiant_win'].mean()

#mean total gold by radiant teams with win
r_gold_win = df[['radiant_total_gold','radiant_win' ]]
radiant_mean_gold_if_win = r_gold_win[r_gold_win['radiant_win'] ==1].mean()
radiant_mean_gold_if_lose = r_gold_win[r_gold_win['radiant_win'] == 0].mean()

#mean total gold by dire teams with win
d_gold_win = df[['dire_total_gold','radiant_win' ]]
dire_mean_gold_if_win = d_gold_win[d_gold_win['radiant_win'] ==0].mean()
dire_mean_gold_if_lose = d_gold_win[d_gold_win['radiant_win'] ==1].mean()

mean_gold_if_win = (radiant_mean_gold_if_win[0] + dire_mean_gold_if_win[0])/2
mean_gold_if_win

mean_gold_if_lose = (radiant_mean_gold_if_lose[0] + dire_mean_gold_if_lose[0])/2
mean_gold_if_lose

#mean total xp by radiant teams with win
r_xp_win = df[['radiant_total_xp','radiant_win' ]]
radiant_mean_xp_if_win = r_xp_win[r_xp_win['radiant_win'] == 1].mean()
radiant_mean_xp_if_lose = r_xp_win[r_xp_win['radiant_win'] == 0].mean()

#mean total gold by dire teams with win
d_xp_win = df[['dire_total_xp','radiant_win' ]]
dire_mean_xp_if_win = d_xp_win[d_xp_win['radiant_win'] == 0].mean()
dire_mean_xp_if_lose = d_xp_win[d_xp_win['radiant_win'] == 1].mean()

mean_xp_if_win = (radiant_mean_xp_if_win[0] + dire_mean_xp_if_win[0])/2
mean_xp_if_win

mean_xp_if_lose = (radiant_mean_xp_if_lose[0] + dire_mean_xp_if_lose[0])/2
mean_xp_if_lose

#added hero win rate to heroes df
heroes = heroes.merge(all_heroes_win_per.rename('win_%').to_frame(), left_index=True, right_index=True)
heroes.head()

#mean total items by radiant teams with win
r_items_win = df[['r1_items','r2_items','r3_items','r4_items','r5_items','radiant_win']]
radiant_mean_items_if_win = r_items_win[r_items_win['radiant_win'] == 1].mean()
radiant_mean_items_if_lose = r_items_win[r_items_win['radiant_win'] == 0].mean()

#mean total items by dire teams with win
d_items_win = df[['d1_items','d2_items','d3_items','d4_items','d5_items','radiant_win']]
dire_mean_items_if_win = d_items_win[d_items_win['radiant_win'] == 0].mean()
dire_mean_items_if_lose = d_items_win[d_items_win['radiant_win'] == 1].mean()

mean_items_if_win = (radiant_mean_items_if_win[0] + dire_mean_items_if_win[0])/2
mean_items_if_win

mean_items_if_lose = (radiant_mean_items_if_lose[0] + dire_mean_items_if_lose[0])/2
mean_items_if_lose

#slice radiant time at first bottle item purchase with win, drop na to subset matches with radiant bottle purchase only
r_bottle = df[['radiant_bottle_time', 'radiant_win']].dropna()

#mean time of radiant bottle purchase with win vs. loss
r_bottle.groupby(['radiant_win']).mean()

#count of matches won by radiant team with bottle purchase
r_bottle_win = r_bottle.groupby(['radiant_win']).size()
r_bottle_win

#radiant win rate with bottle item
r_bottle_win_per = r_bottle_win[1] / (r_bottle_win[0] + r_bottle_win[1])
r_bottle_win_per

#slice dire time at first bottle item purchase with win, drop na to subset matches with dire bottle purchase only
d_bottle = df[['dire_bottle_time', 'radiant_win']].dropna()

#mean time of dire bottle purchase with win vs. loss
d_bottle.groupby(['radiant_win']).mean()

#count of matches won by dire team with bottle purchase
d_bottle_win = d_bottle.groupby(['radiant_win']).size()
d_bottle_win

#dire win rate with bottle item
d_bottle_win_per = d_bottle_win[0] / (d_bottle_win[0] + d_bottle_win[1])
d_bottle_win_per

#overall bottle item win rate
bottle_win_per = (r_bottle_win_per + d_bottle_win_per)/2
bottle_win_per

#slice radiant time at first courier item purchase with win, drop na to subset matches with radiant courier purchase only
r_courier = df[['radiant_courier_time', 'radiant_win']].dropna()
r_no_courier = df[['radiant_courier_time', 'radiant_win']]
r_courier.groupby(['radiant_win']).mean()

r_no_courier["radiant_courier_time"][~r_no_courier["radiant_courier_time"].isna()]=-1
r_no_courier.fillna(1,inplace=True)

r_no_courier.groupby(['radiant_courier_time']).mean()

#count of matches won by radiant team with courier purchase
r_courier_win = r_courier.groupby(['radiant_win']).size()
r_courier_win

#radiant win rate with courier item
r_courier_win_per = r_courier_win[1] / (r_courier_win[0] + r_courier_win[1])
r_courier_win_per

#slice dire time at first courier item purchase with win, drop na to subset matches with dire courier purchase only
d_courier = df[['dire_courier_time', 'radiant_win']].dropna()
d_courier.groupby(['radiant_win']).mean()

#count of matches won by dire team with courier purchase
d_courier_win = d_courier.groupby(['radiant_win']).size()
d_courier_win

#dire win rate with courier item
d_courier_win_per = d_courier_win[0] / (d_courier_win[0] + d_courier_win[1])
d_courier_win_per

#overall courier item win rate
courier_win_per = (r_courier_win_per + d_courier_win_per)/2
courier_win_per

#slice radiant time at first flying courier item purchase with win, drop na to subset matches with radiant flying courier purchase only
r_fcourier = df[['radiant_flying_courier_time', 'radiant_win']].dropna()
r_fcourier.groupby(['radiant_win']).mean()

#count of matches won by radiant team with flying courier purchase
r_fcourier_win = r_fcourier.groupby(['radiant_win']).size()
r_fcourier_win

#radiant win rate with flying courier item
r_fcourier_win_per = r_fcourier_win[1] / (r_fcourier_win[0] + r_fcourier_win[1])*100
r_fcourier_win_per

#slice dire time at first flying courier item purchase with win, drop na to subset matches with dire flying courier purchase only
d_fcourier = df[['dire_flying_courier_time', 'radiant_win']].dropna()
d_fcourier.groupby(['radiant_win']).mean()

#count of matches won by dire team with flying courier purchase
d_fcourier_win = d_fcourier.groupby(['radiant_win']).size()
d_fcourier_win

#dire win rate with flying courier item
d_fcourier_win_per = d_fcourier_win[0] / (d_fcourier_win[0] + d_fcourier_win[1])*100
d_fcourier_win_per

#overall flying courier item win rate
fcourier_win_per = (r_fcourier_win_per + d_fcourier_win_per)/2
fcourier_win_per

#slice radiant time at first tpscroll item purchase with win, drop na to subset matches with radiant tpscroll purchase only
r_scroll = df[['radiant_tpscroll_count', 'radiant_win']].dropna()
r_scroll.groupby(['radiant_win']).mean()

#count of matches won by radiant team with tpscroll purchase
r_scroll_win = r_scroll.groupby(['radiant_win']).size()
r_scroll_win

#radiant win rate with tpscroll item
r_scroll_win_per = r_scroll_win[1] / (r_scroll_win[0] + r_scroll_win[1])*100
r_scroll_win_per

#slice dire time at first tpscroll item purchase with win, drop na to subset matches with dire tpscroll purchase only
d_scroll = df[['dire_tpscroll_count', 'radiant_win']].dropna()
d_scroll.groupby(['radiant_win']).mean()

#count of matches won by dire team with tpscroll purchase
d_scroll_win = d_scroll.groupby(['radiant_win']).size()
d_scroll_win

#dire win rate with tpscroll item
d_scroll_win_per = d_scroll_win[0] / (d_scroll_win[0] + d_scroll_win[1])*100
d_scroll_win_per

#overall tpscroll item win rate
scroll_win_per = (r_scroll_win_per + d_scroll_win_per)/2
scroll_win_per

#slice radiant time at first boots item purchase with win, drop na to subset matches with radiant boots purchase only
r_boots = df[['radiant_boots_count', 'radiant_win']].dropna()
r_boots.groupby(['radiant_win']).mean()

#count of matches won by radiant team with boots purchase
r_boots_win = r_boots.groupby(['radiant_win']).size()
r_boots_win

#radiant win rate with boots item
r_boots_win_per = r_boots_win[1] / (r_boots_win[0] + r_boots_win[1])*100
r_boots_win_per

#slice dire time at first boots item purchase with win, drop na to subset matches with dire boots purchase only
d_boots = df[['dire_boots_count', 'radiant_win']].dropna()
d_boots.groupby(['radiant_win']).mean()

#count of matches won by dire team with boots purchase
d_boots_win = d_boots.groupby(['radiant_win']).size()
d_boots_win

#dire win rate with boots item
d_boots_win_per = d_boots_win[0] / (d_boots_win[0] + d_boots_win[1])*100
d_boots_win_per

#overall boots item win rate
boots_win_per = (r_boots_win_per + d_boots_win_per)/2
boots_win_per

#slice radiant time at first ward observer item purchase with win, drop na to subset matches with radian ward observer purchase only
r_ward_obs = df[['radiant_ward_observer_count', 'radiant_win']].dropna()
r_ward_obs.groupby(['radiant_win']).mean()

#count of matches won by radiant team with ward observer purchase
r_ward_obs_win = r_ward_obs.groupby(['radiant_win']).size()
r_ward_obs_win
#resulting overall win % will be 50%

#slice radiant time at first ward sentry item purchase with win, drop na to subset matches with radiant ward sentry purchase only
r_ward_sen = df[['radiant_ward_sentry_count', 'radiant_win']].dropna()
r_ward_sen.groupby(['radiant_win']).mean()

#count of matches won by radiant team with ward sentry purchase
r_ward_sen_win = r_ward_sen.groupby(['radiant_win']).size()
r_ward_sen_win
#resulting overall win % will be 50%

#slice radiant time at first ward item purchase with win, drop na to subset matches with radiant ward purchase only
r_ward = df[['radiant_first_ward_time', 'radiant_win']].dropna()
r_ward.groupby(['radiant_win']).mean()

#count of matches won by radiant team with ward purchase
r_ward_win = r_ward.groupby(['radiant_win']).size()
r_ward_win

#radiant win rate with ward item
r_ward_win_per = r_ward_win[1] / (r_ward_win[0] + r_ward_win[1])*100
r_ward_win_per

#slice dire time at first ward sentry item purchase with win, drop na to subset matches with dire ward sentry purchase only
d_ward = df[['dire_first_ward_time', 'radiant_win']].dropna()
d_ward.groupby(['radiant_win']).mean()

#count of matches won by dire team with ward purchase
d_ward_win = d_ward.groupby(['radiant_win']).size()
d_ward_win

#dire win rate with ward item
d_ward_win_per = d_ward_win[0] / (d_ward_win[0] + d_ward_win[1])*100
d_ward_win_per

#overall ward item win rate
ward_win_per = (r_ward_win_per + d_ward_win_per)/2
ward_win_per