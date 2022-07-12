from datetime import datetime, timezone

import numpy as np
from hamcrest import *

from src.matrix_profile import MatrixProfile
from src.preprocess import number_of_interval_in_days
from tests.helper.BgDfBuilder import create_time_stamps

# create fake timeseries
interval = 1440  # a value per day
ts_length = number_of_interval_in_days(14, interval)  # 14 days of data
start_date = datetime(year=2019, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)
times = create_time_stamps(start_date, ts_length, interval)
values = np.random.uniform(low=0.1, high=14.6, size=ts_length)


def test_calculates_matrix_profile_for_a_series_and_sequence_size():
    motif_length = 3
    mp = MatrixProfile(times, values, motif_length)
    result = mp.mp

    assert_that(result.shape, is_((ts_length - motif_length + 1, 4)))


def test_gives_index_for_motif_and_its_nearest_neighbour_given_motif_index():
    motif_length = 4
    mp = MatrixProfile(times, values, motif_length)
    profile = mp.mp
    indexes = np.argsort(profile[:, 0])

    # check lowest motif
    lowest = 0
    motif_idx_0, nearest_neighbour_idx_0 = mp.get_motif_and_nearest_neighbor_idx_for_xth_motif(lowest)
    assert_that(motif_idx_0, is_(indexes[lowest]))
    assert_that(nearest_neighbour_idx_0, is_(profile[motif_idx_0, 1]))

    # check second-lowest motif
    second_lowest = 1
    motif_idx_1, nearest_neighbour_idx_1 = mp.get_motif_and_nearest_neighbor_idx_for_xth_motif(second_lowest)
    assert_that(motif_idx_1, is_(indexes[second_lowest]))
    assert_that(nearest_neighbour_idx_1, is_(profile[motif_idx_1, 1]))


def test_can_show_matrix_profile_plot():
    motif_length = 4
    mp = MatrixProfile(times, values, motif_length)
    # there is no assert as this test actually plots the graph, it will throw an exception if something is amiss
    mp.plot_ts_motif_and_profile(0, 'TS Y label', 'Time')


def test_shows_z_score_normalised_values_as_well_matrix_profile_plot():
    motif_length = 4
    mp = MatrixProfile(times, values, motif_length)
    # there is no assert as this test actually plots the graph, it will throw an exception if something is amiss
    mp.plot_ts_motif_and_profile(0, 'TS Y label', 'Time', True)


def test_returns_x_for_discord_area():
    motif_length = 7
    mp = MatrixProfile(times, values, motif_length)

    x = mp.least_similar_x()
    # |ts|-m+1 is number of entries in the matrix profile, but indexed at 0
    assert_that(x, is_(len(times) - motif_length))


def test_returns_top_motifs_for_ts():
    motif_length = 3
    mp = MatrixProfile(times, values, motif_length)
    max_distance = 1.0

    motif_distances, motif_indices = mp.top_motives(max_distance)

    assert_that(len(motif_indices), is_(len(motif_distances)))


def test_can_show_top_motives():
    # no assert just checking the plot function does not error
    MatrixProfile(times, values, 3).plot_top_motives_for_max_distance_and_min_neighbours('y label', 'x_label', 2.0, 2)
