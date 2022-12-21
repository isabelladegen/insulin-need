from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.configurations import GeneralisedCols, Resampling


@dataclass
class AxisResolution:
    label = 'for axis descriptions'
    ticks = []
    labels_for_graph = []


@dataclass
class Weekdays(AxisResolution):
    label = 'Weekdays'
    ticks = [0, 1, 2, 3, 4, 5, 6]
    labels_for_graph = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


@dataclass
class Hours(AxisResolution):
    label = 'Hours'
    ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    ticks = ticks


@dataclass
class Months(AxisResolution):
    label = 'Months'
    ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    labels_for_graph = ['Jan', 'Feb', 'Mar', 'Arp', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class Heatmap:
    """Class for plotting heatmaps from differently sampled, multivariate, time series dataframes
        Methods
        -------
    """

    def __init__(self, df: pd.DataFrame, sampling: Resampling):
        """
            Parameters
            ----------
            df : DataFrame
                Data to plot in heathmap

            sampling : Resampling
                What the time series are resampled as for descriptive strings
        """
        self.__df = df
        self.__sampling = sampling

    def plot_heatmap(self,
                     plot_rows: [str],
                     zip_ids: [str] = [],
                     x_axis: AxisResolution = Months(),
                     y_axis: AxisResolution = Weekdays(),
                     aggfunc=np.mean,
                     show_title: bool = True

                     ):
        """
            Plots a grid of heatmaps with the rows being the variates provided in plot_rows and the columns being the
            different zip ids.

            Parameters
            ----------
            plot_rows : [str]
                Columns of df to plot as rows. Each column in the list will create a new heatmap on a new row.

            zip_ids : [str]
                Ids in the GeneralisedCols.id column of the df to plot.
                Can be used to plot multiple zip_ids at the same time in the column of the grid of heatmaps.
                If empty, there's only one column in the graph.

            x_axis : AxisResolution
                granularity of x-axis of each heatmap in the plot

            y_axis : AxisResolution
                granularity of y-axis of each heatmap in the plot

            aggfunc : function
                Aggregation function for pixel in each heatmap. Default np.mean.
                Can be any function that can aggregate numerical values

            show_title : bool
                If the title and overall plot labels are displayed or not


            Returns
            -------
            plotted_data : {zip_ids : {rows : data}}
        """
        # filter df by zip_ids that need to be plotted
        if zip_ids:
            filtered_df = self.__df.loc[self.__df[GeneralisedCols.id].isin(zip_ids)]
        else:
            filtered_df = self.__df
        columns = list(filtered_df[GeneralisedCols.id].unique())

        # calculate time columns for the resolution required
        timed_df = filtered_df.copy()
        timed_df[x_axis.label] = self.__translate_datetime_to(filtered_df, x_axis)
        timed_df[y_axis.label] = self.__translate_datetime_to(filtered_df, y_axis)

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        # grid with rows being the value columns and columns being the different zip ids
        fig, axes = plt.subplots(nrows=len(plot_rows),
                                 ncols=len(columns),
                                 sharey=True,
                                 sharex=True,
                                 squeeze=0,
                                 figsize=(10, 7))

        if show_title:
            title = 'Heatmaps of ' + self.__sampling.description + ' time series (x=' + x_axis.label + ', y=' + y_axis.label + ')'
            fig.suptitle(title)

        data_plotted = {}

        for column_idx, zip_id in enumerate(columns):
            df_for_id = timed_df.loc[timed_df[GeneralisedCols.id] == zip_id]
            row_data_for_zip_id = {}
            for row_idx, ts_variate in enumerate(plot_rows):
                sub_data = df_for_id[[x_axis.label, y_axis.label, ts_variate]]
                pivoted_data = self.__pivot_data(sub_data, aggfunc, ts_variate, x_axis, y_axis)
                row_data_for_zip_id[ts_variate] = pivoted_data

                ax = sns.heatmap(pivoted_data,
                                 linewidth=0.5,
                                 yticklabels=y_axis.labels_for_graph,
                                 square=False,
                                 ax=axes[row_idx][column_idx],
                                 cmap="mako"
                                 )
                if show_title:
                    ax.set_title('# Samples ' + str(sub_data[ts_variate].count()))
                ax.set(xlabel='', ylabel='')
            data_plotted[zip_id] = row_data_for_zip_id

        # column label
        if show_title:
            for ax, zip_id in zip(axes[-1], columns):
                ax.set_xlabel(zip_id)

        # row labels
        for ax, row in zip(axes[:, 0], plot_rows):
            ax.set_ylabel(str(row))

        fig.tight_layout()
        plt.show()
        return data_plotted

    @staticmethod
    def __translate_datetime_to(df: pd.DataFrame, res: AxisResolution):
        # returns the datetime column translated into AxesResolution (e.g months, daysofweek, ...)
        if isinstance(res, Months):
            return df[GeneralisedCols.datetime].dt.month
        elif isinstance(res, Weekdays):
            return df[GeneralisedCols.datetime].dt.dayofweek
        elif isinstance(res, Hours):
            return df[GeneralisedCols.datetime].dt.hour

    @staticmethod
    def __pivot_data(data, aggfunc, value_col: str, x_axis: AxisResolution, y_axis: AxisResolution):
        df = data.copy()

        # change to float
        df[value_col] = df[value_col].astype(np.float64, copy=False)
        pivot = pd.pivot_table(data=df,
                               index=y_axis.label,
                               values=value_col,
                               columns=x_axis.label,
                               aggfunc=aggfunc)

        # insert columns that are missing in columns
        existing_columns = list(pivot.columns)
        missing_columns = list(set(x_axis.ticks) - set(existing_columns))
        for missing_column in missing_columns:
            pivot[missing_column] = np.NAN

        # insert rows that are missing in the data
        pivot = pivot.reindex(y_axis.ticks)

        # sort columns
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        # sort rows
        return pivot
