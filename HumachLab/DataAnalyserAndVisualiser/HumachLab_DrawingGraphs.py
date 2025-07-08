"""
File Name: HumachLab_GraphManager.py 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 12:54 pm
"""

import os
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib

# if using a Jupyter notebook, include:
# %matplotlib inline
import math
import numbers
# import radar_chart_structure
from numpy import argmax, sqrt
from sklearn.metrics import precision_recall_curve, roc_curve
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from numpy import pi

from sklearn.utils import resample



class HumachLab_DrawingGraphs:

    def __init__(self, logger):
        self.logger = logger

        self.setup_mpl_graph_properties()

        return


    # ######################################
    # ### Settings of graph layout
    # ######################################

    # ## Setup graph dimension, font-size, colors and other properties
    def setup_mpl_graph_properties(self,
        plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=30, bg_face_color='white',
        show_grid=True, border_top=True, border_bottom=True, border_left=True, border_right=True
    ):
        # Plot style
        plt.style.use(plot_style)
        # Plot size to 16" x 12"
        matplotlib.rc('figure', figsize=fig_size)
        # Font size to 30
        matplotlib.rc('font', size=font_size)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = "Times New Roman"
        # Set backgound color to white
        matplotlib.rc('axes', facecolor=bg_face_color)
        # Remove/show grid lines
        matplotlib.rc('axes', grid=show_grid)
        # Do not display top and right frame lines
        matplotlib.rc('axes.spines', top=border_top, bottom=border_bottom, left=border_left, right=border_right)
        return


    def get_plot_resources(n, same_line_style=True):
        colors = []
        markers = []
        style = []
        remove_params = ['None', None, ' ', '', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        colors_array = [i for i in list(matplotlib.colors.cnames.keys()) if i not in remove_params]
        colors_array = sorted(colors_array, key=len)
        markers_array = ([f'{i}' for i in list(matplotlib.markers.MarkerStyle.markers.keys()) if i not in remove_params])#[:-12]
        styles_array = [i for i in list(matplotlib.lines.lineStyles.keys()) if i not in remove_params]
        p = round(n/len(colors_array))+1
        q = round(n/len(markers_array))+1
        r = round(n/len(styles_array))+1
        colors = (p*colors_array)[:n]
        markers = (q*markers_array)[:n] #([f'{i}' for i in (q*markers_array)])[:n] #(q*markers_array)[:n]
        styles = (r*styles_array)[:n]
        if same_line_style:
            styles = n*['--']
        return colors, markers, styles

    def get_plot_resources_custom(n, same_line_style=True):
        colors = []
        markers = []
        style = []
        colors_array = ['green', 'red', 'blue', 'orange', 'gray', 'brown', 'magenta', 'indigo', 'lime', 'olive', 'purple', 'violet', 'navy', 'cyan', 'black', 'yellow', 'lime', 'olive', 'purple', 'violet', 'green', 'red', 'blue', 'orange', 'gray', 'brown']
        markers_array = ['o', 'x', '*', '^', '+', '<', '>', 'v', 'X', 'd', 'D', '.', 's', '|', 'p', 'P', 'h', 'H', '1', '2', '3', '4', '8', '_', ',']
        styles_array = ['--', ':', '-.', '-', ';', '.', ',']
        p = round(n/len(colors_array))+1
        q = round(n/len(markers_array))+1
        r = round(n/len(styles_array))+1
        colors = (p*colors_array)[:n]
        markers = (q*markers_array)[:n] #([f'{i}' for i in (q*markers_array)])[:n] #(q*markers_array)[:n]
        styles = (r*styles_array)[:n]
        if same_line_style:
            styles = n*['--']
        return colors, markers, styles


    # ######################################
    # ### Drawing line graph
    # ######################################

    # ## Single line graph
    def draw_lineplot(self,
            x_data, y_data_list, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=0, y_tick_rotate=0,
            custom_tick_steps=False, x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None,
            line_width=1, marker_size=3,
            legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
    ):
        # Markers, color and style
        markers, ln_colors, ln_style = self.get_plot_resources(len(y_data_list))

        print(f'All lengths: {len(markers)}, {len(ln_colors)}, {len(ln_style)}')
        # Create the plot object
        _, ax = plt.subplots()

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        y_max = -99999999
        y_min = 99999999

        for i in range(len(y_data_list)):
            y_data = y_data_list[i]
            if log_presentation:
                y_data = list(np.log(y_data))

            ax.plot(
                x_data, y_data, f'{markers[i]}-' if not line_style else line_style[i], linewidth=line_width,
                markersize=((3 * line_width) if not line_style else marker_size), color=ln_colors[i]
            )
            mx = np.max(y_data)
            mn = np.min(y_data)
            y_max = mx if mx > y_max else y_max
            y_min = mn if mn < y_min else y_min

        # If there are threshold level
        if threshod_levels:
            for jj in range(len(threshod_levels)):
                threshod_level = threshod_levels[jj]
                y_max = threshod_level if threshod_level > y_max else y_max
                y_min = threshod_level if threshod_level < y_min else y_min
                ax.axhline(y=threshod_level, linestyle=ln_style[jj], color=ln_colors[jj],
                           label=str(threshod_level) if not threshod_names else threshod_names[jj])

        # Label the axes and provide a title
        if title:
            ax.set_title(title)

        # Show legend
        if legend_names:
            if threshod_levels:
                for jj in range(len(threshod_levels)):
                    threshod_level = threshod_levels[jj]
                    legend_names += ['Threshold'] if not threshod_names else [threshod_names[jj]]

            if legend_outside:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                          fontsize=legend_font_size, ncol=2, frameon=True, bbox_to_anchor=(1, 1))
            else:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                          fontsize=legend_font_size, ncol=2, frameon=True)

        # ax.legend(legend_names, loc='best', fancybox=True, framealpha=1.0, fontsize='medium')
        # ax.legend(fancybox=True, framealpha=0.5, loc='best'/'upper left', fontsize='small')

        # Set X & Y labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set visible X & Y limits
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)

        # Set X & Y ticks rotation
        plt.xticks(rotation=x_tick_rotate)
        plt.yticks(rotation=y_tick_rotate)

        # Set X & Y ticks steps size
        ss = str(y_tick_stepsize)
        exx = ss.split('.')[-1]
        num_dig = len(str(int(exx)))
        y_min = round(y_min, num_dig) - y_tick_stepsize
        y_max = round(y_max, num_dig)

        should_add = False
        mynewlist = [s for s in x_data if isinstance(s, numbers.Number)]
        if len(x_data) == len(mynewlist):
            should_add = True

        #     return
        # plt.xticks(np.arange(0, len(x_data)+x_tick_stepsize, x_tick_stepsize))
        if custom_tick_steps:
            plt.xticks(np.arange(0, len(x_data) + (x_tick_stepsize if should_add else 0), x_tick_stepsize))
        #         plt.yticks(np.arange(y_min, y_max+2*y_tick_stepsize, y_tick_stepsize))

        # Set X & Y ticks or step names
        #     print(x_data, '\n', plt.xticks())
        #     if not len(x_data)==len(list(plt.xticks())):
        #         x_tick_names = x_data

        if x_tick_names:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xticks(x_data)
            ax.set_xticklabels(x_tick_names)
        if y_tick_names:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_yticks(y_max)
            ax.set_yticklabels(y_tick_names)

        print(x_data, custom_tick_steps, should_add, x_tick_names)
        # Adjusting left and bottom spaces
        fig = plt.gcf()
        # fig.subplots_adjust(top=adjust_top, bottom=adjust_bottom, left=adjust_left, right=adjust_right)
        fig.subplots_adjust(bottom=adjust_bottom, left=adjust_left)

        if filename:
            # save_format = 'pdf' 'eps' 'png'
            save_format = 'png'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
            save_format = 'eps'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')

        plt.show()
        return

    # ## Draw single line plot with average value
    def draw_lineplot_avg(self,
        x_data, y_data_list, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None, x_tick_rotate=0, y_tick_rotate=0,
        x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None, line_width=1, marker_size=3,
        legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
        adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
        title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
    ):
        # Markers, color and style
        markers, ln_colors, ln_style = self.get_plot_resources(len(y_data_list))

        # Create the plot object
        _, ax = plt.subplots()

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        y_max = -99999999
        y_min = 99999999

        for i in range(len(y_data_list)):
            y_data = y_data_list[i]
            if log_presentation:
                y_data = list(np.log(y_data))

            ax.plot(
                x_data, y_data, f'{markers[i]}-' if not line_style else line_style[i], linewidth=line_width, markersize=((3*line_width) if not line_style else marker_size), color=ln_colors[i]
            )
            mx = np.max(y_data)
            mn = np.min(y_data)
            y_max = mx if mx>y_max else y_max
            y_min = mn if mn<y_min else y_min

        # If there are threshold level
        if threshod_levels:
            for jj in range(len(threshod_levels)):
                threshod_level = threshod_levels[jj]
                y_max = threshod_level if threshod_level>y_max else y_max
                y_min = threshod_level if threshod_level<y_min else y_min
                ax.axhline(y=threshod_level, linestyle=ln_style[0], color=ln_colors[jj], label=str(threshod_level) if not threshod_names else threshod_names[jj])

        # Label the axes and provide a title
        if title:
            ax.set_title(title)

        # Show legend
        if legend_names:
            if threshod_levels:
                for jj in range(len(threshod_levels)):
                    threshod_level = threshod_levels[jj]
                    legend_names += ['Threshold'] if not threshod_names else [threshod_names[jj]]

            if legend_outside:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50, fontsize=legend_font_size, ncol=2, frameon=True, bbox_to_anchor=(1, 1))
            else:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50, fontsize=legend_font_size, ncol=2, frameon=True)

        # ax.legend(legend_names, loc='best', fancybox=True, framealpha=1.0, fontsize='medium')
        # ax.legend(fancybox=True, framealpha=0.5, loc='best'/'upper left', fontsize='small')

        # Set X & Y labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set visible X & Y limits
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)

        # Set X & Y ticks rotation
        plt.xticks(rotation=x_tick_rotate)
        plt.yticks(rotation=y_tick_rotate)

        # Set X & Y ticks steps size
        ss = str(y_tick_stepsize)
        exx = ss.split('.')[-1]
        num_dig = len(str(int(exx)))
        y_min = round(y_min, num_dig)-y_tick_stepsize
        y_max = round(y_max, num_dig)

        should_add = False
        mynewlist = [s for s in x_data if isinstance(s, numbers.Number)]
        if len(x_data)==len(mynewlist):
            should_add = True


        #plt.xticks(np.arange(0, len(x_data)+x_tick_stepsize, x_tick_stepsize))
        plt.xticks(np.arange(0, len(x_data)+(x_tick_stepsize if should_add else 0), x_tick_stepsize))
        plt.yticks(np.arange(y_min, y_max+2*y_tick_stepsize, y_tick_stepsize))

        # ax.ticklabel_format(useOffset=False) #Comment if found error: AttributeError: This method only works with the ScalarFormatter

        # Set X & Y ticks or step names
        if not len(x_data)==len(list(plt.xticks())):
            x_tick_names = x_data

        if x_tick_names:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xticks(x_data)
            ax.set_xticklabels(x_tick_names)
        if y_tick_names:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_yticks(y_max)
            ax.set_yticklabels(y_tick_names)

        # Adjusting left and bottom spaces
        fig = plt.gcf()
        # fig.subplots_adjust(top=adjust_top, bottom=adjust_bottom, left=adjust_left, right=adjust_right)
        fig.subplots_adjust(bottom=adjust_bottom, left=adjust_left)

        if filename:
            # save_format = 'pdf' 'eps' 'png'
            save_format = 'png'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
            save_format = 'eps'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')

        plt.show()
        return



    # ######################################
    # ### Drawing bar graph
    # ######################################

    # ## Single line graph
    def draw_barplot(self,
            x_data, y_data_list, x_label, y_label, col_names, bar_width, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=0, y_tick_rotate=0,
            custom_tick_steps=False, x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None,
            line_width=1, marker_size=3,
            legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
    ):
        # Markers, color and style
        markers, ln_colors, ln_style = self.get_plot_resources(len(y_data_list))

        print(f'All lengths: {len(markers)}, {len(ln_colors)}, {len(ln_style)}')
        # Create the plot object
        _, ax = plt.subplots()

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        y_max = -99999999
        y_min = 99999999

        data_dict = {}
        for i, y_data in enumerate(y_data_list):
            if log_presentation:
                y_data = list(np.log(y_data))
            data_dict[col_names[i]] = y_data
            mx = np.max(y_data)
            mn = np.min(y_data)
            y_max = mx if mx > y_max else y_max
            y_min = mn if mn < y_min else y_min

        df = pd.DataFrame(data_dict, index=x_data)
        ax = df.plot.bar(rot=0)

        # If there are threshold level
        if threshod_levels:
            for jj in range(len(threshod_levels)):
                threshod_level = threshod_levels[jj]
                y_max = threshod_level if threshod_level > y_max else y_max
                y_min = threshod_level if threshod_level < y_min else y_min
                ax.axhline(y=threshod_level, linestyle=ln_style[jj], color=ln_colors[jj],
                           label=str(threshod_level) if not threshod_names else threshod_names[jj])

        # Label the axes and provide a title
        if title:
            ax.set_title(title)

        # Show legend
        if legend_names:
            if threshod_levels:
                for jj in range(len(threshod_levels)):
                    threshod_level = threshod_levels[jj]
                    legend_names += ['Threshold'] if not threshod_names else [threshod_names[jj]]

            if legend_outside:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                          fontsize=legend_font_size, ncol=2, frameon=True, bbox_to_anchor=(1, 1))
            else:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                          fontsize=legend_font_size, ncol=2, frameon=True)

        # ax.legend(legend_names, loc='best', fancybox=True, framealpha=1.0, fontsize='medium')
        # ax.legend(fancybox=True, framealpha=0.5, loc='best'/'upper left', fontsize='small')

        # Set X & Y labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set visible X & Y limits
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)

        # Set X & Y ticks rotation
        plt.xticks(rotation=x_tick_rotate)
        plt.yticks(rotation=y_tick_rotate)

        # Set X & Y ticks steps size
        ss = str(y_tick_stepsize)
        exx = ss.split('.')[-1]
        num_dig = len(str(int(exx)))
        y_min = round(y_min, num_dig) - y_tick_stepsize
        y_max = round(y_max, num_dig)

        should_add = False
        mynewlist = [s for s in x_data if isinstance(s, numbers.Number)]
        if len(x_data) == len(mynewlist):
            should_add = True

        #     return
        # plt.xticks(np.arange(0, len(x_data)+x_tick_stepsize, x_tick_stepsize))
        if custom_tick_steps:
            plt.xticks(np.arange(0, len(x_data) + (x_tick_stepsize if should_add else 0), x_tick_stepsize))
        #         plt.yticks(np.arange(y_min, y_max+2*y_tick_stepsize, y_tick_stepsize))

        # Set X & Y ticks or step names
        #     print(x_data, '\n', plt.xticks())
        #     if not len(x_data)==len(list(plt.xticks())):
        #         x_tick_names = x_data

        if x_tick_names:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xticks(x_data)
            ax.set_xticklabels(x_tick_names)
        if y_tick_names:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_yticks(y_max)
            ax.set_yticklabels(y_tick_names)

        print(x_data, custom_tick_steps, should_add, x_tick_names)
        # Adjusting left and bottom spaces
        fig = plt.gcf()
        # fig.subplots_adjust(top=adjust_top, bottom=adjust_bottom, left=adjust_left, right=adjust_right)
        fig.subplots_adjust(bottom=adjust_bottom, left=adjust_left)

        if filename:
            # save_format = 'pdf' 'eps' 'png'
            save_format = 'png'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
            save_format = 'eps'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')

        plt.show()
        return



    # ######################################
    # ### Drawing error bar graph
    # ######################################

    def draw_errorbar(self,
            x_data, y_data_list, x_error_list, y_error_list, x_label, y_label, log_presentation=False,
            x_tick_names=None, y_tick_names=None, x_tick_rotate=0, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None, line_width=1,
            marker_size=3, er_line_width=1,
            legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
    ):
        # Markers, color and style
        markers, ln_colors, ln_style = self.get_plot_resources(len(y_data_list))
        er_colors = ln_colors

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        y_max = -99999999
        y_min = 99999999
        dat_len = len(y_data_list)
        shft_val = 0.03 * er_line_width  # 1.0/dat_len

        if x_tick_names:
            x_data = [xx for xx in range(len(x_data))]
        else:
            x_tick_names = [xx for xx in x_data]
            x_data = [xx for xx in range(len(x_data))]

        # ## Subplot object and resize figure
        fig = plt.figure()
        figw, figh = fig.get_size_inches()
        wd = len(x_data) * len(y_data_list) * shft_val

        figw = figw if figw > wd else wd
        figw = figw + (2 * shft_val)
        _, ax = plt.subplots(figsize=(figw, figh))

        # ## Shifting plot to adjust
        prev_x_data = x_data
        x_data = np.array(x_data) - ((dat_len * shft_val - shft_val) / 2.0)

        print(x_data, y_data_list)
        for i in range(dat_len):
            y_data = y_data_list[i]
            x_err = x_error_list[i]
            y_err = y_error_list[i]

            if log_presentation:
                y_data = list(np.log(np.abs(y_data)))
                y_err = list(np.log(np.abs(y_err)))

                # if x_tick_names[i] == 'RenEn':
            #         print(x_tick_names[5], y_data[5])
            # np.array(x_data)+(i+1)*shft_val

            #         ax.errorbar(
            #             x_data, y_data, xerr=(None if not (len(x_err)>0) else x_err), yerr=y_err, fmt=f'{markers[i]}--' if not line_style else line_style[i], color=ln_colors[i], linewidth=line_width,
            #             ecolor=er_colors[i], elinewidth=line_width, markersize=((3*line_width) if not line_style else marker_size),
            #             capsize=((3*line_width) if not line_style else marker_size), barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None
            #         )
            print(x_data + i * shft_val, y_data)
            ax.errorbar(
                x_data + i * shft_val, y_data, xerr=(None if not (len(x_err) > 0) else x_err), yerr=y_err,
                fmt=f'{markers[i]}--' if not line_style else line_style[i],
                color=ln_colors[i], linewidth=line_width, ecolor=er_colors[i], elinewidth=er_line_width,
                markersize=((3 * er_line_width) if not line_style else marker_size),
                capsize=((2 * er_line_width) if not line_style else marker_size), barsabove=False, lolims=False,
                uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None
            )

            mx = np.max(y_data)
            mn = np.min(y_data)
            y_max = mx if mx > y_max else y_max
            y_min = mn if mn < y_min else y_min

        # If there are threshold level
        if threshod_levels:
            for jj in range(len(threshod_levels)):
                threshod_level = threshod_levels[jj]
                y_max = threshod_level if threshod_level > y_max else y_max
                y_min = threshod_level if threshod_level < y_min else y_min
                ax.axhline(y=threshod_level, linestyle=ln_style[jj], color=ln_colors[jj],
                           label=str(threshod_level) if not threshod_names else threshod_names[jj])

        # Label the axes and provide a title
        if title:
            ax.set_title(title)

        # Show legend
        if legend_names:
            if threshod_levels:
                for jj in range(len(threshod_levels)):
                    threshod_level = threshod_levels[jj]
                    legend_names += ['Threshold'] if not threshod_names else [threshod_names[jj]]

            if legend_outside:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                          fontsize=legend_font_size, ncol=1, frameon=True, bbox_to_anchor=(1, 1))
            else:
                ax.legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                          fontsize=legend_font_size, ncol=1, frameon=True)

        # ax.legend(legend_names, loc='best', fancybox=True, framealpha=1.0, fontsize='medium') , frameon=True
        # ax.legend(fancybox=True, framealpha=0.5, loc='best'/'upper left', fontsize='small')
        m = list(fig.get_size_inches())

        # Set X & Y labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set visible X & Y limits
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)

        # Set X & Y ticks rotation
        plt.xticks(rotation=x_tick_rotate)
        plt.yticks(rotation=y_tick_rotate)

        # Set X & Y ticks steps size
        ss = str(y_tick_stepsize)
        exx = ss.split('.')[-1]
        num_dig = len(str(int(exx)))
        y_min = round(y_min, num_dig) - y_tick_stepsize
        y_max = round(y_max, num_dig)

        #     should_add = False
        #     mynewlist = [s for s in x_data if isinstance(s, numbers.Number)]
        #     if len(x_data)==len(mynewlist):
        #         should_add = True
        #     #plt.xticks(np.arange(0, len(x_data)+x_tick_stepsize, x_tick_stepsize))
        #     plt.xticks(np.arange(0, len(x_data) + (x_tick_stepsize if should_add else 0), x_tick_stepsize))
        #     plt.yticks(np.arange(y_min, y_max+2*y_tick_stepsize, y_tick_stepsize))

        # Set X & Y ticks or step names
        if x_tick_names:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            # ax.set_xticks(x_data)
            ax.set_xticks(prev_x_data)
            ax.set_xticklabels(x_tick_names)
        if y_tick_names:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_yticks(y_max)
            ax.set_yticklabels(y_tick_names)

        # Adjusting left and bottom spaces
        fig = plt.gcf()
        # fig.subplots_adjust(top=adjust_top, bottom=adjust_bottom, left=adjust_left, right=adjust_right)
        fig.subplots_adjust(bottom=adjust_bottom, left=adjust_left)

        if filename:
            #         save_format = 'pdf' 'eps' 'png'
            save_format = 'png'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
            save_format = 'eps'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')

        plt.show()

        return


    # ## Draw combined error graph in subplots
    def draw_errorbar_group(self,
            x_data, y_data_list, x_label, y_label, chan_feat_mean_std, channels, graph_dim, log_presentation=False,
            x_tick_names=None, y_tick_names=None, x_tick_rotate=0, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None, line_width=1,
            marker_size=3, er_line_width=1,
            legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
            ):
        # Markers, color and style
        markers, ln_colors, ln_style = self.get_plot_resources(len(y_data_list))
        er_colors = ln_colors
        subnum = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        y_max = -99999999
        y_min = 99999999
        dat_len = len(y_data_list)
        shft_val = 0.03 * er_line_width  # 1.0/dat_len

        if x_tick_names:
            x_data = [xx for xx in range(len(x_data))]
        else:
            x_tick_names = [xx for xx in x_data]
            x_data = [xx for xx in range(len(x_data))]

        # ## Subplot object and resize figure
        fig = plt.figure()
        figw, figh = fig.get_size_inches()
        wd = len(x_data) * len(y_data_list) * shft_val

        figw = figw if figw > wd else wd
        figw = figw + (2 * shft_val)
        fig, ax = plt.subplots(graph_dim[0], graph_dim[1], figsize=(figw * graph_dim[0], figh * graph_dim[1]))

        #     fig.suptitle(graph_title)

        # ## Shifting plot to adjust
        prev_x_data = x_data
        x_data = np.array(x_data) - ((dat_len * shft_val - shft_val) / 2.0)

        rown, coln = 0, 0
        print(channels)
        print(len(ax), len(ax[0]))
        # ####### DATA ###########
        for j in range(len(channels)):
            ch = channels[j]
            rown = int(j / graph_dim[1])
            coln = int(j % graph_dim[1])
            print(f'Channel: {ch}---{rown},{coln}')

            #         ch = f': {chan}' if chan else ''
            #         title = f'Feature Mean-Standard Deviation: {ch}'
            title = f'({subnum[j]}) {ch}'
            #         save_file_name2 = save_file_name + f'_chn-{ch}'
            feat_meanstd = chan_feat_mean_std[chan_feat_mean_std['channel'] == ch]
            #         show_feature_mean_and_standardDeviation_for_seiz_nonseiz(save_dir, feat_meanstd, tot_chns, tot_pats, tot_recs, tot_feats, save_file_name=save_file_name2, chan=ch)

            y_data_list = []
            y_data_list.append(feat_meanstd['non_siez_mean'].values.tolist())
            y_data_list.append(feat_meanstd['siez_mean'].values.tolist())
            print(f'yData: {y_data_list}')

            y_error_list = []
            y_error_list.append(feat_meanstd['non_siez_std'].values.tolist())
            y_error_list.append(feat_meanstd['siez_std'].values.tolist())

            x_error_list = [[] for xx in range(len(y_data_list[0]))]

            print(f'yDataLen: {dat_len}')
            for i in range(dat_len):
                y_data = y_data_list[i]
                x_err = x_error_list[i]
                y_err = y_error_list[i]

                if log_presentation:
                    y_data = list(np.log(np.abs(y_data)))
                    y_err = list(np.log(np.abs(y_err)))

                ax[rown][coln].errorbar(
                    x_data + i * shft_val, y_data, xerr=(None if not (len(x_err) > 0) else x_err), yerr=y_err,
                    fmt=f'{markers[i]}--' if not line_style else line_style[i],
                    color=ln_colors[i], linewidth=line_width, ecolor=er_colors[i], elinewidth=er_line_width,
                    markersize=((3 * er_line_width) if not line_style else marker_size),
                    capsize=((2 * er_line_width) if not line_style else marker_size), barsabove=False, lolims=False,
                    uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None
                )

                mx = np.max(y_data)
                mn = np.min(y_data)
                y_max = mx if mx > y_max else y_max
                y_min = mn if mn < y_min else y_min

            # If there are threshold level
            if threshod_levels:
                for jj in range(len(threshod_levels)):
                    threshod_level = threshod_levels[jj]
                    y_max = threshod_level if threshod_level > y_max else y_max
                    y_min = threshod_level if threshod_level < y_min else y_min
                    ax.axhline(y=threshod_level, linestyle=ln_style[jj], color=ln_colors[jj],
                               label=str(threshod_level) if not threshod_names else threshod_names[jj])

            #         # Label the axes and provide a title
            if title:
                #             plt.set_title(title)
                ax[rown][coln].set_title(title)

            # Show legend
            if rown == 0 and coln == 1:
                if legend_names:
                    if threshod_levels:
                        for jj in range(len(threshod_levels)):
                            threshod_level = threshod_levels[jj]
                            legend_names += ['Threshold'] if not threshod_names else [threshod_names[jj]]

                    if legend_outside:
                        ax[rown][coln].legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                                              fontsize=legend_font_size, ncol=1, frameon=True, bbox_to_anchor=(1, 1))
                    else:
                        ax[rown][coln].legend(legend_names, loc=legend_location_inside, fancybox=True, framealpha=0.50,
                                              fontsize=legend_font_size, ncol=1, frameon=True)

            m = list(fig.get_size_inches())

            # Set X & Y labels
            ax[rown][coln].set_xlabel(x_label)
            ax[rown][coln].set_ylabel(y_label)

            # Set visible X & Y limits
            if y_lim is not None:
                ax[rown][coln].set_ylim(y_lim)
            if x_lim is not None:
                ax[rown][coln].set_xlim(x_lim)

            # Set X & Y ticks rotation
            #         ax[rown][coln].xticks(rotation=x_tick_rotate)
            #         ax[rown][coln].yticks(rotation=y_tick_rotate)
            #         plt.xticks(rotation=x_tick_rotate)
            #         plt.yticks(rotation=y_tick_rotate)

            # Set X & Y ticks steps size
            ss = str(y_tick_stepsize)
            exx = ss.split('.')[-1]
            num_dig = len(str(int(exx)))
            y_min = round(y_min, num_dig) - y_tick_stepsize
            y_max = round(y_max, num_dig)

            # Set X & Y ticks or step names
            if x_tick_names:
                ax[rown][coln].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                # ax[rown][coln].set_xticks(x_data)
                ax[rown][coln].set_xticks(prev_x_data)
                ax[rown][coln].set_xticklabels(x_tick_names, rotation=x_tick_rotate)
            if y_tick_names:
                ax[rown][coln].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax[rown][coln].set_yticks(y_max)
                ax[rown][coln].set_yticklabels(y_tick_names, rotation=y_tick_rotate)

        # ########
        #     plt.set(ylabel=y_label)
        for axs in ax.flat:
            axs.set(xlabel=x_label)
        #         axs.set(xlabel=x_label, ylabel=y_label)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for axs in ax.flat:
            axs.label_outer()

        # Adjusting left and bottom spaces
        fig = plt.gcf()
        # fig.subplots_adjust(top=adjust_top, bottom=adjust_bottom, left=adjust_left, right=adjust_right)
        fig.subplots_adjust(bottom=adjust_bottom, left=adjust_left)

        if filename:
            #         save_format = 'pdf' 'eps' 'png'
            save_format = 'png'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
            save_format = 'eps'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
        plt.show()



    # ######################################
    # ### Drawing target-prediction binary bar graph
    # ######################################

    # ## Single line graph
    def draw_target_prediction_bargraph(self,
            x_data, y_data_list, x_label, y_label, bar_width=1.0, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=90, y_tick_rotate=0,
            custom_tick_steps=False, x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None,
            line_width=1, marker_size=3,
            legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
    ):
        # Markers, color and style
        markers, ln_colors, ln_style = self.get_plot_resources(len(y_data_list))
        binary_colors = [('green', 'orange'), ('blue', 'red')]
        # binary_legends = [('nsz', 'sz'), ('Correct', 'Missclassified')]

        print(f'All lengths: {len(markers)}, {len(ln_colors)}, {len(ln_style)}')

        # Plot the data, set the size (s), color and transparency (alpha)
        # of the points
        y_max = -99999999
        y_min = 99999999

        # ##################
        num_plots = len(y_data_list)
        fig, ax = plt.subplots(num_plots)


        # step = int(len(x_data) / 10)
        x = np.array(x_data)
        # xtcks = [i for i in range(min(x), max(x) + 1, step)]
        # xtcks.append(max(x))
        array_1d = np.ones(len(y_data_list[0]))

        for i, ydata in enumerate(y_data_list):
            ydata = np.array(ydata)
            nsz = (ydata == 0)
            sz  = (ydata == 1)
            two_colors = binary_colors[-1] if i==(num_plots-1) else binary_colors[0]
            leg_names = legend_names[-1] if i==(num_plots-1) else legend_names[0]
            ax[i].bar(x[nsz], array_1d[nsz], width=bar_width, color=two_colors[0])
            ax[i].bar(x[sz],  array_1d[sz],  width=bar_width, color=two_colors[1])
            ax[i].set_ylabel(y_label[i])
            # axs[0].xticks([],[])
            ax[i].legend(list(leg_names), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_font_size, frameon=True)

        plt.xticks(x)
        # plt.xticks(xtcks)
        plt.locator_params(axis='x', nbins=10)
        plt.ylim([0, 1])

        plt.xlabel(x_label)  # plt.xlabel('xlabel')
        # ##################

        # Label the axes and provide a title
        if title:
            ax.set_title(title)

        # Set X & Y ticks rotation
        plt.xticks(rotation=x_tick_rotate)
        plt.yticks(rotation=y_tick_rotate)

        # Adjusting left and bottom spaces
        fig = plt.gcf()
        # fig.subplots_adjust(top=adjust_top, bottom=adjust_bottom, left=adjust_left, right=adjust_right)
        fig.subplots_adjust(bottom=adjust_bottom, left=adjust_left)

        if filename:
            # save_format = 'pdf' 'eps' 'png'
            save_format = 'png'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')
            save_format = 'eps'
            plt.savefig(filename + f'.{save_format}', format=save_format, dpi=300, bbox_inches='tight')

        plt.show()
        return
	
	
	
	# ######################################
	# ### Drawing performance error bar graph
	# ######################################
	
	def draw_performance_errorbar(self,
        x_data, y_data_list, std_error_list, minmax_error_list, x_label, y_label, log_presentation=False,
        x_tick_names=None, y_tick_names=None, x_tick_rotate=45, y_tick_rotate=0,
        x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None, line_width=1,
        marker_size=4, er_line_width=2, er_cap_size=2,
        legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
        adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
        title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='pdf'
	):		
		fig, axs = plt.subplots(2, 2, sharex=True, sharey=False)
		show_std = True if len(std_error_list)>0 else False
		show_minmax = True if len(minmax_error_list[0])>0 else False
		ln_wd = er_line_width
		mrk_sz = marker_size
		cap_sz = er_cap_size
		n_cols = 0
		filename2 = f'{filename}'
		
		x = x_data
		
		#### ACCURACY
		means = y_data_list[0]
		mins = minmax_error_list[0][0]
		maxes = minmax_error_list[1][0]
		stds = std_error_list[0]
		if show_std:
			axs[0, 0].errorbar(x, means, stds, fmt='.g', ecolor='gray', lw=ln_wd+int(ln_wd/2), markersize=0, capsize=cap_sz)
			if not show_minmax:
				axs[0, 0].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		if show_minmax:
		#     print(len(x), len(means), len([0]*(means - mins)), len([0]*(maxes - means)))
		#     axs[0, 0].errorbar(x, means, [means - mins, maxes - means], label='min-max', fmt='.c', ecolor='red', lw=ln_wd)
			axs[0, 0].errorbar(x, means, [means - mins, [0]*(maxes - means)], fmt='.r', ecolor='red', lw=ln_wd, markersize=0, capsize=cap_sz, uplims=True)
			axs[0, 0].errorbar(x, means, [[0]*(means - mins), maxes - means], fmt='.g', ecolor='green', lw=ln_wd, markersize=0, capsize=cap_sz, lolims=True)
			axs[0, 0].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		# axs[0, 0].set_title(f'{mets[0]}')
		axs[0, 0].set_title(f'a) Accuracy')
		axs[0, 0].ticklabel_format(useOffset=False, axis='y')

		
		#### SENSITIVITY
		means = y_data_list[1]
		mins = minmax_error_list[0][1]
		maxes = minmax_error_list[1][1]
		stds = std_error_list[1]
		if show_std:
			axs[0, 1].errorbar(x, means, stds, fmt='.g', ecolor='gray', lw=ln_wd++int(ln_wd/2), markersize=0, capsize=cap_sz)
			if not show_minmax:
				axs[0, 1].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		if show_minmax:
		#     axs[0, 1].errorbar(x, means, [means - mins, maxes - means], fmt='.k', ecolor='red', lw=ln_wd)
			axs[0, 1].errorbar(x, means, [means - mins, [0]*(maxes - means)], fmt='.r', ecolor='red', lw=ln_wd, markersize=0, capsize=cap_sz, uplims=True)
			axs[0, 1].errorbar(x, means, [[0]*(means - mins), maxes - means], fmt='.g', ecolor='green', lw=ln_wd, markersize=0, capsize=cap_sz, lolims=True)
			axs[0, 1].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		# axs[0, 1].set_title(f'{mets[1]}')
		axs[0, 1].set_title(f'b) Sensitivity')
		axs[0, 1].ticklabel_format(useOffset=False, axis='y')

		
		#### SPECIFICITY
		means = y_data_list[2]
		mins = minmax_error_list[0][2]
		maxes = minmax_error_list[1][2]
		stds = std_error_list[2]
		if show_std:
			axs[1, 0].errorbar(x, means, stds, fmt='.g', ecolor='gray', lw=ln_wd++int(ln_wd/2), markersize=0, capsize=cap_sz)
			if not show_minmax:
				axs[1, 0].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		if show_minmax:
		#     axs[1, 0].errorbar(x, means, [means - mins, maxes - means], fmt='.k', ecolor='red', lw=ln_wd)
			axs[1, 0].errorbar(x, means, [means - mins, [0]*(maxes - means)], fmt='.r', ecolor='red', lw=ln_wd, markersize=0, capsize=cap_sz, uplims=True)
			axs[1, 0].errorbar(x, means, [[0]*(means - mins), maxes - means], fmt='.g', ecolor='green', lw=ln_wd, markersize=0, capsize=cap_sz, lolims=True)
			axs[1, 0].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		# axs[1, 0].set_title(f'{mets[2]}')
		axs[1, 0].set_title(f'c) Specificity')
		axs[1, 0].ticklabel_format(useOffset=False, axis='y')
		axs[1, 0].set_xticklabels(x, rotation=45)

		
		#### F1 Score
		filename3 = ''
		means = y_data_list[3]
		mins = minmax_error_list[0][3]
		maxes = minmax_error_list[1][3]
		stds = std_error_list[3]
		if show_std:
			n_cols += 1
			filename3 = 'std'
			axs[1, 1].errorbar(x, means, stds, label='STD', fmt='.g', ecolor='gray', lw=ln_wd+int(ln_wd/2), markersize=0, capsize=cap_sz)
			if not show_minmax:
				n_cols += 1
				axs[1, 1].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], label='Avg', fmt='ok', ecolor='black', lw=ln_wd, markersize=mrk_sz, capsize=0)
		if show_minmax:
			n_cols += 3
			filename3 = f'minmax{filename3}'
		#     axs[1, 1].errorbar(x, means, [means - mins, maxes - means], fmt='.k', ecolor='red', lw=ln_wd)
			axs[1, 1].errorbar(x, means, [means - mins, [0]*(maxes - means)], label='Min', fmt='.r', ecolor='red', lw=ln_wd, markersize=0, capsize=cap_sz, uplims=True)
			axs[1, 1].errorbar(x, means, [[0]*(means - mins), maxes - means], label='Max', fmt='.g', ecolor='green', lw=ln_wd, markersize=0, capsize=cap_sz, lolims=True) #, uplims=True, lolims=True
			axs[1, 1].errorbar(x, means, [[0]*(means - mins), [0]*(maxes - means)], label='Avg', fmt='ok', ecolor='black', lw=ln_wd, markersize=cap_sz, capsize=0)
		# axs[1, 1].set_title(f'{mets[3]}')
		axs[1, 1].set_title(f'd) F1 Score')
		axs[1, 1].ticklabel_format(useOffset=False, axis='y')
		axs[1, 1].set_xticklabels(x, rotation=45)

		
		for i, ax in enumerate(axs.flat):
		#     print(i, ax)
		#     ax.set(xlabel='Channels', ylabel='Scores')
			if i==0:
				ax.set(xlabel='', ylabel=y_label)
			if i==1:
				ax.set(xlabel='', ylabel='')
			if i==2:
				ax.set(xlabel=x_label, ylabel=y_label)
			if i==3:
				ax.set(xlabel=x_label, ylabel='')

		# # Hide x labels and tick labels for top plots and y ticks for right plots.
		# for ax in axs.flat:
		#     print(ax)
		#     ax.label_outer()

		# axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
		# handles, labels = axs.get_legend_handles_labels()
		# fig.legend(handles, labels, loc='upper center')
		fig.legend(loc='upper center', ncol=n_cols)

		if filename:
			save_format = 'png'
			plt.savefig(f'{filename2}_{filename3}.pdf', format='pdf', dpi=300, bbox_inches='tight')
			plt.savefig(f'{filename2}_{filename3}.pgf', format='pgf', dpi=300, bbox_inches='tight')
			plt.savefig(f'{filename2}_{filename3}.{save_format}', format=save_format, dpi=300, bbox_inches='tight')

		plt.show()

		return





