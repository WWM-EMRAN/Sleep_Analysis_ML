"""
File Name: HumachLab_GraphManager.py 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 12:54 pm
"""


### SRART: My modules ###
import HumachLab_Global
sys_ind = HumachLab_Global.sys_ind

if sys_ind==0:
    from HumachLab import *
elif sys_ind==1:
    from HumachLab import *
    from HumachLab.SignalProcessor.HumachLab_FeatureDetails import *
    from HumachLab.DataAnalyserAndVisualiser.HumachLab_DrawingGraphs import *
    from HumachLab.Utility.HumachLab_StaticMethods import *
elif sys_ind==2:
    from HumachLab import *
else:
    pass
### END: My modules ###




class HumachLab_GraphManager:

    def __init__(self, logger):
        self.logger = logger
        self.featDetail = HumachLab_FeatureDetails()
        self.drawGraph = HumachLab_DrawingGraphs(logger)

        return


    # ######################################
    # ### Graph for P and AUC presentation
    # ######################################

    # ## Show linechart for p and auc featurewise
    def show_featurewise_pvalue_and_auc(self, save_dir, pAuc, file_naming_detail_for_dataset, save_file_name=''):
        save_file_name2 = f'feat-wise-auc_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        ppvals = pAuc['pvalue'].values.tolist()
        aauc = pAuc['relativeAUC'].values.tolist()
        ffeat = pAuc['features'].values.tolist()

        x_data = self.featDetail.map_feature_names(ffeat)
        y_data = [ppvals, aauc]
        x_label = 'Features'
        y_label = 'p-value & AUC'

        th_vals = [0.05, 0.50]
        th_names = ['p-value Threshold', 'AUC Threshold']
        leg_names = ['p-value', 'AUC']

        plt_title = f'Total p-value and AUC of corresponding features: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4,
            marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


    # ## Show linechart for p and auc featurewise for each channel
    def show_featurewise_pvalue_and_auc_for_all_channels(self, save_dir, ch_pAuc, channels, file_naming_detail_for_dataset, save_file_name=''):
        for ch in channels:
            print(f'Channel: {ch}')
            save_file_name2 = save_file_name + f'_chn-{ch}'
            c_pAuc = ch_pAuc[ch_pAuc['channel'] == ch]
            self.show_featurewise_pvalue_and_auc(save_dir, c_pAuc, file_naming_detail_for_dataset, save_file_name2)


    # ## Show barchart for p and auc featurewise
    def show_featurewise_pvalue_and_auc_bar(self, save_dir, pAuc, file_naming_detail_for_dataset, save_file_name=''):
        save_file_name2 = f'feat-wise-auc_{file_naming_detail_for_dataset}_bar'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        ppvals = pAuc['pvalue'].values.tolist()
        aauc = pAuc['relativeAUC'].values.tolist()
        ffeat = pAuc['features'].values.tolist()

        x_data = self.featDetail.map_feature_names(ffeat)
        y_data = [ppvals, aauc]
        x_label = 'Features'
        y_label = 'p-value & AUC'

        #     matplotlib.rcParams.update({'font.size': 28})
        #     plt.xlabel('x-axis', fontsize=20)

        th_vals = [0.05, 0.50]
        th_names = ['p-value Threshold', 'AUC Threshold']
        leg_names = ['p-value', 'AUC']
        col_names = ['P Value', 'AUC']

        x_label, y_label = 'Features', 'p-value and AUC'
        # set width of bar
        barWidth = 0.25

        plt_title = f'Total p-value and AUC of corresponding features: {save_file_name}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_barplot(
            x_data, y_data, x_label, y_label, col_names, barWidth, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4, marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


    # ## Show barchart for p and auc featurewise for each channel
    def show_featurewise_pvalue_and_auc_for_all_channels_bar(self, save_dir, ch_pAuc, channels, file_naming_detail_for_dataset, save_file_name=''):
        for ch in channels:
            print(f'Channel: {ch}')
            save_file_name2 = save_file_name + f'_chn-{ch}'
            c_pAuc = ch_pAuc[ch_pAuc['channel'] == ch]
            self.show_featurewise_pvalue_and_auc_bar(save_dir, c_pAuc, file_naming_detail_for_dataset, save_file_name2)


    # ## Show linechar for p and auc together featurewise
    def show_patientwise_pvalue_and_auc(self, save_dir, all_pAuc, patients, cols, class_name, file_naming_detail_for_dataset, save_file_name=''):
        save_file_name2 = ''
        # ## Show relative p-value
        save_file_name2 = f'pat-wise-pvalue_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        # Generate Data
        fts = cols[(cols.index(class_name) + 1):]
        dat = [[] for i in range(len(fts))]
        for i in patients:
            for j in fts:
                indc = fts.index(j)
                vval = all_pAuc.loc[(all_pAuc['patient'] == i) & (all_pAuc['features'] == j)]['pvalue'].values[0]
                dat[indc].append(vval)  # df.loc[(df['Salary_in_1000']>=100)

        x_data = [ii for ii in range(1, len(patients) + 1)]  # patients
        y_data = dat
        x_label = 'Patients'
        y_label = 'p-value'

        #     leg_names = map_feature_names(cols2[5:])
        leg_names = self.featDetail.map_feature_names(cols[(cols.index(class_name) + 1):])
        th_vals = [0.05]
        th_names = ['p-value Threshold']

        plt_title = f'p-value of features for corresponding Patients: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')
        print(x_data)

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, custom_tick_steps=True, line_style=None,
            line_width=4, marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font,
            legend_location_inside='upper right',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )

        save_file_name2 = f'pat-wise-auc_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        # Generate Data
        fts = cols[(cols.index(class_name) + 1):]
        dat = [[] for i in range(len(fts))]
        for i in patients:
            for j in fts:
                indc = fts.index(j)
                vval = all_pAuc.loc[(all_pAuc['patient'] == i) & (all_pAuc['features'] == j)]['relativeAUC'].values[0]
                dat[indc].append(vval)  # df.loc[(df['Salary_in_1000']>=100)

        x_data = [ii for ii in range(1, len(patients) + 1)]  # patients
        y_data = dat
        x_label = 'Patients'
        y_label = 'AUC'

        #     leg_names = map_feature_names(cols2[5:])
        leg_names = self.featDetail.map_feature_names(cols[(cols.index(class_name) + 1):])
        th_vals = [0.50]
        th_names = ['AUC Threshold']

        plt_title = f'AUC of features for corresponding Patients: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, custom_tick_steps=True, line_style=None,
            line_width=4, marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font,
            legend_location_inside='lower left',  # 'best'
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


    # ## Show linechar for p and auc together featurewise for all channels
    def show_patientwise_pvalue_and_auc_for_all_channels(self, save_dir, ch_all_pAuc, channels, patients, cols, class_name, file_naming_detail_for_dataset, save_file_name=''):
        for i in range(len(channels)):
            ch = channels[i]
            print(f'Channel: {ch}')
            save_file_name2 = save_file_name + f'_chn-{ch}'
            all_pAuc = ch_all_pAuc[ch_all_pAuc['channel'] == ch]
            self.show_patientwise_pvalue_and_auc(save_dir, all_pAuc, patients, cols, class_name, file_naming_detail_for_dataset, save_file_name2)


    # ## Show linechar for patientwise p and auc together featurewise
    def show_patientwise_pvalue_and_auc_feat(self, save_dir, all_pAuc, channels, patients, file_naming_detail_for_dataset, save_file_name=''):
        save_file_name2 = ''
        # ## Show relative p-value
        save_file_name2 = f'pat-wise_ft-pvalue_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        # Generate Data
        chnls = copy.deepcopy(channels)
        dat = [[] for i in range(len(chnls))]
        for i in patients:
            for j in chnls:
                indc = chnls.index(j)
                vval = all_pAuc.loc[(all_pAuc['patient'] == i) & (all_pAuc['channel'] == j)]['pvalue'].values[0]
                dat[indc].append(vval)  # df.loc[(df['Salary_in_1000']>=100)

        x_data = [ii for ii in range(1, len(patients) + 1)]  # patients
        y_data = dat
        x_label = 'Patients'
        y_label = 'p-value'

        leg_names = chnls  # map_feature_names(channels)
        th_vals = [0.05]
        th_names = ['p-value Threshold']

        plt_title = f'p-value of features for corresponding Patients: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
        plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
        show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font #round(0.80*graph_font)
        self.drawGraph.draw_lineplot(
        x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None, x_tick_rotate=45, y_tick_rotate=0,
        x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4, marker_size=12,
        legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='upper right',
        adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
        title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )

        # ## Show relative AUC
        save_file_name2 = f'pat-wise_ft-auc_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        # Generate Data
        chnls = copy.deepcopy(channels)
        dat = [[] for i in range(len(channels))]
        for i in patients:
            for j in chnls:
                indc = chnls.index(j)
                #             print('=======> ', i, j, indc)
                vval = all_pAuc.loc[(all_pAuc['patient'] == i) & (all_pAuc['channel'] == j)]['relativeAUC'].values[0]
                #             print('=======> ', i, j, indc, vval)
                dat[indc].append(vval)  # df.loc[(df['Salary_in_1000']>=100)

        x_data = [ii for ii in range(1, len(patients) + 1)]  # patients
        y_data = dat
        x_label = 'Patients'
        y_label = 'AUC'

        leg_names = chnls  # map_feature_names(cols2[5:])
        th_vals = [0.50]
        th_names = ['AUC Threshold']

        plt_title = f'AUC of features for corresponding Patients: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, custom_tick_steps=True, line_style=None,
            line_width=4, marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font,
            legend_location_inside='lower left',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


    # ## Show linechar for patientwise p and auc together featurewise for all channels
    def show_patientwise_pvalue_and_auc_for_all_features(self, save_dir, ch_all_pAuc, channels, patients, class_name, file_naming_detail_for_dataset, cols, save_file_name=''):
        feats = cols[(cols.index(class_name) + 1):]
        ffts = self.featDetail.map_feature_names(feats)

        for i in range(len(feats)):
            ff = feats[i]
            print(f'Feature: {ff}')
            save_file_name2 = save_file_name + f'_fts-{ffts[i]}'
            all_pAuc = ch_all_pAuc[ch_all_pAuc['features'] == ff]
            self.show_patientwise_pvalue_and_auc_feat(save_dir, all_pAuc, channels, patients, file_naming_detail_for_dataset, save_file_name2)


    # ## Show linechar for channelwise p and auc together featurewise
    def show_channelwise_pvalue_and_auc(self, save_dir, com_vals, file_naming_detail_for_dataset, save_file_name=''):
        save_file_name2 = f'ch-wise-pvalue_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        feats = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['features'].values.tolist())
        unique_chn = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['channel'].values.tolist())

        # Generate Data
        dat = []
        for ff in feats:
            pv = (com_vals[com_vals['features'] == ff])['pvalue'].values.tolist()
            dat.append(pv)

        x_data = unique_chn
        y_data = dat
        x_label = 'Channels'
        y_label = 'p-value'

        th_vals = [0.05]
        th_names = ['p-value Threshold']
        leg_names = self.featDetail.map_feature_names(feats)

        plt_title = f'p-value of corresponding channels: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = round(0.80 * graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4,
            marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )

        # ## Plot relativeAUC ##
        save_file_name2 = f'ch-wise-auc_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        feats = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['features'].values.tolist())
        unique_chn = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['channel'].values.tolist())

        th_vals = []
        th_names = []

        # Generate Data
        dat = []  # [[] for i in range(feats)]
        for ff in feats:
            auc = (com_vals[com_vals['features'] == ff])['relativeAUC'].values.tolist()
            dat.append(auc)
        #         avg = sum(auc) / len(auc)
        #         print(avg)
        #         th_vals.append(avg)

        x_data = unique_chn
        y_data = dat
        x_label = 'Channels'
        y_label = 'AUC'

        th_vals = [0.50]
        th_names = ['AUC Threshold']
        leg_names = self.featDetail.map_feature_names(feats)
        #     th_names = [f'Avg_{ff}' for ff in leg_names]

        plt_title = f'AUC of corresponding channels: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4,
            marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


    # ## Show linechar for channelwise p and auc together featurewise with average value
    def show_channelwise_pvalue_and_auc_avg(self, save_dir, com_vals, file_naming_detail_for_dataset, save_file_name=''):
        save_file_name2 = f'ch-wiseavg-pvalue_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        feats = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['features'].values.tolist())
        unique_chn = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['channel'].values.tolist())

        # Generate Data
        dat = []
        for ff in feats:
            pv = (com_vals[com_vals['features'] == ff])['pvalue'].values.tolist()
            dat.append(pv)

        x_data = unique_chn
        y_data = dat
        x_label = 'Channels'
        y_label = 'p-value'

        th_vals = [0.05]
        th_names = ['p-value Threshold']
        leg_names = self.featDetail.map_feature_names(feats)

        plt_title = f'p-value of corresponding channels: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = round(0.80 * graph_font)
        self.drawGraph.draw_lineplot(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4,
            marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


        # ## Plot relativeAUC ##
        save_file_name2 = f'ch-wiseavg-auc_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        feats = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['features'].values.tolist())
        unique_chn = HumachLab_StaticMethods.get_unique_items_in_the_list(com_vals['channel'].values.tolist())

        th_vals = []
        th_names = []

        # Generate Data
        dat = []  # [[] for i in range(feats)]
        for ff in feats:
            auc = (com_vals[com_vals['features'] == ff])['relativeAUC'].values.tolist()
            dat.append(auc)
            avg = sum(auc) / len(auc)
            th_vals.append(avg)

        x_data = unique_chn
        y_data = dat
        x_label = 'Channels'
        y_label = 'AUC'

        #     th_vals = [0.50]
        #     th_names = ['AUC Threshold']
        leg_names = self.featDetail.map_feature_names(feats)
        th_names = [f'Avg {ff}' for ff in leg_names]

        plt_title = f'AUC of corresponding channels: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        print(file_name, f'({save_file_name})')

        # Graph settings
        graph_font = 30
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_lineplot_avg(
            x_data, y_data, x_label, y_label, log_presentation=False, x_tick_names=None, y_tick_names=None,
            x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=4,
            marker_size=12,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=th_vals, threshod_names=th_names, filename=file_name
        )


    # ## Show error grpah for sz-non sz mean-std
    def show_feature_mean_and_standardDeviation_for_seiz_nonseiz(self, save_dir, feat_mean_std, file_naming_detail_for_dataset, save_file_name='', chan=None):
        save_file_name2 = f'feat-meanstd_{file_naming_detail_for_dataset}_errbar'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        mean_vals = []
        mean_vals.append(feat_mean_std['non_siez_mean'].values.tolist())
        mean_vals.append(feat_mean_std['siez_mean'].values.tolist())
        std_vals = []
        std_vals.append(feat_mean_std['non_siez_std'].values.tolist())
        std_vals.append(feat_mean_std['siez_std'].values.tolist())
        ffeat = feat_mean_std['features'].values.tolist()

        x_data = self.featDetail.map_feature_names(ffeat)
        y_data = mean_vals
        x_error = [[] for xx in range(len(y_data))]
        y_error = std_vals
        x_label = 'Features'
        y_label = 'Mean-Standard Deviation (log)'
        x_tick_names = x_data

        leg_names = ['Non-seizure', 'Seizure']

        ch = f': {chan}' if chan else ''
        plt_title = f'Feature Mean-Standard Deviation {ch}'

        # plt_title = f'Feature Mean-Standard Deviation: {save_file_name2}'
        file_name = f'{save_dir}{save_file_name2}'

        # print('Data description', x_data, y_data, x_error, y_error)
        print(file_name, f'({save_file_name})')

        # Graph settings
        #     fig_size=(16, 12)
        graph_font = 40
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(12, 8), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        # Graph drawing
        legend_font = graph_font  # round(0.80*graph_font)
        self.drawGraph.draw_errorbar(
            x_data, y_data, x_error, y_error, x_label, y_label, log_presentation=True,
            x_tick_names=x_tick_names, y_tick_names=None, x_tick_rotate=45, y_tick_rotate=0,
            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None, line_width=0,
            marker_size=16, er_line_width=6,
            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font, legend_location_inside='best',
            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
            title=None, threshod_levels=None, threshod_names=None, filename=file_name, save_format='png'
        )
        return


    # ## Show error grpah for sz-non sz mean-std for all channels
    def show_feature_mean_and_standardDeviation_for_seiz_nonseiz_for_all_channels(self, save_dir, chan_feat_mean_std, channels, file_naming_detail_for_dataset, save_file_name=''):
        for i, ch in enumerate(channels):
            print(f'Channel: {ch}')
            save_file_name2 = save_file_name + f'_chn-{ch}'
            feat_meanstd = chan_feat_mean_std[chan_feat_mean_std['channel'] == ch]
            self.show_feature_mean_and_standardDeviation_for_seiz_nonseiz(save_dir, feat_meanstd, file_naming_detail_for_dataset, save_file_name=save_file_name2, chan=ch)

        return



    def show_feature_mean_and_standardDeviation_for_seiz_nonseiz_for_all_channels_all_graphs(self, save_dir, chan_feat_mean_std, channels, file_naming_detail_for_dataset, graph_dim=[3, 3], save_file_name=''):

        save_file_name2 = f'feat-meanstd_{file_naming_detail_for_dataset}_errbar_all'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        # ###########
        ffeat = HumachLab_StaticMethods.get_unique_items_in_the_list(chan_feat_mean_std['features'].values.tolist())
        x_data = self.featDetail.map_feature_names(ffeat)
        x_tick_names = x_data

        feat_meanstd = chan_feat_mean_std[chan_feat_mean_std['channel'] == 'Fp1']
        y_data_list = []
        y_data_list.append(feat_meanstd['non_siez_mean'].values.tolist())
        y_data_list.append(feat_meanstd['siez_mean'].values.tolist())

        x_label = 'Features'
        y_label = 'Mean-Standard Deviation (log)'
        y_label = 'Mean-SD (log)'
        x_tick_names = x_data
        leg_names = ['Non-seizure', 'Seizure']
        graph_title = f'Mean-SD variations for corresponding features in different channels'


        filename = f'{save_dir}{save_file_name2}'

        #     print(x_data, y_data, x_error, y_error)
        print(filename, f'({save_file_name})', f'({save_file_name2})')

        # ###########
        # Graph settings
        #     fig_size=(16, 12)
        graph_font = 40
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(12, 4 * graph_dim[0]), font_size=graph_font,
            bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)
        legend_font = graph_font

        self.drawGraph.draw_errorbar_group(
                            x_data, y_data_list, x_label, y_label, chan_feat_mean_std,
                            channels, graph_dim, log_presentation=True,
                            x_tick_names=None, y_tick_names=None, x_tick_rotate=45, y_tick_rotate=0,
                            x_tick_stepsize=1, y_tick_stepsize=0.1, x_lim=None, y_lim=None, line_style=None,
                            line_width=0, marker_size=16, er_line_width=6,
                            legend_names=leg_names, legend_outside=False, legend_font_size=legend_font,
                            legend_location_inside='best',
                            adjust_top=0.0, adjust_bottom=0.05, adjust_left=0.0, adjust_right=0.0,
                            title=None, threshod_levels=None, threshod_names=None, filename=None, save_format='png'
                            )
        return




    # def draw_target_prediction_binarygraph(save_dir, t, p, f, tot_chns, tot_pats, tot_recs, tot_feats, save_file_name=''):
    def show_target_prediction_bargraph(self, save_dir, true_y, pred_y, final_res, file_naming_detail_for_dataset, save_file_name=''):

        save_file_name2 = f'tp_bin_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        x_data = range(len(final_res))
        y_data_list = []
        y_data_list.append(true_y)
        y_data_list.append(pred_y)
        y_data_list.append(final_res)


        x_label = 'Datapoints'
        y_label = ['Target', 'Prediction', 'Result']
        x_tick_names = x_data
        leg_names = [('Non-seizure', 'Seizure'), ('Correct', 'Missclassified')]
        graph_title = f'Mean-SD variations for corresponding features in different channels'


        filename = f'{save_dir}{save_file_name2}'

        #     print(x_data, y_data, x_error, y_error)
        print(filename, f'({save_file_name})', f'({save_file_name2})')

        # ###########
        # Graph settings
        #     fig_size=(16, 12)
        graph_font = 40
        self.drawGraph.setup_mpl_graph_properties(
            plot_style='seaborn-whitegrid', fig_size=(12, 8), font_size=graph_font, bg_face_color='white',
            show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

        self.drawGraph.draw_target_prediction_bargraph(
                                        x_data, y_data_list, x_label, y_label, bar_width=1.0,
                                        log_presentation=False, x_tick_names=None, y_tick_names=None,
                                        x_tick_rotate=90, y_tick_rotate=0,
                                        custom_tick_steps=False, x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None,
                                        y_lim=None, line_style=None,
                                        line_width=1, marker_size=3,
                                        legend_names=leg_names, legend_outside=False, legend_font_size='medium',
                                        legend_location_inside='best',
                                        adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
                                        title=None, threshod_levels=None, threshod_names=None, filename=filename,
                                        save_format='pdf'
                                        )
        return

    # ## Show performance errorbar for all channels
    def show_performance_minmaxstd_errorbar(self, save_dir, ts_grp, file_naming_detail_for_dataset='', save_file_name=''):
        save_file_name2 = f'performance-errorbar_{file_naming_detail_for_dataset}'
        if len(save_file_name) > 0:
            save_file_name2 += f'_{save_file_name}'

        mets = ['acc', 'rec', 'spe', 'f1s']
		cats = ['mean', 'min', 'max', 'std']
        x_data = ts_grp.index.values
        y_data = [ts_grp[mets[0]][cats[0]], ts_grp[mets[1]][cats[0]], ts_grp[mets[2]][cats[0]], ts_grp[mets[3]][cats[0]]]
		std_error_list = [ts_grp[mets[0]][cats[3]], ts_grp[mets[1]][cats[3]], ts_grp[mets[2]][cats[3]], ts_grp[mets[3]][cats[3]]]
		minmax_error_list = [ [ts_grp[mets[0]][cats[1]], ts_grp[mets[1]][cats[1]], ts_grp[mets[2]][cats[1]], ts_grp[mets[3]][cats[1]]], 
							 [ts_grp[mets[0]][cats[2]], ts_grp[mets[1]][cats[2]], ts_grp[mets[2]][cats[2]], ts_grp[mets[3]][cats[2]]] ]
		x_label = 'Channels'
		y_label = 'Scores'

	#     th_vals = [0.05, 0.50]
	#     th_names = ['p-value Threshold', 'AUC Threshold']
	#     leg_names = ['p-value', 'AUC']

	#     plt_title = f'Total p-value and AUC of corresponding features: {save_file_name2}'
		file_name = f'{save_dir}{save_file_name2}'

		print(file_name, f'({save_file_name})')

		# Graph settings
		graph_font = 30
	    self.drawGraph.setup_mpl_graph_properties(
			plot_style='seaborn-whitegrid', fig_size=(16, 12), font_size=graph_font, bg_face_color='white',
			show_grid=False, border_top=True, border_bottom=True, border_left=True, border_right=True)

		# Graph drawing
		legend_font = graph_font  # round(0.80*graph_font)
	    self.drawGraph.draw_performance_errorbar(
			x_data, y_data, std_error_list, minmax_error_list, x_label, y_label, log_presentation=False,
			x_tick_names=None, y_tick_names=None, x_tick_rotate=45, y_tick_rotate=0,
			x_tick_stepsize=1, y_tick_stepsize=0.5, x_lim=None, y_lim=None, line_style=None, line_width=1,
			marker_size=4, er_line_width=2, er_cap_size=2,
			legend_names=None, legend_outside=False, legend_font_size='medium', legend_location_inside='best',
			adjust_top=0.0, adjust_bottom=0.0, adjust_left=0.0, adjust_right=0.0,
			title=None, threshod_levels=None, threshod_names=None, filename=file_name, save_format='pdf')
        return


