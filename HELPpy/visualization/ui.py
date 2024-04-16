import ipywidgets as wid
from typing import List
import matplotlib.pyplot as plt
from ..models.labelling import labelling
from ..utility.selection import select_cell_lines, delrows_with_nan_percentage
from ..preprocess.loaders import feature_assemble
import pandas as pd
import numpy as np
import os
from ipyfilechooser import FileChooser
from IPython.display import HTML as html_print
from IPython.display import display

def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)

def print_color(t):
    display(html_print(' '.join([cstr(ti, color=ci) for ti,ci in t])))

class Help_Dashboard():
    def __init__(self, verbose: bool = False):
        """
        Initialize the Help Dashboard.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose messages during processing (default is False).
        """
        self.verbose = verbose

    def process_features(self, label_path: str = ".", feature_path: str = ".", rows: int = 5):
        """
        Create an interactive widget for processing features.

        Parameters
        ----------
        label_path : str, optional
            Path to the label file (default is ".").
        feature_path : str, optional
            Path to the feature files (default is ".").
        rows : int, optional
            Number of rows to display in the widget (default is 5).

        Returns
        -------
        ipywidgets.ValueWidget
            Widget containing the assembled features and labels DataFrames.
        """
        layout_hidden  = wid.Layout(visibility = 'hidden')
        layout_visible = wid.Layout(visibility = 'visible')
        selfeature = wid.SelectMultiple(
            options=os.listdir(feature_path),
            value=[],
            rows=rows,
            description='Features',
            disabled=False
        )
        def selfeature_changed(b):
            save_textbox.value = f"data_{'_'.join([s.split('.')[0] for s in selfeature.value])}.csv"
            with out2:
                out2.clear_output()
                display(selfeature.value)
        selfeature.observe(selfeature_changed, names='value')
        sellabel = wid.Select(
            options=os.listdir(label_path),
            value=None,
            rows=rows,
            description='Labels',
            disabled=False
        )
        def sellabel_changed(b):
            with out3:
                out3.clear_output()
                display(sellabel.value)
        sellabel.observe(sellabel_changed, names='value')
        saveto_but = wid.ToggleButton(value=True,
                description='Save to:',
                disabled=False,
                button_style='success', # 'success', 'info', 'warning', 'danger' or ''
        )
        def saveto_but_change(b):
            out2.clear_output()
            if save_textbox.layout == layout_hidden:
                save_textbox.layout = layout_visible
            else:
                save_textbox.layout = layout_hidden
        saveto_but.observe(saveto_but_change)
        save_textbox = wid.Text(
            value="",
            description='',
        )
        save_textbox.layout = layout_visible
        button = wid.Button(description="Loading ...")
        def on_button_clicked(b):
            with out1:
                out1.clear_output()
                display(f'Loading/Processing {selfeature.value} with label {sellabel.value} ... wait until DONE...')
            with out4:
                out4.clear_output()
                features = [{'fname' : os.path.join(feature_path, fname), 'fixna': True, 'normalize': 'std'} for fname in selfeature.value]
                val.value = feature_assemble(os.path.join(label_path, sellabel.value), features=features,verbose=self.verbose)
                if save_textbox.layout == layout_visible and save_textbox.value != '':
                    pd.merge(val.value[0], val.value[1], left_index=True, right_index=True, how='outer').to_csv(save_textbox.value, index=True)
                    display(f'Saved dataset to file: {save_textbox.value}.')
                display('DONE')
        button.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        out3 = wid.Output()
        out4 = wid.Output()
        val = wid.ValueWidget()
        cnt = wid.VBox([wid.HBox([button, out1]), 
                        wid.HBox([selfeature, out2]), 
                        wid.HBox([sellabel, out3]), wid.HBox([saveto_but, save_textbox]), out4])
        display(cnt)
        return val
        
    def select_cell_lines(self, df: pd.DataFrame, df_map: pd.DataFrame, outvar: object, rows: int=5, minlines=1, line_group='OncotreeLineage', line_col='ModelID', verbose=False):
        """
        Generate an interactive widget for labeling cell lines based on specified criteria.

        Parameters
        ----------
        df : pd.DataFrame
            The main DataFrame containing the data.
        df_map : pd.DataFrame
            A DataFrame used for mapping data.
        rows : int, optional
            The number of rows to display in the widget for selecting tissues (default is 5).
        minlines : int, optional
            Minimum number of cell lines for tissue/lineage to be considered (default is 1).
        line_group : str, optional
            The column in 'df_map' to use for tissue selection (default is 'OncotreeLineage').
        line_col : str, optional
            The column in 'df_map' to use for line selection (default is 'ModelID').

        Returns
        -------
        ipywidgets.ValueWidget
            Widget containing the labeled cell lines.
        """

        tl = df_map[line_group].dropna().value_counts()
        tissue_list = [x[0] for x in list(filter(lambda x: x[1] >= minlines, zip(tl.index.values.astype(str) , tl.values)))]
        # tissue_list = (np.unique(df_map[line_col].dropna().values.astype(str)))
        layout_hidden  = wid.Layout(visibility = 'hidden')
        layout_visible = wid.Layout(visibility = 'visible')
        seltissue = wid.SelectMultiple(
            options=tissue_list,
            value=[],
            rows=rows,
            description='Tissues',
            disabled=False
        )
        def seltissue_changed(b):
            save_textbox.value = f"{'_'.join([s.replace(' ','-').replace('/','-') for s in seltissue.value])}.csv"
            with out1:
                out1.clear_output()
                display(seltissue.value)
        seltissue.observe(seltissue_changed, names='value')
        minline_set = wid.SelectionSlider(
            options=range(1, 100),
            value=minlines,
            description='Min lines:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        def minline_set_changed(b):
            tl = df_map[line_group].dropna().value_counts()
            tissue_list = [x[0] for x in list(filter(lambda x: x[1] >= minline_set.value, zip(tl.index.values.astype(str) , tl.values)))]
            seltissue.options = tissue_list
            seltissue.value=[]
        minline_set.observe(minline_set_changed, names='value')
        saveto_but = wid.ToggleButton(value=False,
                description='Save to:',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        def saveto_but_change(b):
            out2.clear_output()
            if save_textbox.layout == layout_hidden:
                save_textbox.layout = layout_visible
            else:
                save_textbox.layout = layout_hidden
        saveto_but.observe(saveto_but_change)
        save_textbox = wid.Text(
            value="",
            description='',
        )
        save_textbox.layout = layout_visible
        butsel = wid.Button(description="Selecting ...")
        def on_button_clicked(b):
            with out1:
                out1.clear_output()
                display(f'Saving cell lines {seltissue.value} in file {save_textbox.value}... wait until DONE...')
            with out2:
                out2.clear_output()
                cell_lines = select_cell_lines(df, df_map, seltissue.value, line_group=line_group, line_col=line_col, nested = False, verbose=verbose)
                val.value = df[cell_lines]
                globals()[outvar] = val.value   # set output variable
                if save_textbox.layout == layout_visible and save_textbox.value != '':
                    val.value.to_csv(save_textbox.value, index=True)
                    display(f'Saved cell lines to file: {save_textbox.value}.')
                display("DONE!")
        butsel.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        val = wid.ValueWidget()
        cnt = wid.VBox([wid.HBox([butsel, out1]), minline_set, seltissue, wid.HBox([saveto_but, save_textbox]), out2])
        display(cnt)
        return val
    
    def labelling(self, path: str=os.getcwd(), filename: str='', modelname:str='', rows: int=5, minlines=10, percent = 100.0, line_group='OncotreeLineage', line_col='ModelID', verbose=False):
        """
        Generate an interactive widget for labeling cell lines based on specified criteria.

        Parameters
        ----------
        path : str
            path for input file loading.
        filename : str
            name of CRISPR effect input file.
        modelname : str
            name of Model input file.
        rows : int, optional
            The number of rows to display in the widget for selecting tissues (default is 5).
        minlines : int, optional
            Minimum number of cell lines for tissue/lineage to be considered (default is 1).
        line_group : str, optional
            The column in 'df_map' to use for tissue selection (default is 'OncotreeLineage').
        line_col : str, optional
            The column in 'df_map' to use for line selection (default is 'ModelID').

        Returns
        -------
        ipywidgets.ValueWidget
            Widget containing the labeled cell lines.
        """
        df_map = None
        df = None
        df_orig = None
        val = wid.ValueWidget()
        val.value = pd.DataFrame(), df, df_orig, df_map 
        tissue_list = []
        #tissue_list = [tissue for tissue in np.unique(df_map[line_group].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[line_group] == tissue][line_col].values)) >= 1]
        layout_hidden  = wid.Layout(visibility = 'hidden')
        layout_visible = wid.Layout(visibility = 'visible')

        nanrem_set = wid.SelectionSlider(
            options=range(0, 101),
            value=int(percent),
            description='Nan %:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            tooltip='set percentage of nan allowed in genes',
        )
        def nanrem_set_changed(b):
            #global df, df_orig
            try:
                df = delrows_with_nan_percentage(val.value[2], perc=float(nanrem_set.value))
                df_orig = val.value[2]
                val.value = val.value[0], df, val.value[2]
                with out3:
                    out3.clear_output()
                    print_color(((f'Removed {len(df_orig)-len(df)}/{len(df_orig)} rows (with at least {nanrem_set.value}% NaN)', 'green'),))
            except:
                pass 
        nanrem_set.observe(nanrem_set_changed, names='value')

        minline_set = wid.SelectionSlider(
            options=range(1, 100),
            value=minlines,
            description='Min lines:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            tooltip='set minimum number of lines for the tissue',
        )
        def minline_set_changed(b):
            df = val.value[1]
            df_map = val.value[3]
            try:
                tissue_list = [tissue for tissue in np.unique(df_map[line_group].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[line_group] == tissue][line_col].values)) >= minline_set.value]
                seltissue.options = ['__all__'] +  tissue_list
                seltissue.value=['__all__']
            except:
                pass
        minline_set.observe(minline_set_changed, names='value')
        seltissue = wid.SelectMultiple(
            options=['__all__'] + tissue_list if tissue_list != [] else [],
            value=['__all__'] if tissue_list != [] else [],
            rows=rows,
            description='Tissues',
            disabled=False
        )
        def seltissue_changed(b):
            if seltissue.value != ():
                if seltissue.value == ('__all__',):
                    save_textbox.value = f"PanTissue.csv"
                else:
                    save_textbox.value = f"{'_'.join([s.replace(' ','-').replace('/','-') for s in seltissue.value if s != '__all__'])}.csv"
                with out1:
                    out1.clear_output()
                    display(seltissue.value)
        seltissue.observe(seltissue_changed, names='value')
        saveto_but = wid.Button(description="Save ...", button_style='primary')
        def on_savebutton_clicked(b):
            if save_textbox.value != '':
                try:
                    val.value[0].to_csv(save_textbox.value, index=True)
                    with out4:
                        out4.clear_output()
                        print_color(((f'Saved dataset to file: {save_textbox.value}.', 'green'),))
                except:
                    with out4:
                        out4.clear_output()
                        print_color(((f'Problem saving label file (maybe empty)!', 'red'),))
            else:
                with out4:
                    out4.clear_output()
                    print_color(((f'Set a non empty filename!', 'red'),))
        saveto_but.on_click(on_savebutton_clicked)

        mode_buttons = wid.ToggleButtons(
            options=["E|NE", "E|aE|sNE", "E|(aE|sNE)"],
            button_style='success',
            description='',
            tooltips=['2 classes (one division)', '3 classes (one division)', '3 classes (two-times subdivision)'],
        )
        selmode_button = wid.Checkbox(
            value=False,
            description='Nested',
            disabled=False,
            indent=False
        )
        save_textbox = wid.Text(
            value="",
            description='',
        )
        save_textbox.layout = layout_visible
        button = wid.Button(description="Apply ...", button_style='primary')
        def on_button_clicked(b):
            df = val.value[1]
            df_map = val.value[3]
            with out1:
                out1.clear_output()
                print_color(((f'Labelling {len(df)} genes of {seltissue.value} ...', 'orange'),))
            with out2:
                out2.clear_output()
                if seltissue.value == ('__all__',):
                    selector = [x for x in seltissue.options if x != '__all__']
                else:
                    selector = [x for x in seltissue.value if x != '__all__']
                #display(selector)
                cell_lines = select_cell_lines(df, df_map, selector, line_group=line_group, line_col=line_col, 
                                               nested = selmode_button.value, verbose=verbose)
                if mode_buttons.value == "E|(aE|sNE)":
                    mode = 'two-by-two' 
                    nclasses = 3
                    labelnames = {0: 'E', 1: 'aE', 2: 'sNE'}
                else:
                    mode = 'flat-multi' 
                    if mode_buttons.value == "E|NE":
                        nclasses = 2
                        labelnames = {0: 'E', 1: 'NE'}
                    else:
                        nclasses = 3
                        labelnames = {0: 'E', 1: 'aE', 2: 'sNE'}
                val.value = labelling(df, columns=cell_lines, mode=mode, n_classes=nclasses, labelnames=labelnames, verbose=verbose), df, val.value[2], val.value[3]
            with out1:
                out1.clear_output()
                print_color(((f'DONE', 'green'),))
        button.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        out3 = wid.Output()
        out4 = wid.Output()

        # Create and display a FileChooser widget
        if filename != '':
            fc1 = FileChooser(path, title='Choose CRISPR effect file', filter='*.csv', filename=filename, select_default=True)
            try:
                df_orig = pd.read_csv(fc1.selected).rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
                df = delrows_with_nan_percentage(df_orig, perc=float(nanrem_set.value))
                val.value = val.value[0],df, df_orig, val.value[3]
                with out3:
                    out3.clear_output()
                    print_color(((f'Removed {len(df_orig)-len(df)}/{len(df_orig)} rows (with at least {nanrem_set.value}% NaN)', 'green'),))
                try:
                    if len(np.unique(df_map[line_group].dropna().values)) > 0:
                        seltissue.options = ['__all__'] + [tissue for tissue in np.unique(df_map[line_group].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[line_group] == tissue][line_col].values)) >= 1]
                        seltissue.value=['__all__']
                except:
                    pass
                fc1._label.value = fc1._LBL_TEMPLATE.format(f'{fc1.selected}', 'green')
            except:
                fc1._label.value = fc1._LBL_TEMPLATE.format(f'Problem loading {fc1.selected} file ...', 'red') 
        else:
            fc1 = FileChooser(path, title='Choose CRISPR effect file', filter='*.csv')
        def fc1_change_title(fc1):
            #global df_map, df, df_orig
            try:
                df_orig = pd.read_csv(fc1.selected).rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
                df = delrows_with_nan_percentage(df_orig, perc=float(nanrem_set.value))
                val.value = val.value[0],df ,df_orig, val.value[3]
                with out3:
                    out3.clear_output()
                    print_color(((f'Removed {len(df_orig)-len(df)}/{len(df_orig)} rows (with at least {nanrem_set.value}% NaN)', 'green'),))
                try:
                    if len(np.unique(df_map[line_group].dropna().values)) > 0:
                        seltissue.options = ['__all__'] + [tissue for tissue in np.unique(df_map[line_group].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[line_group] == tissue][line_col].values)) >= 1]
                        seltissue.value=['__all__']
                except:
                    pass 
                fc1._label.value = fc1._LBL_TEMPLATE.format(f'{fc1.selected}', 'green')
            except:
                fc1._label.value = fc1._LBL_TEMPLATE.format(f'Problem loading {fc1.selected} file ...', 'red') 

        fc1.register_callback(fc1_change_title)
        if modelname != '':
            fc2 = FileChooser(path, title='Choose Model file', filter='*.csv', filename=modelname, select_default=True)
            try:
                df_map = pd.read_csv(fc2.selected)
                df = val.value[1]
                val.value = val.value[0], val.value[1] ,val.value[2], df_map
                try:
                    if len(np.unique(df_map[line_group].dropna().values)) > 0:
                        seltissue.options = ['__all__'] + [tissue for tissue in np.unique(df_map[line_group].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[line_group] == tissue][line_col].values)) >= 1]
                        seltissue.value=['__all__']
                except:
                    pass 
                fc2._label.value = fc2._LBL_TEMPLATE.format(f'{fc2.selected}', 'green')
            except:
                fc2._label.value = fc2._LBL_TEMPLATE.format(f'Problem loading {fc2.selected} file ...', 'red') 
        else:
            fc2 = FileChooser(path, title='Choose Model file', filter='*.csv')
        def fc2_change_title(fc2):
            #global df_map, df
            try:
                df_map = pd.read_csv(fc2.selected)
                df = val.value[1]
                val.value = val.value[0], val.value[1] ,val.value[2], df_map
                try:
                    if len(np.unique(df_map[line_group].dropna().values)) > 0:
                        seltissue.options = ['__all__'] + [tissue for tissue in np.unique(df_map[line_group].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[line_group] == tissue][line_col].values)) >= 1]
                        seltissue.value=['__all__']
                except:
                    pass
                fc2._label.value = fc2._LBL_TEMPLATE.format(f'{fc2.selected}', 'green')
            except:
                fc2._label.value = fc2._LBL_TEMPLATE.format(f'Problem loading {fc2.selected} file ...', 'red') 
        fc2.register_callback(fc2_change_title)
        Vb1 = wid.VBox([#wid.HTML(value = f"<b>NaN removal:</b>"), 
            nanrem_set, out3])
        #Vb1.box_style()
        Vb2 = wid.VBox([fc1, fc2])
        Vb3 = wid.VBox([#wid.HTML(value = f"<b>Line filtering:</b>"), 
            minline_set, wid.HBox([seltissue, selmode_button])])
        Vb4 = wid.VBox([wid.VBox([mode_buttons, #wid.HTML(value = f"<b>Labelling:</b>"), 
                        wid.HBox([button, out1])]),  out2])
        Vb5 = wid.VBox([#wid.HTML(value = f"<b>Saving:</b>"), 
                        wid.HBox([saveto_but, save_textbox, out4])])
        tabs = wid.Tab((Vb1,Vb2, Vb3 ,Vb4, Vb5))
        tabs.set_title(0, 'NaN removal')
        tabs.set_title(1, 'File input')
        tabs.set_title(2, 'Line filtering')
        tabs.set_title(3, 'Labelling')
        tabs.set_title(4, 'Saving')
        #cnt = wid.VBox([Vb1, fc1, fc2, Vb2, Vb3 ,Vb4, Vb5, out2])
        display(tabs) #display(cnt)
        return val
