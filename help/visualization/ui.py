import ipywidgets as wid
from typing import List
import matplotlib.pyplot as plt
from ..models.labelling import Help
from ..utility.selection import select_cell_lines
from ..preprocess.loaders import feature_assemble
import pandas as pd
import numpy as np
import os

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
        selfeature = wid.SelectMultiple(
            options=os.listdir(feature_path),
            value=[],
            rows=rows,
            description='Features',
            disabled=False
        )
        def selfeature_changed(b):
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
        button = wid.Button(description="Loading ...")
        def on_button_clicked(b):
            with out1:
                out1.clear_output()
                display(f'Loading {selfeature.value} on label {sellabel.value}... wait util DONE...')
            with out4:
                out4.clear_output()
                features = [{'fname' : os.path.join(feature_path, fname), 'fixna': True, 'normalize': 'std'} for fname in selfeature.value]
                val.value = feature_assemble(os.path.join(label_path, sellabel.value), features=features,verbose=self.verbose)
                display('DONE')
        button.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        out3 = wid.Output()
        out4 = wid.Output()
        val = wid.ValueWidget()
        cnt = wid.VBox([wid.HBox([button, out1]), 
                        wid.HBox([selfeature, out2]), 
                        wid.HBox([sellabel, out3]), out4])
        display(cnt)
        return val
        
    def select_cell_lines(self, df: pd.DataFrame, df_map: pd.DataFrame, rows: int=5, minlines=1, line_group='OncotreeLineage', line_col='ModelID'):
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
            value=1,
            description='Min lines:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        def minline_set_changed(b):
            tl = df_map[line_col].dropna().value_counts()
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
        button = wid.Button(description="Selecting ...")
        def on_button_clicked(b):
            with out2:
                out2.clear_output()
                cell_lines = select_cell_lines(df, df_map, seltissue.value, line_group=line_group, line_col=line_col, nested = False)
                val.value = df[cell_lines]
                if save_textbox.layout == layout_visible:
                    df[cell_lines].to_csv(save_textbox.value, index=True)
                    display(f'Saved cell lines to file: {save_textbox.value}!')
        button.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        val = wid.ValueWidget()
        cnt = wid.VBox([wid.HBox([button, out1]), minline_set, seltissue, wid.HBox([saveto_but, save_textbox]), out2])
        display(cnt)
        return val
    
    def labelling(self, df: pd.DataFrame, df_map: pd.DataFrame, rows: int=5, minlines=1, line_group='OncotreeLineage', line_col='ModelID'):
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

        tl = df_map[line_col].dropna().value_counts()
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
        mode_buttons = wid.ToggleButtons(
            options=[("Two classes", False), ("Three classes", True)],
            description='',
        )
        save_textbox = wid.Text(
            value="",
            description='',
        )
        button = wid.Button(description="Labelling ...")
        def on_button_clicked(b):
            with out2:
                out2.clear_output()
                cell_lines = select_cell_lines(df, df_map, seltissue.value, line_group=line_group, line_col=line_col, nested = False)
                val.value = Help(verbose=self.verbose).labelling(df, columns=cell_lines, three_class=mode_buttons.value)
                if save_textbox.layout == layout_visible:
                    val.value.to_csv(save_textbox.value, index=True)
                    display(f'Saved cell lines to file: {save_textbox.value}!')
        button.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        val = wid.ValueWidget()
        cnt = wid.VBox([wid.HBox([button, out1]), mode_buttons, seltissue, wid.HBox([saveto_but, save_textbox]), out2])
        display(cnt)
        return val
