import ipywidgets as wid
from typing import List
import matplotlib.pyplot as plt
from ..models.labelling import Help
from ..utility.selection import select_cell_lines, feature_assemble
import pandas as pd
import numpy as np
import os

class Help_Dashboard():
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def process_features(self, label_path: str = ".", feature_path: str = ".", rows: int = 5):
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
        
    def labelling(self, df: pd.DataFrame, df_map: pd.DataFrame, rows: int=5, minlines=1, column='lineage1'):
        """
        Generate an interactive widget for labeling cell lines based on specified criteria.

        Parameters:
        - df (pd.DataFrame): The main DataFrame containing the data.
        - df_map (pd.DataFrame): A DataFrame used for mapping data.
        - rows (int): The number of rows to display in the widget for selecting tissues (default is 5).
        - minlines (in): Minimum number of cell lines for tissue/lineage to be considered
        - column (str): The column in 'df_map' to use for tissue selection (default is 'lineage1').

        Returns:
        - val (ipywidgets.ValueWidget): A widget containing the labeled cell lines.
        
        The function creates an interactive widget with the following components:
        - A multi-select widget for choosing tissues.
        - A button for triggering the labeling process.
        - An output widget for displaying selected tissues and labeled cell lines.
        - Options for saving the results to a CSV file.

        Example:
        ```
        # Usage example:
        result_widget = Help().labelling(my_dataframe, my_map_dataframe, rows=7, column='lineage_category')
        ```

        """
        tl = df_map[column].dropna().value_counts()
        tissue_list = [x[0] for x in list(filter(lambda x: x[1] >= minlines, zip(tl.index.values.astype(str) , tl.values)))]
        # tissue_list = (np.unique(df_map[column].dropna().values.astype(str)))
        layout_hidden  = wid.Layout(visibility = 'hidden')
        layout_visible = wid.Layout(visibility = 'visible')
        seltissue = wid.SelectMultiple(
            options=tissue_list,
            value=None,
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
                cell_lines = select_cell_lines(df, df_map, seltissue.value, nested = False)
                val.value = Help(verbose=self.verbose).labelling(df, columns=cell_lines, three_class=mode_buttons.value)
                if save_textbox.layout == layout_visible:
                    val.value.to_csv(save_textbox.value, index=True)
                display(val.value.value_counts())
        button.on_click(on_button_clicked)
        out1 = wid.Output()
        out2 = wid.Output()
        val = wid.ValueWidget()
        cnt = wid.VBox([wid.HBox([button, out1]), mode_buttons, seltissue, wid.HBox([saveto_but, save_textbox]), out2])
        display(cnt)
        return val

def helpbox(df: pd.DataFrame, df_map: pd.DataFrame, rows: int=5, column='lineage1'):
    """
    Generate an interactive widget for labeling cell lines based on specified criteria.

    Parameters:
    - df (pd.DataFrame): The main DataFrame containing the data.
    - df_map (pd.DataFrame): A DataFrame used for mapping data.
    - rows (int): The number of rows to display in the widget for selecting tissues (default is 5).
    - column (str): The column in 'df_map' to use for tissue selection (default is 'lineage1').

    Returns:
    - val (ipywidgets.ValueWidget): A widget containing the labeled cell lines.
    
    The function creates an interactive widget with the following components:
    - A multi-select widget for choosing tissues.
    - A button for triggering the labeling process.
    - An output widget for displaying selected tissues and labeled cell lines.
    - Options for saving the results to a CSV file.

    Example:
    ```
    # Usage example:
    result_widget = helpbox(my_dataframe, my_map_dataframe, rows=7, column='lineage_category')
    ```

    """
    tissue_list = (np.unique(df_map[column].dropna().values.astype(str)))
    layout_hidden  = wid.Layout(visibility = 'hidden')
    layout_visible = wid.Layout(visibility = 'visible')
    seltissue = wid.SelectMultiple(
        options=tissue_list,
        value=['Kidney'],
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
            cell_lines = select_cell_lines(df, df_map, seltissue.value, nested = False)
            val.value = HELP(df, columns=cell_lines, three_class=mode_buttons.value)
            if save_textbox.layout == layout_visible:
                val.value.to_csv(save_textbox.value, index=True)
            display(val.value.value_counts())
    button.on_click(on_button_clicked)
    out1 = wid.Output()
    out2 = wid.Output()
    val = wid.ValueWidget()
    cnt = wid.VBox([wid.HBox([button, out1]), mode_buttons, seltissue, wid.HBox([saveto_but, save_textbox]), out2])
    display(cnt)
    return val
