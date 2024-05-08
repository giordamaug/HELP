import ipywidgets as wid
from typing import List
import matplotlib.pyplot as plt
from ..models.labelling import labelling
from ..utility.selection import select_cell_lines, delrows_with_nan_percentage
from ..preprocess.loaders import feature_assemble_df
from ..models.prediction import VotingSplitClassifier, k_fold_cv
import pandas as pd
import numpy as np
import os, glob
from ipyfilechooser import FileChooser
from .filecollector import FileCollector
from IPython.display import HTML as html_print
from IPython.display import display
from ..visualization.plot import svenn_intesect
from typing import List, Sequence, Iterable, Optional
import fnmatch

_LB_APPLY = 'Apply on'
_LB_DONE = 'DONE'
_LB_NANREM = 'Nan Removal'
_LB_FILTER = "Line Filtering"
_LB_LABEL = "Labelling"
_LB_SAVE = "Saving"
_LB_INPUT = "File input"
_LB_SELGENE = "Select genes files"
_LB_CNGGENE = "Change genes files"
_LB_SELATTR = "Select attribute files"
_LB_CNGATTR = "Change attribute files"
_LB_SELGENE_SUB = "Select genes to subtract"
_LB_CNGGENE_SUB = "Change genes to subtract"
_LB_SEL_LAB = "Select labelling file"
_LB_CNG_LAB = "Change labelling file"
_LB_PREPROC = "Preprocessing"
_LB_PREDICT = "Prediction"
_LB_INTERSET = "Intersection"
_LB_IDENTIFY = "Identification"
_LB_CNG_FILE1 = "Change CRISPR file"
_LB_SEL_FILE1 = "Select CRISPR file"
_LB_CNG_FILE2 = "Change Model file"
_LB_SEL_FILE2 = "Select Model file"

def file_with_ext(path, extension='.csv'):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                file_list.append(file)
    return file_list

def match_item(item: str, filter_pattern: Sequence[str]) -> bool:
    """Check if a string matches one or more fnmatch patterns."""
    if isinstance(filter_pattern, str):
        filter_pattern = [filter_pattern]
    idx = 0
    found = False
    while idx < len(filter_pattern) and not found:
        found |= fnmatch.fnmatch(item.lower(), filter_pattern[idx].lower())
        idx += 1
    return found

def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)

def print_color(t):
    display(html_print(' '.join([cstr(ti, color=ci) for ti,ci in t])))
    
def pipeline(path: str=os.getcwd(), savepath: str=os.getcwd(), labelpath: str=os.getcwd(), attributepath: str=os.getcwd(),
               filename: str='', modelname:str='', labelname:str='', commonlabelname:str = '',
               rows: int=5, minlines=10, percent = 100.0, 
               line_group='OncotreeLineage', line_col='ModelID', 
               verbose=False, show_progress=False):
    """
    Create a data processing pipeline.

    This function creates a data processing pipeline for handling input data, model files,
    label files, and various parameters involved in data processing. It initializes a GUI
    with widgets for user interaction and displays the processed data frames.

    :param path: Path for input file loading.
    :param savepath: Path for saving files.
    :param labelpath: Path for label files.
    :param filename: Name of the CRISPR effect input file.
    :param modelname: Pathname of the Model input file.
    :param labelname: Name of the label input file.
    :param rows: The number of rows to display in the widget for selecting tissues (default is 5).
    :param minlines: Minimum number of cell lines for tissue/lineage to be considered (default is 1).
    :param percent: Percentage of NaN allowed in genes (default is 100.0).
    :param line_group: The column in 'df_map' to use for tissue selection (default is 'OncotreeLineage').
    :param line_col: The column in 'df_map' to use for line selection (default is 'ModelID').
    :param verbose: Whether to print detailed messages (default is False).
    :param show_progress: Whether to show progress bars (default is False).

    :return: Widget containing the labeled cell lines.
    """
    tabs = wid.Tab()
    df_map = None
    df = None
    df_orig = None
    val = wid.ValueWidget()
    val.value = None, df, df_orig, df_map 
    tissue_list = []
    selector_list = []
    out1 = wid.Output()
    out2 = wid.Output()
    out3 = wid.Output()
    out4 = wid.Output()
    out6 = wid.Output()
    out7 = wid.Output()
    acd2 = wid.Accordion()
    acd6 = wid.Accordion()
    acd7 = wid.Accordion()
    # PRE-PROCESSING TAB
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
        try:
            df_orig = val.value[2]
            df = delrows_with_nan_percentage(df_orig, perc=float(nanrem_set.value))
            try:
                df_map = val.value[3]
                tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) 
                               if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
                seltissue.options = ['__all__'] +  tissue_list
                seltissue.value=['__all__']
                val.value = val.value[0], df[np.intersect1d(df.columns,df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
            except:
                val.value = val.value[0], df, val.value[2], val.value[3]
            with out3:
                out3.clear_output()
                print_color(((f'Removed {len(df_orig)-len(df)}/{len(df_orig)} rows (with at least {nanrem_set.value}% NaN)', 'green'),))
            Vb1.set_title(0, f"{_LB_NANREM} ({nanrem_set.value})")
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
        try:
            tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
            seltissue.options = ['__all__'] +  tissue_list
            seltissue.value=['__all__']
            val.value = val.value[0], df[np.intersect1d(df.columns, df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
            Vb1.set_title(1, f"{_LB_FILTER} (Lines: {minline_set.value})")
        except:
            val.value = val.value[0], df, val.value[2], val.value[3]
            with out1:
                out1.clear_output()
                print_color(((f'Problem processing map file ...', 'red'),)) 
    minline_set.observe(minline_set_changed, names='value')
    selselector = wid.Dropdown(
        options=['__all__'] + selector_list,
        value='__all__',
        description='Selector:',
        tooltip = 'select the group type of lines',
        disabled=False,
    )
    def selselector_changed(b):
        df = val.value[1]
        df_map = val.value[3]
        if selselector.value != ():
            try:
                tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
                seltissue.options = ['__all__'] +  tissue_list
                seltissue.value=['__all__']
                val.value = val.value[0], df[np.intersect1d(df.columns, df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
            except:
                val.value = val.value[0], df, val.value[2], val.value[3]
                with out1:
                    out1.clear_output()
                    print_color(((f'Problem processing map file ...', 'red'),)) 
            with out1:
                out1.clear_output()
                display(selselector.value)
          
    selselector.observe(selselector_changed, names='value')
    seltissue = wid.SelectMultiple(
        options=['__all__'] + tissue_list if tissue_list != [] else [],
        value=['__all__'] if tissue_list != [] else [],
        rows=rows,
        description=line_group if line_group in selselector.options else '',
        tooltip = 'select lines by the chosen group',
        disabled=False
    )
    def seltissue_changed(b):
        if seltissue.value != ():
            if seltissue.value == ('__all__',):
                fname = f"{selselector.value}_all{'_mom' if selmode_button.value else ''}.csv"
                fc3._filename.value = fname
            else:
                fname = f"{selselector.value}_{'_'.join([str(s).replace(' ','-').replace('/','-') for s in seltissue.value if str(s) != '__all__'])}.csv"
                fc3._filename.value = fname
            fc3._apply_selection()
        with out1:
            out1.clear_output()
            print_color(((f"{','.join(seltissue.value)}", "orange"),))
    seltissue.observe(seltissue_changed, names='value')
    # IDENTIFICATION TAB
    fc3 = FileChooser(savepath, title='Choose file', filter_pattern='*.csv', layout=wid.Layout(width='auto'))
    def fc3_change_title(fc3):
        if os.path.isfile(fc2.selected):
            fc3._label.value = fc3._LBL_TEMPLATE.format(f'{fc3.selected}', 'green')
            acd4.set_title(1, f"{_LB_SAVE} ({fc3.selected_filename})")
        else:
            fc3._label.value = fc3._LBL_TEMPLATE.format(f'{fc3.selected} not a file', 'red')
            acd4.set_title(1, f"{_LB_SAVE}")
    fc3.register_callback(fc3_change_title)
    saveto_but = wid.Button(description="Save ...", button_style='primary')
    def on_savebutton_clicked(b):
        if isinstance(val.value[0], pd.DataFrame):
            try:
                fc3._label.value = fc3._LBL_TEMPLATE.format(f'{fc3.selected}', 'orange')
                val.value[0].to_csv(fc3.selected, index=True)
                fc3._label.value = fc3._LBL_TEMPLATE.format(f'{fc3.selected}', 'green')
            except:
                fc3._label.value = fc3._LBL_TEMPLATE.format(f'Problem saving {fc3.selected}!', 'green')
                with out4:
                    out4.clear_output()
                    print_color(((f'Problem saving label file (maybe empty)!', 'red'),))
            with out4:
                out4.clear_output()
        else:
            with out4:
                out4.clear_output()
                print_color(((f'Label dataframe is null (apply labelling before saving)!', 'red'),))
    saveto_but.on_click(on_savebutton_clicked)

    mode_buttons = wid.RadioButtons(
        options=["E|NE", "E|aE|sNE", "E|(aE|sNE)"],
        value='E|NE',
        description='',
        tooltips=['2 classes (one division)', '3 classes (one division)', '3 classes (two-times subdivision)'],
    )
    selmode_button = wid.Checkbox(
        value=False,
        description='Nested',
        disabled=False,
        indent=False
    )
    button = wid.Button(description=_LB_APPLY, button_style='primary')
    def on_button_clicked(b):
        df = val.value[1]
        df_map = val.value[3]
        with out1:
            out1.clear_output()
            print_color(((f'Labelling {len(df)} genes of {",".join(seltissue.value)} ...', 'orange'),))
        with out2:
            out2.clear_output()
            if seltissue.value == ('__all__',):
                selector = [x for x in seltissue.options if x != '__all__']
            else:
                selector = [x for x in seltissue.value if x != '__all__']
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
            val.value = labelling(df, columns=cell_lines, mode=mode, n_classes=nclasses, labelnames=labelnames, verbose=verbose, show_progress=show_progress), df, val.value[2], val.value[3]
        with out1:
            out1.clear_output()
            print_color(((_LB_DONE, 'green'),))

    button.on_click(on_button_clicked)
    # INPUT TAB
    if os.path.isfile(filename):
        fc1 = FileChooser(os.path.dirname(os.path.abspath(filename)), filter_pattern='*.csv', filename=os.path.basename(filename), select_default=True, layout=wid.Layout(width='auto'))
        acd2.children = (fc1,)
        try:
            df_orig = pd.read_csv(fc1.selected).rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
            df = delrows_with_nan_percentage(df_orig, perc=float(nanrem_set.value))
            df_map = val.value[3]
            val.value = val.value[0],df, df_orig, val.value[3]
            with out3:
                out3.clear_output()
                print_color(((f'Removed {len(df_orig)-len(df)}/{len(df_orig)} rows (with at least {nanrem_set.value}% NaN)', 'green'),))
            if df_map is not None and len(np.unique(df_map[line_group].dropna().values)) > 0:
                try:
                    tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
                    seltissue.options = ['__all__'] + tissue_list
                    seltissue.value=['__all__']
                    val.value = val.value[0], df[np.intersect1d(df.columns, df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
                except:
                    val.value = val.value[0], df, val.value[2], val.value[3]
                    fc1._label.value = fc1._LBL_TEMPLATE.format(f'Problem reading {fc1.selected} file ...', 'red') 
            fc1._label.value = fc1._LBL_TEMPLATE.format(f'{fc1.selected}', 'green')
            acd2.set_title(0, f"{_LB_CNG_FILE1} ({fc1.selected_filename})")
        except:
            fc1._label.value = fc1._LBL_TEMPLATE.format(f'Problem loading {fc1.selected} file ...', 'red') 
            acd2.set_title(0, f"{_LB_SEL_FILE1}")
    else:
        if os.path.isdir(path):
            fc1 = FileChooser(path, filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        else:
            fc1 = FileChooser(filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        acd2.children = (fc1,)
        acd2.set_title(0, f"{_LB_SEL_FILE1}")
    
    def fc1_change_title(fc1):
        try:
            df_orig = pd.read_csv(fc1.selected).rename(columns={'Unnamed: 0': 'gene'}).rename(columns=lambda x: x.split(' ')[0]).set_index('gene').T
            df = delrows_with_nan_percentage(df_orig, perc=float(nanrem_set.value))
            df_map = val.value[3]
            val.value = val.value[0],df ,df_orig, val.value[3]
            with out3:
                out3.clear_output()
                print_color(((f'Removed {len(df_orig)-len(df)}/{len(df_orig)} rows (with at least {nanrem_set.value}% NaN)', 'green'),))
            if df_map is not None and len(np.unique(df_map[line_group].dropna().values)) > 0:
                try:
                    tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
                    seltissue.options = ['__all__'] + tissue_list
                    seltissue.value=['__all__']
                    val.value = val.value[0], df[np.intersect1d(df.columns, df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
                except:
                    val.value = val.value[0], df, val.value[2], val.value[3]
                    fc1._label.value = fc1._LBL_TEMPLATE.format(f'Problem reading {fc1.selected} file ...', 'red') 
            fc1._label.value = fc1._LBL_TEMPLATE.format(f'{fc1.selected}', 'green')
            acd2.set_title(0, f"{_LB_CNG_FILE1} ({fc1.selected_filename})")
        except:
            fc1._label.value = fc1._LBL_TEMPLATE.format(f'Problem loading {fc1.selected} file ...', 'red') 
            acd2.set_title(0, f"{_LB_SEL_FILE1}")
        acd2.set_title(0, f"{_LB_SEL_FILE1}")

    fc1.register_callback(fc1_change_title)
    if os.path.isfile(modelname):
        fc2 = FileChooser(os.path.dirname(os.path.abspath(modelname)), filter_pattern='*.csv', filename=os.path.basename(modelname), select_default=True, layout=wid.Layout(width='auto'))
        acd2.children += (fc2,)
        try:
            df_map = pd.read_csv(fc2.selected)
            df = val.value[1]
            val.value = val.value[0], val.value[1] ,val.value[2], df_map
            selselector.options = list(df_map.columns)
            selselector.value = line_group if line_group in selselector.options else selselector.options[0]
            if len(np.unique(df_map[selselector.value].dropna().values)) > 0:
                try:
                    tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
                    seltissue.options = ['__all__'] + tissue_list
                    seltissue.value=['__all__']
                    val.value = val.value[0], df[np.intersect1d(df.columns, df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
                except:
                    val.value = val.value[0], df, val.value[2], val.value[3]
                    fc2._label.value = fc2._LBL_TEMPLATE.format(f'Problem reading {fc2.selected} file ...', 'red') 
            fc2._label.value = fc2._LBL_TEMPLATE.format(f'{fc2.selected}', 'green')
            acd2.set_title(1, f"{_LB_CNG_FILE2} ({fc2.selected_filename})")
        except:
            fc2._label.value = fc2._LBL_TEMPLATE.format(f'Problem loading {fc2.selected} file ...', 'red') 
            acd2.set_title(1, f"{_LB_SEL_FILE2}")
    else:
        if os.path.isdir(path):
            fc2 = FileChooser(path, filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        else:
            fc2 = FileChooser(filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        acd2.children += (fc2,)
        acd2.set_title(1, f"{_LB_SEL_FILE2}")
        
    def fc2_change_title(fc2):
        try:
            df_map = pd.read_csv(fc2.selected)
            df = val.value[1]
            val.value = val.value[0], val.value[1] ,val.value[2], df_map
            selselector.options = list(df_map.columns)
            selselector.value = line_group if line_group in selselector.options else selselector.options[0]
            if len(np.unique(df_map[selselector.value].dropna().values)) > 0:
                try:
                    tissue_list = [tissue for tissue in np.unique(df_map[selselector.value].dropna().values) if len(np.intersect1d(df.columns, df_map[df_map[selselector.value] == tissue][line_col].values)) >= minline_set.value]
                    seltissue.options = ['__all__'] + tissue_list
                    seltissue.value=['__all__']
                    val.value = val.value[0], df[np.intersect1d(df.columns, df_map[df_map[line_group].isin(tissue_list)][line_col].values)], val.value[2], val.value[3]
                except:
                    val.value = val.value[0], df, val.value[2], val.value[3]
                    fc2._label.value = fc2._LBL_TEMPLATE.format(f'Problem reading {fc2.selected} file ...', 'red') 
            fc2._label.value = fc2._LBL_TEMPLATE.format(f'{fc2.selected}', 'green')
            acd2.set_title(1, f"{_LB_CNG_FILE2} ({fc2.selected_filename})")
        except:
            fc2._label.value = fc2._LBL_TEMPLATE.format(f'Problem loading {fc2.selected} file ...', 'red') 
            acd2.set_title(1, f"{_LB_SEL_FILE2}")
    fc2.register_callback(fc2_change_title)
    # INTERSECTION TAB
    if os.path.isdir(labelpath):
        fc4 = FileCollector(labelpath, default_path=labelpath, filter_pattern='*.csv')
        acd6.children = (fc4,)
        acd6.set_title(0, f"{_LB_CNGGENE} ({os.path.basename(labelpath)})")
    else:
        fc4 = FileCollector(filter_pattern='*.csv')
        acd6.children = (fc4,)
        acd6.set_title(0, f"{_LB_SELGENE}")
    def fc4_change_title(fc4):
        if fc4.selected != ():
            acd6.set_title(0, f"{_LB_CNGGENE} ({os.path.basename(fc4.selected_path)})")
        else:
            acd6.set_title(0, f"{_LB_SELGENE}")
    fc4.register_callback(fc4_change_title)

    if os.path.isfile(commonlabelname):
        fc5 = FileChooser(os.path.dirname(os.path.abspath(commonlabelname)), filter_pattern='*.csv', 
                          filename=os.path.basename(commonlabelname), default_path=os.path.dirname(os.path.abspath(commonlabelname)), layout=wid.Layout(width='auto'))
        fc5._filename.value = os.path.basename(commonlabelname)
        fc5._apply_selection()
        acd6.children += (fc5,)
        acd6.set_title(1, f"{_LB_CNGGENE_SUB} ({os.path.basename(fc5.selected)})")
    else:
        fc5 = FileChooser(filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        acd6.children += (fc5,)
        acd6.set_title(1, f"{_LB_SELGENE_SUB}")

    def fc5_change_title(fc5):
        if os.path.isfile(fc5.selected) and match_item(fc5.selected, '*.csv'):
            fc5._label.value = fc5._LBL_TEMPLATE.format(f'{fc5.selected}', 'green')
            acd6.set_title(1, f"{_LB_CNGGENE_SUB} ({os.path.basename(fc5.selected)})")
        else:
            fc5._label.value = fc5._LBL_TEMPLATE.format(f'{fc5.selected}', 'red')
            acd6.set_title(1, f"{_LB_SELGENE_SUB}")
    fc5.register_callback(fc5_change_title)

    setbut = wid.Button(description="Intersect ...", button_style='primary')
    def on_setbut_clicked(b):
        if fc4.selected == ():
            with out6:
                out6.clear_output()
                print_color(((f'No file selected!', 'orange'),))
        else:
            try:
                csEGs = []
                #for f in files.value:
                for f in fc4.selected:
                    dfl = pd.read_csv(f, index_col=0)
                    csEG = dfl[dfl['label'] == 'E'].index.values
                    if fc5.selected is not None and os.path.isfile(fc5.selected):
                        df_common = pd.read_csv(os.path.join(savepath,fc5.selected), index_col=0)
                        cEG = df_common[df_common['label']=='E'].index.values
                        csEG = np.setdiff1d(csEG, cEG)
                    csEGs += [set(csEG)]
                with out6:
                    out6.clear_output()
                    fig1, axes1 = svenn_intesect(csEGs, labels=[x.split('.')[0] for x in fc4.selected], ylabel='EGs', figsize=(10,4))
                    plt.show(fig1)
            except Exception as e:
                with out6:
                    out6.clear_output()
                    print_color(((f'Problem processing label files!', 'red'),))
                    print_color(((f'{e}', 'black'),))

    setbut.on_click(on_setbut_clicked)
    # PREDICTION TAB
    if os.path.isdir(attributepath):
        fc6 = FileCollector(attributepath, default_path=attributepath, filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        acd7.children += (fc6,)
        acd7.set_title(0, f"{_LB_CNGATTR} ({os.path.basename(attributepath)})")
    else:
        fc6 = FileCollector(filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        acd7.children += (fc6,)
        acd7.set_title(0, f"{_LB_SELATTR}")

    def fc6_change_title(fc4):
        if fc6.selected != ():
            acd7.set_title(0, f"{_LB_CNGATTR} ({os.path.basename(fc6.selected_path)})")
        else:
            acd7.set_title(0, f"{_LB_SELATTR}")
    fc6.register_callback(fc6_change_title)

    if os.path.isfile(labelname):
        fc7 = FileChooser(os.path.dirname(os.path.abspath(labelname)), filename=os.path.basename(labelname), select_default=True, layout=wid.Layout(width='auto'))
        acd7.children += (fc7,)
        try:
            fc7._label.value = fc7._LBL_TEMPLATE.format(f'{fc7.selected}', 'green')
            acd7.set_title(1, f"{_LB_CNG_LAB} ({os.path.basename(fc7.selected)})")
        except:
            fc7._label.value = fc7._LBL_TEMPLATE.format(f'Problem loading {fc7.selected} file ...', 'red') 
            acd7.set_title(1, f"{_LB_SEL_LAB}")
    else:
        if os.path.isdir(labelpath):
            fc7 = FileChooser(labelpath, filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        else:
            fc7 = FileChooser(filter_pattern='*.csv', layout=wid.Layout(width='auto'))
        acd7.children += (fc7,)
        acd7.set_title(1, f"{_LB_SEL_LAB}")

    def fc7_change_title(fc7):
        if os.path.isfile(fc7.selected):
            fc7._label.value = fc7._LBL_TEMPLATE.format(f'{fc7.selected}', 'green')
            acd7.set_title(1, f"{_LB_CNG_LAB} ({os.path.basename(fc7.selected)})")
        else:
            fc7._label.value = fc7._LBL_TEMPLATE.format(f'{fc7.selected}', 'red')
            acd7.set_title(1, f"{_LB_SEL_LAB}")

    fc7.register_callback(fc7_change_title)
    valbut = wid.Button(description="Validate ...", button_style='primary')
    def on_valbut_clicked(b):
        if fc6.selected == ():
            with out7:
                out7.clear_output()
                print_color(((f'No attribute file!', 'orange'),))
        else:
            try:
                with out7:
                    out7.clear_output()
                    print_color(((f'Validating model ...', 'orange'),))
                features = [{'fname': f, 'fixna' : False, 'normalize': 'std'} for f in fc6.selected]
                df_y = pd.read_csv(fc7.selected, index_col=0)
                df_y = df_y.replace({'aE': 'NE', 'sNE': 'NE'})
            except Exception as e:
                with out7:
                    out7.clear_output()
                    print_color(((f'Problem processing label files!', 'red'),))
                    print_color(((f'{e}', 'black'),))
            try:
                df_X, df_y = feature_assemble_df(df_y, features=features, saveflag=False, verbose=verbose, show_progress=show_progress)
            except Exception as e:
                with out7:
                    out7.clear_output()
                    print_color(((f'Problem assembling attributes files!', 'red'),))
                    print_color(((f'{e}', 'black'),))
            try:
                clf = VotingSplitClassifier(n_voters=10, n_jobs=-1, random_state=-1)
                with out7:
                    df_scores, scores, predictions = k_fold_cv(df_X, df_y, clf, n_splits=5, seed=0, verbose=verbose, show_progress=show_progress)
                    out7.clear_output()
                    print_color(((_LB_DONE, 'green'),))
                    display(df_scores)
            except Exception as e:
                with out7:
                    out7.clear_output()
                    print_color(((f'Problem in validation!', 'red'),))
                    print_color(((f'{e}', 'black'),))

    valbut.on_click(on_valbut_clicked)

    # MAIN WIDGET GUI    
    txt1 = wid.HTMLMath(
        value=r"""In this section you filter the CRIPR score lines by:
                <ol>
                  <li>removing genes with a certain percentage of missing cell line scores;</li>
                  <li>select the type of cell lines grouping (by tissue, by disease, etc.) and</li>
                  <li>filter the grous with a minimum amount of lines;</li>
                  <li>select a specific set of groups from which to extract cell line score.</li>
                </ol>""",
    )
    txt2 = wid.HTMLMath(
        value=r"""In this section you select: 
                <ol>
                    <li>the CRIPR effect file contanin cell lines scores, and
                    <li>the Model file mapping cell line names to tissues/diseases,etc. 
                </ol>
                NOTE: the selected file is loaded when the file path appears in green text.""",
    )
    txt6 = wid.HTMLMath(
        value=r"""In this section you can intersect contet specific EGs from different tissues/diseases:
                <ol>
                  <li>select the directory where are the label files of context-specific genes;</li>
                  <li>select the an option label file representing EGs you want to exclude from intersection</li>
                  <li>apply intersection an display the resulting Super Venn diagram.</li>
                </ol>""",
    )
    txt4 = wid.HTMLMath(
        value=r"""In this section you can compute the labelling of a set of gene by setting dome parameter: 
                <ol>
                  <li>the type of labelling: binary (E|NE), ternary (E|aE|sNE), or binary amd then binary in the second class (E|(aE|sNE));</li>
                  <li>the separation algorithm (Otsu is the default)</li>
                </ol>
                The labelling results can be saved in a CSV file for future use.<p>
                NOTE: the labelling process is complete once you see a green-colored "DONE".""",
    )
    txt7 = wid.HTMLMath(
        value=r"""In this section you can compute make prediction with model trained on labelling files: 
                  In the first widget you can select the directory (and then the files) used as feature input for the
                  builfding model.
                  In the second widget you can select the label file used for training the model.<p>
                  NOTE: the labelling process is complete once you see a green-colored "DONE"."""
    )
    Vb2 = wid.VBox([txt2, acd2])
    acd1 = wid.Accordion(children=[wid.VBox([nanrem_set, out3]), wid.VBox([minline_set, selselector, wid.HBox([seltissue, selmode_button])])])
    acd1.set_title(0, f"{_LB_NANREM} ({percent}%)")
    acd1.set_title(1, f"{_LB_FILTER} (Lines: {minline_set.value})")
    Vb1 = wid.VBox([txt1, acd1])
    acd4 = wid.Accordion(children=[mode_buttons, wid.HBox([fc3, saveto_but,out4])])
    acd4.set_title(0, f"{_LB_LABEL} ({mode_buttons.value})")
    acd4.set_title(1, f"{_LB_SAVE}") if fc3.selected_filename == "" else acd4.set_title(1, f"{_LB_SAVE} ({fc3.selected_filename})")
    Vb4 = wid.VBox([txt4, wid.HBox([acd4, wid.VBox([wid.HBox([button, out1]), out2])])])
    Vb6 = wid.VBox([txt6, wid.HBox([acd6, wid.VBox([setbut, out6])])])
    Vb7 = wid.VBox([txt7, wid.HBox([acd7,wid.VBox([valbut, out7])])])
    tabs.children = [Vb2, Vb1, Vb4, Vb6, Vb7]
    tabs.set_title(0, f'{_LB_INPUT}')
    tabs.set_title(1, f'{_LB_PREPROC}')
    tabs.set_title(2, f'{_LB_IDENTIFY}')
    tabs.set_title(3, f'{_LB_INTERSET}')
    tabs.set_title(4, f'{_LB_PREDICT}')
    display(tabs)
    return val