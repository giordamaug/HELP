import sys
import os
import fnmatch
import string
import warnings
from typing import Optional, Sequence, Mapping, Callable, List, Sequence, Iterable, Tuple
from ipywidgets import Dropdown, Text, Select, Button, HTML, SelectMultiple
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget

""" Errors """
class ParentPathError(Exception):
    """ParentPathError class."""

    def __init__(self, path: str, parent_path: str, message: Optional[str] = None):
        self.path = path
        self.sandbox_path = parent_path
        self.message = message or f'{path} is not a part of {parent_path}'
        super().__init__(self.message)


class InvalidPathError(Exception):
    """InvalidPathError class."""

    def __init__(self, path: str, message: Optional[str] = None):
        self.path = path
        self.message = message or f'{path} does not exist'
        super().__init__(self.message)


class InvalidFileNameError(Exception):
    """InvalidFileNameError class."""
    invalid_str = [os.sep, os.pardir]

    if os.altsep:
        invalid_str.append(os.altsep)

    def __init__(self, filename: Tuple[str], message: Optional[str] = None):
        self.filename = filename
        self.message = message or f'{filename} cannot contain {self.invalid_str}'
        super().__init__(self.message)

"""Utility"""
def get_subpaths(path: str) -> List[str]:
    """Walk a path and return a list of subpaths."""
    if os.path.isfile(path):
        path = os.path.dirname(path)

    paths = [path]
    path, tail = os.path.split(path)

    while tail:
        paths.append(path)
        path, tail = os.path.split(path)

    return paths


def has_parent(path: str) -> bool:
    """Check if a path has a parent folder."""
    return os.path.basename(path) != ''


def has_parent_path(path: str, parent_path: Optional[str]) -> bool:
    """Verifies if path falls under parent_path."""
    check = True

    if parent_path:
        check = os.path.commonpath([path, parent_path]) == parent_path

    return check


def strip_parent_path(path: str, parent_path: Optional[str]) -> str:
    """Remove a parent path from a path."""
    stripped_path = path

    if parent_path and path.startswith(parent_path):
        stripped_path = path[len(parent_path):]

    return stripped_path


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


def get_dir_contents(
        path: str,
        show_hidden: bool = False,
        dir_icon: Optional[str] = None,
        dir_icon_append: bool = False,
        filter_pattern: Optional[Sequence[str]] = None,
        top_path: Optional[str] = None) -> List[str]:
    """Get directory contents."""
    files = list()
    dirs = list()

    if os.path.isdir(path):
        for item in os.listdir(path):
            append = True
            if item.startswith('.') and not show_hidden:
                append = False
            full_item = os.path.join(path, item)
            if append and os.path.isdir(full_item):
                dirs.append(item)
            elif append:
                if filter_pattern:
                    if match_item(item, filter_pattern):
                        files.append(item)
                else:
                    files.append(item)
        if has_parent(strip_parent_path(path, top_path)):
            dirs.insert(0, os.pardir)
    if dir_icon:
        return prepend_dir_icons(sorted(dirs), dir_icon, dir_icon_append) + sorted(files)
    else:
        return sorted(dirs) + sorted(files)


def prepend_dir_icons(dir_list: Iterable[str], dir_icon: str, dir_icon_append: bool = False) -> List[str]:
    """Prepend unicode folder icon to directory names."""
    if dir_icon_append:
        str_ = [dirname + f'{dir_icon}' for dirname in dir_list]
    else:
        str_ = [f'{dir_icon}' + dirname for dirname in dir_list]

    return str_


def get_drive_letters() -> List[str]:
    """Get all drive letters minus the drive used in path."""
    drives: List[str] = []

    if sys.platform == 'win32':
        # Windows has drive letters
        drives = [os.path.realpath(f'{d}:\\') for d in string.ascii_uppercase if os.path.exists(f'{d}:')]

    return drives


def is_valid_filename(filename: str) -> bool:
    """Verifies if a filename does not contain illegal character sequences"""
    valid = True
    valid = valid and os.pardir not in filename
    valid = valid and os.sep not in filename

    if os.altsep:
        valid = valid and os.altsep not in filename

    return valid


def normalize_path(path: str) -> str:
    """Normalize a path string."""
    normalized_path = os.path.realpath(path)

    if not os.path.isdir(normalized_path):
        raise InvalidPathError(path)

    return normalized_path

"""Main Class"""
class FileCollector(VBox, ValueWidget):
    """FileChooser class."""

    _LBL_TEMPLATE = '<span style="color:{1};">{0}</span>'
    _LBL_NOFILE = 'No selection'

    def __init__(
            self,
            path: str = os.getcwd(),
            filename: Tuple[str] = (),
            title: str = '',
            select_desc: str = 'Select',
            change_desc: str = 'Change',
            show_hidden: bool = False,
            select_default: bool = False,
            dir_icon: Optional[str] = '\U0001F4C1 ',
            dir_icon_append: bool = False,
            filter_pattern: Optional[Sequence[str]] = None,
            sandbox_path: Optional[str] = None,
            layout: Layout = Layout(width='500px'),
            **kwargs):
        """Initialize FileChooser object."""
        # Check if path and sandbox_path align
        if sandbox_path and not has_parent_path(normalize_path(path), normalize_path(sandbox_path)):
            raise ParentPathError(path, sandbox_path)

        # Verify the filename is valid
        if not is_valid_filename(filename):
            raise InvalidFileNameError(filename)

        self._default_path = normalize_path(path)
        self._default_filename = filename
        self._selected_path: Optional[str] = None
        self._selected_filename: Optional[Tuple[str]] = ()
        self._show_hidden = show_hidden
        self._select_desc = select_desc
        self._change_desc = change_desc
        self._select_default = select_default
        self._dir_icon = dir_icon
        self._dir_icon_append = dir_icon_append
        self._filter_pattern = filter_pattern
        self._sandbox_path = normalize_path(sandbox_path) if sandbox_path is not None else None
        self._callback: Optional[Callable] = None

        # Widgets
        self._pathlist = Dropdown(
            description="",
            layout=Layout(
                width='auto',
                grid_area='pathlist'
            )
        )
        self._filename = SelectMultiple(
            rows=8,
            options= (),
            layout=Layout(
                width='auto',
                grid_area='filename',
            ),
        )
        self._dircontent = SelectMultiple(
            rows=8,
            layout=Layout(
                width='auto',
                grid_area='dircontent'
            )
        )
        self._cancel = Button(
            description='Cancel',
            layout=Layout(
                min_width='6em',
                width='6em',
                display='none'
            )
        )
        self._select = Button(
            description=self._select_desc,
            layout=Layout(
                min_width='6em',
                width='6em'
            )
        )
        self._add = Button(
            description='>>',
            layout=Layout(
                min_width='3em',
                width='3em',
                display='none'
            )
        )
        self._remove = Button(
            description='<<',
            layout=Layout(
                min_width='3em',
                width='3em',
                display='none'
            )
        )
        self._title = HTML(
            value=title
        )

        if title == '':
            self._title.layout.display = 'none'

        # Widget observe handlers
        self._pathlist.observe(self._on_pathlist_select, names='value')
        self._dircontent.observe(self._on_dircontent_select, names='value')
        self._filename.observe(self._on_filename_change, names='value')
        self._select.on_click(self._on_select_click)
        self._cancel.on_click(self._on_cancel_click)
        self._add.on_click(self._on_add_click)
        self._remove.on_click(self._on_remove_click)

        # Selected file label
        self._label = HTML(
            value=self._LBL_TEMPLATE.format(self._LBL_NOFILE, 'black'),
            placeholder='',
            description='',
            layout=Layout(margin='0 0 0 1em')
        )

        # Layout
        self._gb = GridBox(
            children=[
                self._pathlist,
                self._filename,
                self._add,
                self._remove,
                self._dircontent
            ],
            layout=Layout(
                display='none',
                min_width='450px',
                width='auto',
                grid_gap='2px 2px 2px',
                grid_template_rows='auto auto auto',
                grid_template_columns='auto auto auto',
                grid_template_areas='''
                    'pathlist pathlist pathlist'
                    'dircontent add filename'
                    'dircontent remove filename'
                    '''
            )
        )

        buttonbar = HBox(
            children=[
                self._select,
                self._cancel,
                #self._add,
                Box([self._label], layout=Layout(overflow='auto'))
            ],
            layout=Layout(width='auto')
        )

        # Call setter to set initial form values
        self._set_form_values(self._default_path, self._default_filename)

        # Use the defaults as the selected values
        if self._select_default:
            self._apply_selection()

        # Call VBox super class __init__
        super().__init__(
            children=[
                self._title,
                self._gb,
                buttonbar
            ],
            layout=layout,
            **kwargs
        )

    def _set_form_values(self, path: str, filename: Tuple[str]) -> None:
        """Set the form values."""
        # Check if the path falls inside the configured sandbox path
        if self._sandbox_path and not has_parent_path(path, self._sandbox_path):
            raise ParentPathError(path, self._sandbox_path)

        # Disable triggers to prevent selecting an entry in the Select
        # box from automatically triggering a new event.
        self._pathlist.unobserve(self._on_pathlist_select, names='value')
        self._dircontent.unobserve(self._on_dircontent_select, names='value')
        self._filename.unobserve(self._on_filename_change, names='value')

        try:
            # Fail early if the folder can not be read
            _ = os.listdir(path)

            # Set form values
            restricted_path = self._restrict_path(path)
            subpaths = get_subpaths(restricted_path)

            if os.path.splitdrive(subpaths[-1])[0]:
                # Add missing Windows drive letters
                drives = get_drive_letters()
                subpaths.extend(list(set(drives) - set(subpaths)))

            self._pathlist.options = subpaths
            self._pathlist.value = restricted_path
            #self._filename.options = tuple(set.union(set(filename),set(self._filename.options)))
            
            # file/folder real names
            dircontent_real_names = get_dir_contents(
                path,
                show_hidden=self._show_hidden,
                dir_icon=None,
                filter_pattern=self._filter_pattern,
                top_path=self._sandbox_path
            )

            # file/folder display names
            dircontent_display_names = get_dir_contents(
                path,
                show_hidden=self._show_hidden,
                dir_icon=self._dir_icon,
                dir_icon_append=self._dir_icon_append,
                filter_pattern=self._filter_pattern,
                top_path=self._sandbox_path
            )

            # Dict to map real names to display names
            self._map_name_to_disp = {
                real_name: disp_name
                for real_name, disp_name in zip(
                    dircontent_real_names,
                    dircontent_display_names
                )
            }

            # Dict to map display names to real names
            self._map_disp_to_name = {
                disp_name: real_name
                for real_name, disp_name in self._map_name_to_disp.items()
            }

            # Set _dircontent form value to display names
            self._dircontent.options = dircontent_display_names

            # If the value in the filename Text box equals a value in the
            # Select box and the entry is a file then select the entry.
            #self._dircontent.value = ()
            #for f in filename:
            #    if ((f in dircontent_real_names) and os.path.isfile(os.path.join(path, f))):
            #        self._dircontent.value += (self._map_name_to_disp[f],)

        except PermissionError:
            # Deselect the unreadable folder and generate a warning
            self._dircontent.value = ()
            warnings.warn(f'Permission denied for {path}', RuntimeWarning)

        # Reenable triggers
        self._pathlist.observe(self._on_pathlist_select, names='value')
        self._dircontent.observe(self._on_dircontent_select, names='value')
        self._filename.observe(self._on_filename_change, names='value')

    def _on_pathlist_select(self, change: Mapping[str, str]) -> None:
        """Handle selecting a path entry."""
        self._set_form_values(self._expand_path(change['new']), self._filename.options)

    def _on_dircontent_select(self, change: Mapping[str, str]) -> None:
        """Handle selecting a folder entry."""
        new_path = os.path.realpath(os.path.join(
            self._expand_path(self._pathlist.value),
            self._map_disp_to_name[change['new'][0]]
        ))

        # Check if folder or file
        if os.path.isdir(new_path):
            path = new_path
            filename = self._filename.options
        else:
            path = self._expand_path(self._pathlist.value)
            self._selected_filename = tuple([self._map_disp_to_name[f] for f in change['new']])
            filename = self._selected_filename

        self._set_form_values(path, filename)

    def _on_filename_change(self, change: Mapping[str, str]) -> None:
        """Handle filename field changes."""
        if self._filename.options != ():
            self._select.disabled = False
        else:
            self._select.disabled = True
        #self._set_form_values(self._expand_path(self._pathlist.value), change['new'])
        #self._set_form_values(self._expand_path(self._pathlist.value), self._filename.value)

    def _on_select_click(self, _b) -> None:
        """Handle select button clicks."""
        if self._gb.layout.display == 'none':
            # If not shown, open the dialog
            self._show_dialog()
        else:
            # If shown, close the dialog and apply the selection
            self._apply_selection()

            # Execute callback function
            if self._callback is not None:
                try:
                    self._callback(self)
                except TypeError:
                    # Support previous behaviour of not passing self
                    self._callback()

    def _on_add_click(self, _b) -> None:
        """Handle add button clicks."""
        # If shown, close the dialog and apply the selection
        self._apply_add()
        if self._filename.options != ():
            self._select.disabled = False

        # Execute callback function
        if self._callback is not None:
            try:
                self._callback(self)
            except TypeError:
                # Support previous behaviour of not passing self
                self._callback()

    def _on_remove_click(self, _b) -> None:
        """Handle add button clicks."""
        # If shown, close the dialog and apply the selection
        self._apply_rm()

        # Execute callback function
        if self._callback is not None:
            try:
                self._callback(self)
            except TypeError:
                # Support previous behaviour of not passing self
                self._callback()

    def _show_dialog(self) -> None:
        """Show the dialog."""
        # Show dialog and cancel button
        self._gb.layout.display = None
        self._cancel.layout.display = None
        self._add.layout.display = None
        self._remove.layout.display = None

        # Show the form with the correct path and filename
        if ((self._selected_path is not None) and (self._selected_filename != ())):
            path = self._selected_path
            filename = self._selected_filename
        else:
            path = self._default_path
            filename = self._default_filename

        self._set_form_values(path, filename)
        if self._filename.options == ():
            self._select.disabled = True

    def _apply_add(self) -> None:
        """Add element to the selection list."""
        path = self._expand_path(self._pathlist.value)
        if self._filter_pattern and isinstance(self._filter_pattern, str):
            oldpaths = tuple([os.path.join(path, f) for f in self._selected_filename if match_item(f, self._filter_pattern)])
        else:
            oldpaths = tuple([os.path.join(path, f) for f in self._selected_filename])
        self._filename.options = tuple(set.union(set(oldpaths),set(self._filename.options)))

    def _apply_rm(self) -> None:
        """Remove element to the selection list."""
        self._filename.options = tuple(set(self._filename.options) - set(list(self._filename.value)))
        self._set_form_values(self._expand_path(self._pathlist.value), self._filename.options)

    def _apply_selection(self) -> None:
        """Close the dialog and apply the selection."""
        self._selected_path = self._expand_path(self._pathlist.value)
        self._selected_filename = self._filename.options

        if ((self._selected_path is not None) and (self._selected_filename != ())):
            selected = tuple([os.path.join(self._selected_path, f) for f in self._selected_filename])
            self._gb.layout.display = 'none'
            self._cancel.layout.display = 'none'
            self._add.layout.display = 'none'
            self._select.description = self._change_desc
            self._select.disabled = False

            if all([os.path.isfile(f) for f in selected]):
                self._label.value = self._LBL_TEMPLATE.format("<style>p{line-height:10px;word-wrap: break-word}</style><p> " + f'{"<p>".join(selected)}', 'green')
            else:
                self._label.value = self._LBL_TEMPLATE.format('some paths are not files', 'orange')

    def _on_cancel_click(self, _b) -> None:
        """Handle cancel button clicks."""
        self._gb.layout.display = 'none'
        self._cancel.layout.display = 'none'
        self._add.layout.display = 'none'
        self._remove.layout.display = 'none'
        self._select.disabled = False

    def _expand_path(self, path) -> str:
        """Calculate the full path using the sandbox path."""
        if self._sandbox_path:
            path = os.path.join(self._sandbox_path, path.lstrip(os.sep))

        return path

    def _restrict_path(self, path) -> str:
        """Calculate the sandboxed path using the sandbox path."""
        if self._sandbox_path == os.sep:
            pass
        elif self._sandbox_path == path:
            path = os.sep
        elif self._sandbox_path:
            if os.path.splitdrive(self._sandbox_path)[0] and len(self._sandbox_path) == 3:
                # If the value is 'c:\\', strip 'c:' so we retain the leading os.sep char
                path = strip_parent_path(path, os.path.splitdrive(self._sandbox_path)[0])
            else:
                path = strip_parent_path(path, self._sandbox_path)

        return path

    def reset(self, path: Optional[str] = None, filename: Optional[Tuple[str]] = None) -> None:
        """Reset the form to the default path and filename."""
        # Check if path and sandbox_path align
        if path is not None and self._sandbox_path and not has_parent_path(normalize_path(path), self._sandbox_path):
            raise ParentPathError(path, self._sandbox_path)

        # Verify the filename is valid
        if filename != () and not all([is_valid_filename(f) for f in filename]):
            raise InvalidFileNameError(filename)

        # Remove selection
        self._selected_path = None
        self._selected_filename = tuple([])

        # Hide dialog and cancel button
        self._gb.layout.display = 'none'
        self._cancel.layout.display = 'none'
        self._add.layout.display = 'none'
        self._remove.layout.display = 'none'

        # Reset select button and label
        self._select.description = self._select_desc
        self._select.disabled = False
        self._label.value = self._LBL_TEMPLATE.format(self._LBL_NOFILE, 'black')

        if path is not None:
            self._default_path = normalize_path(path)

        if filename != ():
            self._default_filename = filename

        self._set_form_values(self._default_path, self._default_filename)

        # Use the defaults as the selected values
        if self._select_default:
            self._apply_selection()

    def refresh(self) -> None:
        """Re-render the form."""
        self._set_form_values(self._expand_path(self._pathlist.value), self._filename.options)

    @property
    def show_hidden(self) -> bool:
        """Get _show_hidden value."""
        return self._show_hidden

    @show_hidden.setter
    def show_hidden(self, hidden: bool) -> None:
        """Set _show_hidden value."""
        self._show_hidden = hidden
        self.refresh()

    @property
    def dir_icon(self) -> Optional[str]:
        """Get dir icon value."""
        return self._dir_icon

    @dir_icon.setter
    def dir_icon(self, dir_icon: Optional[str]) -> None:
        """Set dir icon value."""
        self._dir_icon = dir_icon
        self.refresh()

    @property
    def dir_icon_append(self) -> bool:
        """Get dir icon value."""
        return self._dir_icon_append

    @dir_icon_append.setter
    def dir_icon_append(self, dir_icon_append: bool) -> None:
        """Prepend or append the dir icon."""
        self._dir_icon_append = dir_icon_append
        self.refresh()

    @property
    def rows(self) -> int:
        """Get current number of rows."""
        return self._dircontent.rows

    @rows.setter
    def rows(self, rows: int) -> None:
        """Set number of rows."""
        self._dircontent.rows = rows

    @property
    def title(self) -> str:
        """Get the title."""
        return self._title.value

    @title.setter
    def title(self, title: str) -> None:
        """Set the title."""
        self._title.value = title

        if title == '':
            self._title.layout.display = 'none'
        else:
            self._title.layout.display = None

    @property
    def default(self) -> str:
        """Get the default value."""
        return os.path.join(self._default_path, self._default_filename)

    @property
    def default_path(self) -> str:
        """Get the default_path value."""
        return self._default_path

    @default_path.setter
    def default_path(self, path: str) -> None:
        """Set the default_path."""
        # Check if path and sandbox_path align
        if self._sandbox_path and not has_parent_path(normalize_path(path), self._sandbox_path):
            raise ParentPathError(path, self._sandbox_path)

        self._default_path = normalize_path(path)
        self._set_form_values(self._default_path, self._filename.options)

    @property
    def default_filename(self) -> str:
        """Get the default_filename value."""
        return self._default_filename

    @default_filename.setter
    def default_filename(self, filename: str) -> None:
        """Set the default_filename."""
        # Verify the filename is valid
        if not is_valid_filename(filename):
            raise InvalidFileNameError(filename)

        self._default_filename = filename
        self._set_form_values(self._expand_path(self._pathlist.value), self._default_filename)

    @property
    def sandbox_path(self) -> Optional[str]:
        """Get the sandbox_path."""
        return self._sandbox_path

    @sandbox_path.setter
    def sandbox_path(self, sandbox_path: str) -> None:
        """Set the sandbox_path."""
        # Check if path and sandbox_path align
        if sandbox_path and not has_parent_path(self._default_path, normalize_path(sandbox_path)):
            raise ParentPathError(self._default_path, sandbox_path)

        self._sandbox_path = normalize_path(sandbox_path) if sandbox_path is not None else None

        # Reset the dialog
        self.reset()

    @property
    def filter_pattern(self) -> Optional[Sequence[str]]:
        """Get file name filter pattern."""
        return self._filter_pattern

    @filter_pattern.setter
    def filter_pattern(self, filter_pattern: Optional[Sequence[str]]) -> None:
        """Set file name filter pattern."""
        self._filter_pattern = filter_pattern
        self.refresh()

    @property
    def value(self) -> Optional[Tuple[str]]:
        """Get selected value."""
        return self.selected

    @property
    def selected(self) -> Optional[str]:
        """Get selected value."""
        selected = ()

        if ((self._selected_path is not None) and (self._selected_filename != ())):
            selected = tuple([os.path.join(self._selected_path, f) for f in self._selected_filename])

        return selected

    @property
    def selected_path(self) -> Optional[str]:
        """Get selected_path value."""
        return self._selected_path

    @property
    def selected_filename(self) -> Optional[Tuple[str]]:
        """Get the selected_filename."""
        return self._selected_filename

    def __repr__(self) -> str:
        """Build string representation."""
        properties = f"path='{self._default_path}'"
        properties += f", filename='{self._default_filename}'"
        properties += f", title='{self._title.value}'"
        properties += f", show_hidden={self._show_hidden}"
        properties += f", select_desc='{self._select_desc}'"
        properties += f", change_desc='{self._change_desc}'"
        properties += f", select_default={self._select_default}"
        properties += f", dir_icon_append={self._dir_icon_append}"

        if self._sandbox_path is not None:
            properties += f", sandbox_path='{self._sandbox_path}'"

        if self._dir_icon:
            properties += f", dir_icon='{self._dir_icon}'"

        if self._filter_pattern:
            if isinstance(self._filter_pattern, str):
                properties += f", filter_pattern='{self._filter_pattern}'"
            else:
                properties += f", filter_pattern={self._filter_pattern}"

        return f"{self.__class__.__name__}({properties})"

    def register_callback(self, callback: Callable[[Optional['FileChooser']], None]) -> None:
        """Register a callback function."""
        self._callback = callback

    def get_interact_value(self) -> Optional[Tuple[str]]:
        """Return the value which should be passed to interactive functions."""
        return self.selected
