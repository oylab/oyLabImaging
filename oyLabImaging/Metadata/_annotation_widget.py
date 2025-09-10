# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "pyqt5",
#     "qtawesome",
# ]
# ///
from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qtawesome as qta
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector, RectangleSelector
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

# make axis where things will be drawn
matplotlib.use("Agg")


### make toolbar and special buttons for selecting gates

UNASSIGNED = "#CCCCCC"
ASSIGNED = "black"
HIGHLIGHTED = "red"
OFFSET = 0  # offset for text labels
CIRCLE_SIZE = 100  # size of scatter plot circles


class MainWidget(QDialog):
    """Widget for assigning positions to different groups.

    Parameters
    ----------
    unique_positions : np.ndarray
        Nx2 array of unique XY positions for each position name.
    posnames : list[str]
        List of position names corresponding to the rows in unique_positions.
    groups : dict[str, str], optional
        Optional initial mapping of position names to group names, by default None.
    """

    def __init__(self, unique_positions, posnames, groups=None) -> None:
        super().__init__()

        # bits of DATA ------------------
        # map of position name to group names
        self._groups: dict[str, str] = dict(groups) if groups else {}

        # Store original position data for reference
        self._unique_positions = unique_positions
        self._posnames = posnames

        self._active_gate = None
        self._original_item_text = None  # Store original text before editing
        self._point_cid = None  # Store point selector connection ID

        # ------------------- WIDGETS

        # group list_widget on the left with buttons
        self._add_group_btn = QPushButton("Add Group")
        self._remove_group_btn = QPushButton("Remove Group")
        self._group_list = QListWidget()
        for group in set(self._groups.values()):
            item = QListWidgetItem(group)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self._group_list.addItem(item)

        # Configure the list widget for editing
        self._group_list.setSelectionMode(QListWidget.SingleSelection)

        mpl_fig, ax = plt.subplots()
        self._axis = ax
        self._figure_canvas = FigureCanvasQTAgg(mpl_fig)
        toolbar = NavigationToolbar2QT(self._figure_canvas)

        # ------------------- setup

        ax.set_position([0.18, 0.18, 0.8, 0.8])
        ax.cla()

        # label each point with the position name (offset from circle center)
        for i, pos in enumerate(posnames):
            ax.text(
                unique_positions[i, 0] + OFFSET,
                unique_positions[i, 1] + OFFSET,
                pos,
                fontsize=6,
                ha="left",
                va="bottom",
            )

        # Store the scatter plot for later updates (initially gray for unassigned)
        self._scatter = ax.scatter(
            unique_positions[:, 0], unique_positions[:, 1], c=UNASSIGNED, s=CIRCLE_SIZE
        )
        self._figure_canvas.draw()

        # ----------------------- Actions and signals

        self.act_poly = toolbar.addAction(
            qta.icon("mdi.shape-polygon-plus"), "add polygon gate"
        )
        self.act_poly.triggered.connect(self._onpolybutton)  ##TODO

        self.act_point = toolbar.addAction(
            qta.icon("mdi.cursor-default-click"), "point selector"
        )
        self.act_point.triggered.connect(self._onpointbutton)

        self.act_rect = toolbar.addAction(
            qta.icon("mdi.selection"), "rectangle selector"
        )
        self.act_rect.triggered.connect(self._onrectbutton)

        # Connect button signals
        self._add_group_btn.clicked.connect(self._on_add_group)
        self._remove_group_btn.clicked.connect(self._on_remove_group)

        # Enable double-click editing on list items
        self._group_list.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Connect to item changed signal for validation
        self._group_list.itemChanged.connect(self._on_item_changed)

        # Connect to selection changed signal for color updates
        self._group_list.itemSelectionChanged.connect(self._on_group_selection_changed)

        # --------------------- LAYOUT

        # Left side layout with buttons and list
        left_side = QVBoxLayout()
        left_side.addWidget(self._add_group_btn)
        left_side.addWidget(self._remove_group_btn)
        left_side.addWidget(self._group_list)

        right_side = QVBoxLayout()
        right_side.addWidget(toolbar)
        right_side.addWidget(self._figure_canvas)

        layout = QHBoxLayout(self)
        layout.addLayout(left_side)
        layout.addLayout(right_side, 1)
        self._update_plot_colors()

    def _onpolybutton(self) -> None:
        """
        Callback for when span gate button is pushed, created a SpanSelector
        """
        self._clear_active_gate()
        self._active_gate = PolygonSelector(
            self._axis,
            self._onselectpoly,
            useblit=True,
            props=dict(alpha=0.5, color="tab:blue"),
        )

    def _onpointbutton(self) -> None:
        """
        Callback for when point selector button is pushed, enables point clicking
        """
        self._clear_active_gate()
        # Connect to mouse click events for point selection
        self._point_cid = self._figure_canvas.mpl_connect(
            "button_press_event", self._onpointclick
        )

    def _onrectbutton(self) -> None:
        """
        Callback for when rectangle selector button is pushed, creates a RectangleSelector
        """
        self._clear_active_gate()
        self._active_gate = RectangleSelector(
            self._axis,
            self._onselectrect,
            useblit=True,
            props=dict(alpha=0.5, color="tab:green"),
        )

    def _clear_active_gate(self) -> None:
        """Clear any active selection tool"""
        if self._active_gate:
            self._active_gate.clear()
            self._active_gate = None

        # Disconnect point selector if active
        if hasattr(self, "_point_cid") and self._point_cid:
            self._figure_canvas.mpl_disconnect(self._point_cid)
            self._point_cid = None

    def _update_plot_colors(self) -> None:
        """Update the scatter plot colors based on group assignments and current selection."""
        colors = []
        currently_selected_group = self._group_list.currentItem()
        selected_group_name = (
            currently_selected_group.text() if currently_selected_group else None
        )

        for pos_name in self._posnames:
            if pos_name in self._groups:
                # Point has been assigned to a group
                if self._groups[pos_name] == selected_group_name:
                    colors.append(HIGHLIGHTED)  # Member of currently selected group
                else:
                    colors.append(ASSIGNED)  # Assigned but not current group
            else:
                colors.append(UNASSIGNED)  # Not assigned to any group

        # Update scatter plot colors
        self._scatter.set_color(colors)
        self._figure_canvas.draw()

    def _onselectpoly(self, verts: Any) -> None:
        """Callback for when polygon selection is completed."""
        currently_selected_group = self._group_list.currentItem()
        if currently_selected_group is None:
            self._clear_active_gate()
            return

        # check which points are in the polygon
        path = Path(verts)
        points = self._unique_positions
        ind = np.nonzero(np.array(path.contains_points(points)))[0]
        group_name = currently_selected_group.text()

        self._assign_positions_to_group(ind.tolist(), group_name)

    def _onpointclick(self, event: Any) -> None:
        """Callback for when a point is clicked in point selection mode."""
        if event.inaxes != self._axis:
            return

        currently_selected_group = self._group_list.currentItem()
        if currently_selected_group is None:
            return

        # Find the closest point to the click
        click_point = np.array([event.xdata, event.ydata])
        distances = np.sqrt(np.sum((self._unique_positions - click_point) ** 2, axis=1))
        closest_idx = np.argmin(distances)

        # Convert the closest point and click to screen coordinates for pixel-based distance check
        closest_point_data = self._unique_positions[closest_idx]
        closest_point_screen = self._axis.transData.transform(closest_point_data)
        click_point_screen = self._axis.transData.transform(click_point)
        # Calculate distance in pixels
        pixel_distance = np.sqrt(np.sum((closest_point_screen - click_point_screen) ** 2))
        # Only select if click is reasonably close (within 5 pixels)
        if pixel_distance < 5:
            group_name = currently_selected_group.text()
            self._assign_positions_to_group([closest_idx], group_name)

    def _onselectrect(self, eclick: Any, erelease: Any) -> None:
        """Callback for when rectangle selection is completed."""
        currently_selected_group = self._group_list.currentItem()
        if currently_selected_group is None:
            self._clear_active_gate()
            return

        # Get rectangle bounds
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure proper ordering
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # Find points within rectangle
        points = self._unique_positions
        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )
        ind = np.nonzero(mask)[0]

        group_name = currently_selected_group.text()
        self._assign_positions_to_group(ind.tolist(), group_name)

    def _assign_positions_to_group(self, indices: list, group_name: str) -> None:
        """Helper method to assign positions to a group and update colors."""
        for i in indices:
            pos_name = self._posnames[i]
            self._groups[pos_name] = group_name

        # Update the plot colors
        self._update_plot_colors()

        self._clear_active_gate()
        #print("current groups", self._groups)

    def _on_add_group(self) -> None:
        """Add a new group to the list with a unique name."""
        # Get all existing names
        existing_names = {
            self._group_list.item(i).text() for i in range(self._group_list.count())
        }

        # Find a unique name
        counter = 1
        while f"Group {counter}" in existing_names:
            counter += 1

        item = QListWidgetItem(f"Group {counter}")
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self._group_list.addItem(item)

    def _on_remove_group(self) -> None:
        """Remove the selected group from the list."""
        current_row = self._group_list.currentRow()
        if current_row >= 0:
            item = self._group_list.item(current_row)
            group_name = item.text()

            # Remove group assignments
            positions_to_unassign = [
                pos for pos, grp in self._groups.items() if grp == group_name
            ]
            for pos in positions_to_unassign:
                del self._groups[pos]

            # Remove from list widget
            self._group_list.takeItem(current_row)

            # Update plot colors
            self._update_plot_colors()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle double-click on list item to enable editing."""
        # Store the original text before editing starts
        self._original_item_text = item.text()
        self._group_list.editItem(item)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Validate that the new item name is unique."""
        new_text = item.text().strip()
        old_text = self._original_item_text

        # Check if the new name already exists in other items
        for i in range(self._group_list.count()):
            other_item = self._group_list.item(i)
            if other_item != item and other_item.text() == new_text:
                # Name already exists, revert to original
                item.setText(old_text)
                self._original_item_text = None
                return

        # If name changed successfully, update group assignments
        if old_text and old_text != new_text:
            # Update group assignments
            for pos_name in list(self._groups.keys()):
                if self._groups[pos_name] == old_text:
                    self._groups[pos_name] = new_text

        # Clear the stored original text
        self._original_item_text = None

    def _on_group_selection_changed(self) -> None:
        """Handle group selection changes to update point colors."""
        self._update_plot_colors()

    def groups(self) -> dict[str, str]:
        """Return the current mapping of position names to group names."""
        return self._groups


if __name__ == "__main__":
    app = QApplication([])
    posnames = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6"]
    unique_positions = np.array(
        [[i % 6, i // 6] for i in range(len(posnames))]
    )  # 6 columns, 4 rows
    main_widget = MainWidget(unique_positions, posnames, groups={"A1": "Group 1"})
    main_widget.exec()
