"""Graphical interface of the audio leveler (PyQt6 + pyqtgraph).

Every user-visible text is written in English and wrapped in ``tr()`` so it can
be extracted with ``pylupdate6`` and translated with Qt Linguist (see
``translations.py`` for the workflow). Changing the language in the "Language"
menu re-installs the translator and calls ``_retranslate()``.

The window lets you load a recording, view its waveform like Audacity, navigate
the timeline, analyze where the volume is too low, preview the planned
correction as a mask, listen to the A/B comparison (original / corrected) and
export a new WAV without ever modifying the original file.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

import pyqtgraph as pg

from . import translations
from .analysis import (
    AnalysisResult,
    Settings,
    WaveLevels,
    analyze,
    compute_gain,
    compute_stats,
    detect_segments,
    export_corrected,
    predicted_peak_db,
)
from .audiofile import AudioError, AudioFile, Cancelled
from .profiles import ProfileManager

COL_ORIG = (90, 150, 255, 120)      # blue: original waveform
COL_CORR = (255, 170, 60, 70)       # orange: planned correction
COL_GAIN = (80, 220, 130, 220)      # green: gain curve
COL_SEG = (255, 200, 60, 45)        # background of the detected segments
COL_PLAY = (255, 80, 80, 220)       # playback head

CH_AXIS = {"color": "#9fb0c8", "font-size": "10pt"}
GAIN_AXIS = {"color": "#9fb0c8", "font-size": "9pt"}


# --------------------------------------------------------------------------- threads
class Worker(QtCore.QThread):
    """Runs a heavy function in the background, with progress and cancellation."""

    progressed = QtCore.pyqtSignal(float)
    statusChanged = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._fn(self.progressed.emit, self.statusChanged.emit, lambda: self._cancel)
        except Cancelled:
            pass
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------- time axis
class SeekViewBox(pg.ViewBox):
    """ViewBox that emits the position (in seconds) when the waveform is clicked.

    A single click on the plot returns the clicked instant; dragging still pans
    because pyqtgraph does not generate the click event once the mouse has moved
    far enough.
    """

    seeked = QtCore.pyqtSignal(float)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and not ev.double():
            self.seeked.emit(float(self.mapSceneToView(ev.scenePos()).x()))
            ev.accept()
            return
        super().mouseClickEvent(ev)


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            v = max(0.0, float(v))
            m, s = divmod(v, 60.0)
            h, m = divmod(int(m), 60)
            if h:
                out.append("%d:%02d:%02d" % (h, m, int(s)))
            elif spacing >= 1:
                out.append("%d:%02d" % (m, int(s)))
            else:
                out.append("%d:%04.1f" % (m, s))
        return out


# --------------------------------------------------------------------------- window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1360, 860)

        self.audio: AudioFile | None = None
        self.levels: WaveLevels | None = None
        self.res: AnalysisResult | None = None
        self.gain: np.ndarray | None = None
        self.settings = Settings()
        self.worker: Worker | None = None
        self.busy = False
        self.profiles = ProfileManager()
        self.settings_q = QtCore.QSettings("Leveler", "Leveler")
        self._last_dir = self.settings_q.value("last_dir", "", str)
        self._tempdir = tempfile.mkdtemp(prefix="leveler_")
        self._render_path: str | None = None
        self._render_dirty = True
        self._span = 1.0
        self._playing_path: str | None = None
        self._pending_seek: tuple[float, bool] | None = None
        self.lang_actions: list[tuple[str, QtGui.QAction]] = []
        self._select_mode = False
        self._select_region: pg.LinearRegionItem | None = None
        self._select_start: float | None = None
        self._history: list[tuple[list[Segment], np.ndarray]] = []  # (segments, gain)
        self._history_index: int = -1

        pg.setConfigOptions(antialias=False)
        pg.setConfigOption("background", "#161a20")
        pg.setConfigOption("foreground", "#c8ccd4")

        self._build_ui()
        self._build_audio()
        self._retranslate()
        self._update_enabled()

    # ------------------------------------------------------------------ construction
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        self._build_language_menu()
        root.addWidget(self._build_toolbar())

        body = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self._build_wave_area())
        body.addWidget(self._build_side_panel())
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 1)
        body.setSizes([1000, 340])
        root.addWidget(body, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.lbl_progress = QtWidgets.QLabel()
        bottom.addWidget(self.lbl_progress)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bar.setFormat("")
        bottom.addWidget(self.bar, 1)
        root.addLayout(bottom)

        self.status = self.statusBar()

        # --- shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, self.open_file)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+E"), self, self.export_file)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Space), self, self.toggle_play)

    def _build_language_menu(self):
        """Menu with one entry per available language (compiled ``.qm`` files)."""
        self.mnu_lang = self.menuBar().addMenu("")
        for code, name in translations.available():
            act = self.mnu_lang.addAction(name)
            act.setCheckable(True)
            act.setChecked(code == translations.current())
            act.triggered.connect(
                lambda _checked=False, c=code: self._change_language(c))
            self.lang_actions.append((code, act))

    def _build_toolbar(self):
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

        icon_path = os.path.join(os.path.dirname(__file__), "icons")

        self.act_open = self._make_action(icon_path, "open.svg", "Open…", self.open_file, "Ctrl+O")
        self.toolbar.addAction(self.act_open)

        # "Prepare correction" with hammer icon and text beside icon
        self.act_prepare = self._make_action(icon_path, "prepare.svg", "Prepare correction", self.prepare_render)
        btn_prepare = QtWidgets.QToolButton()
        btn_prepare.setDefaultAction(self.act_prepare)
        btn_prepare.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.addWidget(btn_prepare)
        self.toolbar.addSeparator()

        self.act_play = self._make_action(icon_path, "play.svg", "Play", self.toggle_play)
        self.toolbar.addAction(self.act_play)
        self.act_stop = self._make_action(icon_path, "stop.svg", "Stop", self.stop_play)
        self.toolbar.addAction(self.act_stop)
        self.lbl_listen = QtWidgets.QLabel()
        self.toolbar.addWidget(self.lbl_listen)
        self.cmb_ab = QtWidgets.QComboBox()
        self.cmb_ab.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cmb_ab.installEventFilter(self)
        self.cmb_ab.currentIndexChanged.connect(self._on_ab_changed)
        self.toolbar.addWidget(self.cmb_ab)
        self.toolbar.addSeparator()

        self.act_zoom_in = self._make_action(icon_path, "zoom_in.svg", "Zoom in", lambda: self._zoom(0.5))
        self.toolbar.addAction(self.act_zoom_in)
        self.act_zoom_out = self._make_action(icon_path, "zoom_out.svg", "Zoom out", lambda: self._zoom(2.0))
        self.toolbar.addAction(self.act_zoom_out)
        self.act_fit = self._make_action(icon_path, "fit.svg", "Fit all", self.fit_view)
        self.toolbar.addAction(self.act_fit)

        # Undo/Redo
        self.act_undo = self._make_action(icon_path, "undo.svg", self.tr("Undo"), self._undo, "Ctrl+Z")
        self.toolbar.addAction(self.act_undo)
        self.act_redo = self._make_action(icon_path, "redo.svg", self.tr("Redo"), self._redo, "Ctrl+Y")
        self.toolbar.addAction(self.act_redo)

        # Selection tool
        self.act_select = self._make_action(icon_path, "select.svg", "Select range", self._toggle_select_mode)
        self.act_select.setCheckable(True)
        self.toolbar.addAction(self.act_select)

        # Selection gain control (shown when selection exists)
        self.sp_sel_gain = QtWidgets.QDoubleSpinBox()
        self.sp_sel_gain.setRange(-24, 24)
        self.sp_sel_gain.setValue(0.0)
        self.sp_sel_gain.setSingleStep(0.5)
        self.sp_sel_gain.setDecimals(1)
        self.sp_sel_gain.setSuffix(" dB")
        self.sp_sel_gain.setFixedWidth(90)
        self.sp_sel_gain.setToolTip(self.tr("Gain to apply to the selected range"))
        self.sp_sel_gain.editingFinished.connect(self._apply_selection_gain)
        self.sp_sel_gain.hide()
        self.toolbar.addWidget(self.sp_sel_gain)

        self.toolbar.addSeparator()
        self.act_export = self._make_action(icon_path, "export.svg", "Export WAV…", self.export_file, "Ctrl+E")
        self.toolbar.addAction(self.act_export)

        self.lbl_file = QtWidgets.QLabel()
        self.lbl_file.setStyleSheet("color:#8fa4c8;")
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.lbl_file)
        return self.toolbar

    def _make_action(self, icon_path: str, icon_file: str, text: str, slot, shortcut: str = "") -> QtGui.QAction:
        icon = QtGui.QIcon(os.path.join(icon_path, icon_file))
        act = QtGui.QAction(icon, text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(shortcut)
        act.setToolTip(text)
        return act

    def _build_wave_area(self):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.pg = pg.GraphicsLayoutWidget()
        self.plots: list[pg.PlotItem] = []
        self.chplots: list[pg.PlotItem] = []
        self.curves: list[dict] = []
        self.playheads: list[pg.InfiniteLine] = []
        self.region_items: list[pg.LinearRegionItem] = []
        self.gcurve: pg.PlotDataItem | None = None
        lay.addWidget(self.pg, 1)

        # Channel 1 and Channel 2 with the original waveform + the correction mask
        for ch in range(2):
            plot = self._make_plot(ch, TimeAxis(orientation="bottom"))
            plot.setYRange(-1.05, 1.05, padding=0)
            plot.getViewBox().setAutoVisible(y=False)
            corr = pg.PlotDataItem(pen=pg.mkPen(COL_CORR[:3], width=1),
                                   brush=pg.mkBrush(*COL_CORR), fillLevel=0.0)
            corr.setZValue(1)
            orig = pg.PlotDataItem(pen=pg.mkPen(COL_ORIG[:3], width=1),
                                   brush=pg.mkBrush(*COL_ORIG), fillLevel=0.0)
            orig.setZValue(2)
            plot.addItem(corr)
            plot.addItem(orig)
            head = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                   pen=pg.mkPen(COL_PLAY[:3], width=1))
            head.setZValue(5)
            plot.addItem(head)
            plot.sigXRangeChanged.connect(self._on_view_changed)
            if self.chplots:
                plot.setXLink(self.chplots[0])
            self.chplots.append(plot)
            self.curves.append({"orig": orig, "corr": corr})
            self.playheads.append(head)

        # Bottom strip with the gain curve (dB)
        gplot = self._make_plot(2, TimeAxis(orientation="bottom"))
        gplot.setMaximumHeight(120)
        gplot.setYRange(-1, 30, padding=0)
        gplot.getViewBox().setAutoVisible(y=False)
        gplot.setXLink(self.chplots[0])
        gplot.setLabel("left", "dB", **GAIN_AXIS)
        gplot.sigXRangeChanged.connect(self._on_view_changed)
        self.gcurve = pg.PlotDataItem(pen=pg.mkPen(COL_GAIN[:3], width=1.4))
        gplot.addItem(self.gcurve)
        zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False,
                               pen=pg.mkPen("#556", style=Qt.PenStyle.DashLine))
        gplot.addItem(zero)
        ghead = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                pen=pg.mkPen(COL_PLAY[:3], width=1))
        gplot.addItem(ghead)
        self.playheads.append(ghead)
        self.gplot = gplot
        self.plots = self.chplots + [gplot]

        self.hbar = QtWidgets.QScrollBar(Qt.Orientation.Horizontal)
        self.hbar.setRange(0, 10000)
        self.hbar.valueChanged.connect(self._on_hbar)
        lay.addWidget(self.hbar)
        return box

    def _make_plot(self, row: int, axis: TimeAxis) -> pg.PlotItem:
        plot = self.pg.addPlot(row=row, col=0, axisItems={"bottom": axis},
                               viewBox=SeekViewBox())
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setMouseEnabled(x=True, y=False)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        vb = plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.installEventFilter(self)  # for selection mode
        vb.seeked.connect(self._seek_to)
        return plot

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Wheel:
            if isinstance(obj, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
                ev.ignore()
                return True

        # Handle mouse events for selection mode
        if self._select_mode and self.res is not None:
            if isinstance(obj, pg.ViewBox):
                if ev.type() == QtCore.QEvent.Type.GraphicsSceneMousePress:
                    if ev.button() == Qt.MouseButton.LeftButton:
                        self._selection_press(obj, ev)
                        return True
                elif ev.type() == QtCore.QEvent.Type.GraphicsSceneMouseMove:
                    if ev.buttons() & Qt.MouseButton.LeftButton:
                        self._selection_move(obj, ev)
                        return True
                elif ev.type() == QtCore.QEvent.Type.GraphicsSceneMouseRelease:
                    if ev.button() == Qt.MouseButton.LeftButton:
                        self._selection_release(obj, ev)
                        return True

        return super().eventFilter(obj, ev)

    def _selection_press(self, vb: pg.ViewBox, ev):
        """Start a new selection on mouse press."""
        pos = vb.mapSceneToView(ev.scenePos())
        x = pos.x()
        x = max(0.0, min(self.audio.duration, x))
        self._select_start = x
        # Create a temporary region
        self._select_region = pg.LinearRegionItem([x, x], movable=False,
                                                   brush=pg.mkBrush(255, 255, 0, 50),
                                                   pen=pg.mkPen("#ff0", width=1, style=Qt.PenStyle.DashLine))
        for plot in self.plots:
            plot.addItem(self._select_region)

    def _selection_move(self, vb: pg.ViewBox, ev):
        """Update selection region on mouse drag."""
        if self._select_region is None or self._select_start is None:
            return
        pos = vb.mapSceneToView(ev.scenePos())
        x = pos.x()
        x = max(0.0, min(self.audio.duration, x))
        start, end = sorted([self._select_start, x])
        self._select_region.setRegion([start, end])

    def _selection_release(self, vb: pg.ViewBox, ev):
        """Finalize selection on mouse release."""
        if self._select_region is None or self._select_start is None:
            return
        pos = vb.mapSceneToView(ev.scenePos())
        x = pos.x()
        x = max(0.0, min(self.audio.duration, x))
        start, end = sorted([self._select_start, x])

        # Remove temporary region
        for plot in self.plots:
            plot.removeItem(self._select_region)

        if end - start < 0.1:
            self._select_region = None
            self._select_start = None
            self.sp_sel_gain.hide()
            return

        # Create persistent selection region
        self._select_region = pg.LinearRegionItem([start, end], movable=True,
                                                   brush=pg.mkBrush(255, 255, 0, 50),
                                                   pen=pg.mkPen("#ff0", width=1))
        self._select_region.sigRegionChanged.connect(self._on_select_region_changed)
        for plot in self.plots:
            plot.addItem(self._select_region)

        self.sp_sel_gain.show()
        self.sp_sel_gain.setValue(0.0)
        self.sp_sel_gain.setFocus()
        self.status.showMessage(self.tr("Selection: %s – %s. Set gain and press Enter.")
                                % (self._mmss(start), self._mmss(end)))

    def _build_side_panel(self):
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(300)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # --- profiles
        self.grp_prof = QtWidgets.QGroupBox()
        lp = QtWidgets.QHBoxLayout(self.grp_prof)
        self.cmb_prof = QtWidgets.QComboBox()
        self.cmb_prof.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cmb_prof.installEventFilter(self)
        self.cmb_prof.setEditable(True)
        self.cmb_prof.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.cmb_prof.currentIndexChanged.connect(self._on_profile_selected)
        lp.addWidget(self.cmb_prof, 1)

        self.btn_save_prof = QtWidgets.QPushButton()
        self.btn_save_prof.setFixedWidth(40)
        self.btn_save_prof.clicked.connect(self._save_profile)
        lp.addWidget(self.btn_save_prof)

        self.btn_del_prof = QtWidgets.QPushButton()
        self.btn_del_prof.setFixedWidth(40)
        self.btn_del_prof.clicked.connect(self._delete_profile)
        lp.addWidget(self.btn_del_prof)
        lay.addWidget(self.grp_prof)

        # --- analysis summary
        self.grp_info = QtWidgets.QGroupBox()
        f = QtWidgets.QFormLayout(self.grp_info)
        self.lbl_ref = QtWidgets.QLabel("—")
        self.lbl_target = QtWidgets.QLabel("—")
        self.lbl_floor = QtWidgets.QLabel("—")
        self.lbl_peak = QtWidgets.QLabel("—")
        self.lbl_nseg = QtWidgets.QLabel("—")
        self.lbl_ref_title = QtWidgets.QLabel()
        self.lbl_target_title = QtWidgets.QLabel()
        self.lbl_floor_title = QtWidgets.QLabel()
        self.lbl_peak_title = QtWidgets.QLabel()
        self.lbl_nseg_title = QtWidgets.QLabel()
        f.addRow(self.lbl_ref_title, self.lbl_ref)
        f.addRow(self.lbl_target_title, self.lbl_target)
        f.addRow(self.lbl_floor_title, self.lbl_floor)
        f.addRow(self.lbl_peak_title, self.lbl_peak)
        f.addRow(self.lbl_nseg_title, self.lbl_nseg)
        lay.addWidget(self.grp_info)

        # --- Tabs for Parameters and Segments
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)

        # --- parameters tab
        self.pane_par = QtWidgets.QWidget()
        self.tabs.addTab(self.pane_par, "")
        lp_par = QtWidgets.QVBoxLayout(self.pane_par)
        lp_par.setContentsMargins(4, 4, 4, 4)

        self.grp_par = QtWidgets.QGroupBox()
        self.grp_par.setFlat(True)
        fp = QtWidgets.QFormLayout(self.grp_par)

        def spin(lo, hi, val, step, dec=1, suffix=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            s.installEventFilter(self)
            if suffix:
                s.setSuffix(suffix)
            return s

        self.sp_target = spin(-30, 12, self.settings.target_db, 0.5, 1, " dB")
        self.sp_boost = spin(0, 25, self.settings.global_boost_db, 0.5, 1, " dB")
        self.sp_boost.setToolTip(self.tr("Constant gain added to the entire recording. "
                                        "Useful when the reference level is too low or for voice-only recordings."))
        self.sp_maxgain = spin(0, 48, self.settings.max_gain_db, 1, 1, " dB")
        self.sp_slow = spin(1, 120, self.settings.slow_s, 1, 0, " s")
        self.sp_content = spin(5, 60, self.settings.content_floor_db, 1, 0, " dB")
        self.sp_ceiling = spin(-6, 0, self.settings.ceiling_db, 0.5, 1, " dBFS")

        self.chk_limiter = QtWidgets.QCheckBox()
        self.chk_limiter.setChecked(self.settings.limiter)
        self.chk_limiter.toggled.connect(self._on_params_changed)

        self.chk_att = QtWidgets.QCheckBox()
        self.chk_att.setChecked(self.settings.allow_attenuate)
        self.chk_att.toggled.connect(self._on_params_changed)
        self.sp_att = spin(0, 24, self.settings.max_atten_db, 1, 1, " dB")

        self.lbl_target_hint = QtWidgets.QLabel()
        self.lbl_target_hint.setWordWrap(True)
        self.lbl_target_hint.setStyleSheet("color:#8fa4c8; font-size:9pt;")

        self.lbl_target_title2 = QtWidgets.QLabel()
        self.lbl_boost_title = QtWidgets.QLabel()
        self.lbl_maxgain_title = QtWidgets.QLabel()
        self.lbl_slow_title = QtWidgets.QLabel()
        self.lbl_content_title = QtWidgets.QLabel()
        self.lbl_ceiling_title = QtWidgets.QLabel()
        self.lbl_att_title = QtWidgets.QLabel()

        fp.addRow(self.lbl_target_title2, self.sp_target)
        fp.addRow("", self.lbl_target_hint)
        fp.addRow(self.lbl_boost_title, self.sp_boost)
        fp.addRow(self.lbl_maxgain_title, self.sp_maxgain)
        fp.addRow(self.lbl_slow_title, self.sp_slow)
        fp.addRow(self.lbl_content_title, self.sp_content)
        fp.addRow(self.lbl_ceiling_title, self.sp_ceiling)
        fp.addRow("", self.chk_limiter)
        fp.addRow("", self.chk_att)
        fp.addRow(self.lbl_att_title, self.sp_att)
        lp_par.addWidget(self.grp_par)
        lp_par.addStretch(1)

        for w in (self.sp_target, self.sp_boost, self.sp_maxgain, self.sp_slow, self.sp_content,
                  self.sp_ceiling, self.sp_att):
            w.valueChanged.connect(self._on_params_changed)

        # --- detected segments tab
        self.pane_seg = QtWidgets.QWidget()
        self.tabs.addTab(self.pane_seg, "")
        ls = QtWidgets.QVBoxLayout(self.pane_seg)
        ls.setContentsMargins(4, 4, 4, 4)

        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.itemChanged.connect(self._on_seg_changed)
        ls.addWidget(self.tbl)
        self.btn_goto = QtWidgets.QPushButton()
        self.btn_goto.clicked.connect(self._goto_selected_segment)
        ls.addWidget(self.btn_goto)

        return panel

    # ------------------------------------------------------------------ language
    def _change_language(self, code: str):
        if code == translations.current():
            return
        translations.install(QtWidgets.QApplication.instance(), code)
        for c, act in self.lang_actions:
            act.setChecked(c == code)
        self._retranslate()

    def _retranslate(self):
        """Writes every visible text again in the currently installed language."""
        if self.audio is not None:
            self.setWindowTitle(self.tr("Leveler — %s") % os.path.basename(self.audio.path))
        else:
            self.setWindowTitle(self.tr("Leveler — volume correction for console recordings"))
        self.mnu_lang.setTitle(self.tr("&Language"))
        self.toolbar.setWindowTitle(self.tr("Main"))

        self.act_open.setText(self.tr("Open…"))
        self.act_open.setToolTip(self.tr("Open…"))
        self.act_prepare.setText(self.tr("Prepare correction"))
        self.act_prepare.setToolTip(self.tr("Prepare correction"))
        playing = (getattr(self, "player", None) is not None
                   and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)
        self.act_play.setText(self.tr("Pause") if playing else self.tr("Play"))
        self.act_play.setToolTip(self.tr("Pause") if playing else self.tr("Play"))
        self.act_stop.setText(self.tr("Stop"))
        self.act_stop.setToolTip(self.tr("Stop"))
        self.lbl_listen.setText(self.tr("  Listen: "))
        self.act_zoom_in.setText(self.tr("Zoom in"))
        self.act_zoom_in.setToolTip(self.tr("Zoom in"))
        self.act_zoom_out.setText(self.tr("Zoom out"))
        self.act_zoom_out.setToolTip(self.tr("Zoom out"))
        self.act_fit.setText(self.tr("Fit all"))
        self.act_fit.setToolTip(self.tr("Fit all"))
        self.act_export.setText(self.tr("Export WAV…"))
        self.act_export.setToolTip(self.tr("Export WAV…"))
        self.act_select.setText(self.tr("Select range"))
        self.act_select.setToolTip(self.tr("Select a range on the waveform to apply gain/attenuation"))

        idx = self.cmb_ab.currentIndex()
        self.cmb_ab.blockSignals(True)
        self.cmb_ab.clear()
        self.cmb_ab.addItems([self.tr("Original"), self.tr("Corrected")])
        self.cmb_ab.setCurrentIndex(max(0, idx))
        self.cmb_ab.blockSignals(False)

        if self.audio is not None:
            self.lbl_file.setText("  %s " % os.path.basename(self.audio.path))
        else:
            self.lbl_file.setText(" %s " % self.tr("(no file)"))

        for i, plot in enumerate(self.chplots):
            plot.setLabel("left", self.tr("Channel %d") % (i + 1), **CH_AXIS)
        self.gplot.setLabel("left", self.tr("dB"), **GAIN_AXIS)

        self.lbl_progress.setText(self.tr("Progress"))

        self.grp_prof.setTitle(self.tr("Profiles"))
        self.btn_save_prof.setToolTip(self.tr("Save current settings to a profile"))
        self.btn_save_prof.setText("💾")
        self.btn_del_prof.setToolTip(self.tr("Delete selected profile"))
        self.btn_del_prof.setText("🗑")
        self._refresh_profiles()

        self.grp_info.setTitle(self.tr("Analysis"))
        self.lbl_ref_title.setText(self.tr("Reference level (music):"))
        self.lbl_target_title.setText(self.tr("Target level (voice):"))
        self.lbl_floor_title.setText(self.tr("Noise floor:"))
        self.lbl_peak_title.setText(self.tr("Predicted peak after correction:"))
        self.lbl_nseg_title.setText(self.tr("Segments with low volume:"))

        self.tabs.setTabText(0, self.tr("Correction parameters"))
        self.tabs.setTabText(1, self.tr("Detected segments"))

        self.grp_par.setTitle("")
        self.lbl_target_title2.setText(self.tr("Target level (voice):"))
        self.lbl_boost_title.setText(self.tr("Global boost:"))
        self.lbl_target_hint.setText(self.tr(
            "Level the voice is taken to, in dB relative to the music: "
            "0 dB = same as the music, negative = quieter, positive = louder."))
        self.lbl_maxgain_title.setText(self.tr("Maximum gain:"))
        self.lbl_slow_title.setText(self.tr("Region size:"))
        self.lbl_content_title.setText(self.tr("Content threshold:"))
        self.lbl_ceiling_title.setText(self.tr("Ceiling:"))
        self.chk_limiter.setText(self.tr("Enable limiter"))
        self.chk_att.setText(self.tr("Allow lowering volume"))
        self.lbl_att_title.setText(self.tr("Maximum attenuation:"))

        self.tbl.setHorizontalHeaderLabels(
            [self.tr("Apply"), self.tr("Start"), self.tr("End"), self.tr("Gain")])
        self.btn_goto.setText(self.tr("Go to selected segment"))

        if self.res is not None:
            self._refresh_info()
            self._show_ready_status()
        else:
            self.status.showMessage(self.tr("Ready. Open a recording to begin."))

    # ------------------------------------------------------------------ audio backend
    def _build_audio(self):
        self.player = QMediaPlayer(self)
        self.out = QAudioOutput(self)
        self.out.setVolume(1.0)
        self.player.setAudioOutput(self.out)
        self.player.positionChanged.connect(self._on_position)
        self.player.playbackStateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_player_error)

    # ------------------------------------------------------------------ helpers
    def _busy(self, busy: bool, msg: str = ""):
        self.busy = busy
        if busy:
            self.bar.setValue(0)
            self.bar.setFormat("%p% — " + msg)
            self.status.showMessage(msg)
        else:
            self.bar.setValue(0)
            self.bar.setFormat("")
        self._update_enabled()

    def _start(self, fn, msg: str, on_done=None):
        if self.busy:
            return
        self._busy(True, msg)
        w = Worker(fn, self)
        w.progressed.connect(lambda p: self.bar.setValue(int(p * 1000)))
        w.statusChanged.connect(self._on_worker_status)
        w.failed.connect(self._on_worker_error)
        if on_done is not None:
            w.finished.connect(on_done)
        w.finished.connect(self._job_finished)
        self.worker = w
        w.start()

    def _on_worker_status(self, msg: str):
        self.bar.setFormat("%p% — " + msg)
        self.status.showMessage(msg)

    def _job_finished(self):
        w, self.worker = self.worker, None
        self._busy(False)
        if w is not None:
            w.deleteLater()

    def _on_worker_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, self.tr("Error"), msg)
        self.status.showMessage(self.tr("Error: ") + msg)
        self._busy(False)

    def _update_enabled(self):
        have = self.audio is not None and self.res is not None
        idle = not self.busy
        self.act_open.setEnabled(idle)
        self.act_prepare.setEnabled(have and idle)
        self.act_export.setEnabled(have and idle)
        self.act_play.setEnabled(have)
        self.act_stop.setEnabled(have)
        self.cmb_ab.setEnabled(have)
        self.act_zoom_in.setEnabled(have)
        self.act_zoom_out.setEnabled(have)
        self.act_fit.setEnabled(have)

    # ------------------------------------------------------------------ open and analyze
    def open_file(self):
        if self._last_dir and os.path.isdir(self._last_dir):
            default_dir = self._last_dir
        else:
            default_dir = os.path.expanduser("~/Música/Recordings")

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("Open recording"), default_dir,
            self.tr("Audio (*.wav *.flac *.mp3 *.aiff *.ogg);;All files (*)"))
        if not path:
            return

        # Remember the folder where we opened the file
        self._last_dir = os.path.dirname(path)
        self.settings_q.setValue("last_dir", self._last_dir)
        self._open_path(path)

    def _open_path(self, path: str):
        self.stop_play()
        if self.audio is not None:
            self.audio.close()
            self.audio = None
        self.levels = None
        self.res = None
        self.gain = None
        self._render_path = None
        self._render_dirty = True

        def job(progress, status, cancel):
            status(self.tr("Opening file…"))
            audio = AudioFile(path)                    # converts to float if needed
            status(self.tr("Building waveform…"))
            levels = WaveLevels(audio, progress=lambda p: progress(p * 0.55),
                                cancel=cancel)
            status(self.tr("Analyzing levels…"))
            res, gain = analyze(audio, self._settings_copy(),
                                progress=lambda p: progress(0.55 + p * 0.45),
                                cancel=cancel)
            self.audio = audio
            self.levels = levels
            self.res = res
            self.gain = gain

        self._start(job, self.tr("Analyzing “%s”…") % os.path.basename(path),
                    self._after_open)

    def _settings_copy(self) -> Settings:
        return Settings(**{k: getattr(self.settings, k)
                           for k in self.settings.__dataclass_fields__})

    def _after_open(self):
        if self.audio is None:
            return
        self.setWindowTitle(self.tr("Leveler — %s") % os.path.basename(self.audio.path))
        self.lbl_file.setText("  " + os.path.basename(self.audio.path) + " ")
        self._bind_audio()
        self._refresh_seg_table()
        self._refresh_info()
        self.fit_view()
        self._update_enabled()
        self._show_ready_status()
        # Initialize history with the initial state
        self._history = []
        self._history_index = -1
        self._save_history()

    def _show_ready_status(self):
        self.status.showMessage(self.tr(
            "Ready: %s. Click on the waveform to play from that point.")
            % os.path.basename(self.audio.path))

    def _bind_audio(self):
        """Fits the plots to the file that has just been loaded."""
        nch = min(2, self.audio.channels)
        for i, plot in enumerate(self.chplots):
            vis = i < nch
            plot.setVisible(vis)
            plot.setMaximumHeight(16777215 if vis else 0)
            plot.setMinimumHeight(60 if vis else 0)
            plot.setLimits(xMin=0.0, xMax=max(1e-3, self.audio.duration))
            if not vis:
                self.curves[i]["orig"].setData([], [])
                self.curves[i]["corr"].setData([], [])
        self._render_segment_regions()
        self.fit_view()
        self._redraw()

    # ------------------------------------------------------------------ navigation
    def _view_range(self):
        return self.plots[0].viewRange()[0]

    def _on_view_changed(self, *_):
        if self.audio is None or not self.plots:
            return
        x0, x1 = self._view_range()
        total = max(1e-6, self.audio.duration)
        self._span = max(1e-6, min(1.0, (x1 - x0) / total))
        self._schedule_redraw()
        self._sync_hbar(x0, x1)

    def _sync_hbar(self, x0, x1):
        total = max(1e-6, self.audio.duration)
        span = max(1e-6, min(1.0, (x1 - x0) / total))
        maxv = int(round(10000 * (1.0 - span)))
        self.hbar.blockSignals(True)
        self.hbar.setRange(0, max(1, maxv))
        self.hbar.setPageStep(max(1, int(10000 * span)))
        self.hbar.setValue(int(round(10000 * max(0.0, x0) / total)))
        self.hbar.blockSignals(False)

    def _on_hbar(self, v):
        if not self.plots or self.audio is None:
            return
        total = self.audio.duration
        x0 = min(total, max(0.0, v / 10000.0 * total))
        self._set_range(x0, x0 + self._span * total)

    def _set_range(self, x0, x1):
        t = self.audio.duration
        x0 = max(0.0, min(t, x0))
        x1 = max(x0 + 1e-3, min(t, x1))
        self.plots[0].setXRange(x0, x1, padding=0)

    def _zoom(self, factor):
        if not self.plots:
            return
        x0, x1 = self._view_range()
        c = 0.5 * (x0 + x1)
        half = max(0.05, 0.5 * (x1 - x0) * factor)
        self._set_range(c - half, c + half)

    def fit_view(self):
        if self.plots and self.audio:
            self._set_range(0.0, self.audio.duration)

    def _schedule_redraw(self):
        if getattr(self, "_timer", None) is None:
            self._timer = QtCore.QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setInterval(30)
            self._timer.timeout.connect(self._redraw)
        self._timer.start()

    # ------------------------------------------------------------------ drawing
    def _redraw(self):
        if self.audio is None or self.levels is None or not self.plots:
            return
        x0, x1 = self._view_range()
        x0 = max(0.0, x0)
        if x1 <= x0:
            return
        w = max(200, int(self.pg.width()))
        ceil_v = self.settings.ceiling
        nch = min(len(self.curves), self.audio.channels)
        for ch in range(nch):
            t, mn, mx = self.levels.envelope(self.audio, ch, x0, x1, w)
            if t.size == 0:
                self.curves[ch]["orig"].setData([], [])
                self.curves[ch]["corr"].setData([], [])
                continue
            x = np.concatenate([t, t[::-1]])
            yo = np.concatenate([mx, mn[::-1]])
            self.curves[ch]["orig"].setData(x, yo)
            if self.gain is not None and self.res is not None:
                f = np.power(10.0, self._gain_at(t) / 20.0)
                yc = np.concatenate([np.clip(mx * f, -ceil_v, ceil_v),
                                     np.clip(mn * f, -ceil_v, ceil_v)[::-1]])
                self.curves[ch]["corr"].setData(x, yc)
            else:
                self.curves[ch]["corr"].setData([], [])

        if self.gain is not None and self.res is not None and self.gcurve is not None:
            ts = self.res.times
            sel = (ts >= x0) & (ts <= x1)
            self.gcurve.setData(ts[sel], self.gain[sel])
        elif self.gcurve is not None:
            self.gcurve.setData([], [])

    def _gain_at(self, t: np.ndarray) -> np.ndarray:
        if self.res is None or self.gain is None or self.res.times.size == 0:
            return np.zeros_like(t)
        return np.interp(t, self.res.times, self.gain)

    # ------------------------------------------------------------------ segments
    def _render_segment_regions(self):
        for plot, it in self.region_items:
            plot.removeItem(it)
        self.region_items.clear()
        if self.res is None or self.audio is None or len(self.res.segments) > 300:
            return
        nch = min(len(self.chplots), self.audio.channels)
        for seg in self.res.segments:
            for plot in self.chplots[:nch]:
                it = pg.LinearRegionItem([seg.start_s, seg.end_s], movable=False,
                                         brush=pg.mkBrush(*COL_SEG))
                it.setZValue(0)
                plot.addItem(it)
                self.region_items.append((plot, it))

    def _refresh_seg_table(self):
        self.tbl.blockSignals(True)
        segs = self.res.segments if self.res else []
        self.tbl.setRowCount(len(segs))
        for r, seg in enumerate(segs):
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if seg.enabled else Qt.CheckState.Unchecked)
            self.tbl.setItem(r, 0, chk)
            self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(self._mmss(seg.start_s)))
            self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(self._mmss(seg.end_s)))
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem("%+.1f dB" % seg.gain_db))
        self.tbl.resizeColumnToContents(0)
        self.tbl.blockSignals(False)

    @staticmethod
    def _mmss(t: float) -> str:
        m, s = divmod(max(0.0, t), 60.0)
        h, m = divmod(int(m), 60)
        return ("%d:%02d:%02d" % (h, m, int(s))) if h else ("%d:%02d" % (m, int(s)))

    def _on_seg_changed(self, item):
        if self.res is None:
            return
        r = item.row()
        if r >= len(self.res.segments):
            return
        seg = self.res.segments[r]
        seg.enabled = self.tbl.item(r, 0).checkState() == Qt.CheckState.Checked
        self._save_history()
        self._recompute_gain(keep_segments=True)

    def _goto_selected_segment(self):
        r = self.tbl.currentRow()
        if self.res is None or r < 0 or r >= len(self.res.segments):
            return
        seg = self.res.segments[r]
        pad = max(2.0, 0.15 * seg.duration)
        self._set_range(seg.start_s - pad, seg.end_s + pad)

    # ------------------------------------------------------------------ undo/redo
    def _save_history(self):
        """Save current segments and gain state to history."""
        if self.res is None or self.gain is None:
            return
        # Deep copy segments and gain
        segs_copy = [Segment(
            i0=s.i0, i1=s.i1, start_s=s.start_s, end_s=s.end_s,
            level_db=s.level_db, gain_db=s.gain_db,
            trim_db=s.trim_db, enabled=s.enabled
        ) for s in self.res.segments]
        gain_copy = self.gain.copy()
        # Truncate history after current index
        self._history = self._history[:self._history_index + 1]
        self._history.append((segs_copy, gain_copy))
        self._history_index = len(self._history) - 1
        # Limit history size
        if len(self._history) > 50:
            self._history.pop(0)
            self._history_index -= 1
        self._update_undo_redo()

    def _update_undo_redo(self):
        self.act_undo.setEnabled(self._history_index > 0)
        self.act_redo.setEnabled(self._history_index < len(self._history) - 1)

    def _undo(self):
        if self._history_index <= 0 or self.res is None:
            return
        self._history_index -= 1
        self._restore_history()

    def _redo(self):
        if self._history_index >= len(self._history) - 1 or self.res is None:
            return
        self._history_index += 1
        self._restore_history()

    def _restore_history(self):
        segs, gain = self._history[self._history_index]
        self.res.segments = [Segment(
            i0=s.i0, i1=s.i1, start_s=s.start_s, end_s=s.end_s,
            level_db=s.level_db, gain_db=s.gain_db,
            trim_db=s.trim_db, enabled=s.enabled
        ) for s in segs]
        self.gain = gain.copy()
        self._refresh_seg_table()
        self._render_segment_regions()
        self._redraw()
        self._update_undo_redo()
        msg = self.tr("Undo") if self._history_index < len(self._history) - 1 else self.tr("Redo")
        self.status.showMessage(msg)

    # ------------------------------------------------------------------ manual selection
    def _toggle_select_mode(self, checked: bool):
        self._select_mode = checked
        if checked:
            self.status.showMessage(self.tr("Select mode: drag on the waveform to choose a range"))
            # Change mouse mode to rectangle selection on all plots
            for plot in self.plots:
                vb = plot.getViewBox()
                vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self._clear_selection()
            self.status.showMessage(self.tr("Ready"))

    def _clear_selection(self):
        if self._select_region is not None:
            for plot in self.plots:
                plot.removeItem(self._select_region)
            self._select_region = None
            self._select_start = None
        self.sp_sel_gain.hide()
        self.sp_sel_gain.setValue(0.0)
        self.act_select.blockSignals(True)
        self.act_select.setChecked(False)
        self.act_select.blockSignals(False)
        for plot in self.plots:
            vb = plot.getViewBox()
            vb.setMouseMode(pg.ViewBox.PanMode)

    def _on_select_region_changed(self, region: pg.LinearRegionItem):
        if self.res is None:
            return
        r = region.getRegion()
        start, end = sorted(r)
        # Update the selection gain spinbox tooltip with current range
        self.sp_sel_gain.setToolTip(self.tr("Gain for range %s – %s") % (self._mmss(start), self._mmss(end)))

    def _apply_selection_gain(self, gain_db: float):
        """Creates/updates a segment for the current selection with the given gain."""
        if self.res is None or self._select_region is None:
            return
        r = self._select_region.getRegion()
        start, end = sorted(r)
        if end - start < 0.1:
            return
        # Create a new segment for this selection
        seg = Segment(
            i0=0, i1=0,  # will be set by segment_range
            start_s=start, end_s=end,
            level_db=0.0, gain_db=gain_db,
            trim_db=gain_db,  # use trim_db to apply the gain
            enabled=True
        )
        # Add to segments list
        self.res.segments.append(seg)
        # Recompute gain with the new segment
        self._save_history()
        self._recompute_gain(keep_segments=True)
        self._refresh_seg_table()
        self._render_segment_regions()
        self._redraw()
        # Clear selection after applying
        self.act_select.setChecked(False)
        self._toggle_select_mode(False)

    # ------------------------------------------------------------------ parameters
    def _refresh_profiles(self):
        self.cmb_prof.blockSignals(True)
        current = self.cmb_prof.currentText()
        self.cmb_prof.clear()
        self.cmb_prof.addItems(self.profiles.get_profile_names())
        self.cmb_prof.setCurrentText(current)
        self.cmb_prof.blockSignals(False)

    def _on_profile_selected(self, _):
        name = self.cmb_prof.currentText()
        s = self.profiles.get_settings(name)
        if s:
            self.settings = s
            self._update_spins()
            self._on_params_changed()

    def _update_spins(self):
        self.sp_target.blockSignals(True)
        self.sp_target.setValue(self.settings.target_db)
        self.sp_target.blockSignals(False)

        self.sp_boost.blockSignals(True)
        self.sp_boost.setValue(self.settings.global_boost_db)
        self.sp_boost.blockSignals(False)

        self.sp_maxgain.blockSignals(True)
        self.sp_maxgain.setValue(self.settings.max_gain_db)
        self.sp_maxgain.blockSignals(False)

        self.sp_slow.blockSignals(True)
        self.sp_slow.setValue(self.settings.slow_s)
        self.sp_slow.blockSignals(False)

        self.sp_content.blockSignals(True)
        self.sp_content.setValue(self.settings.content_floor_db)
        self.sp_content.blockSignals(False)

        self.sp_ceiling.blockSignals(True)
        self.sp_ceiling.setValue(self.settings.ceiling_db)
        self.sp_ceiling.blockSignals(False)

        self.chk_limiter.blockSignals(True)
        self.chk_limiter.setChecked(self.settings.limiter)
        self.chk_limiter.blockSignals(False)

        self.chk_att.blockSignals(True)
        self.chk_att.setChecked(self.settings.allow_attenuate)
        self.chk_att.blockSignals(False)

        self.sp_att.blockSignals(True)
        self.sp_att.setValue(self.settings.max_atten_db)
        self.sp_att.blockSignals(False)

    def _save_profile(self):
        name = self.cmb_prof.currentText().strip()
        if not name:
            return
        self.profiles.save_profile(name, self.settings)
        self._refresh_profiles()
        self.cmb_prof.setCurrentText(name)
        self.status.showMessage(self.tr("Profile saved: %s") % name)

    def _delete_profile(self):
        name = self.cmb_prof.currentText()
        if not name or name not in self.profiles.profiles:
            return
        res = QtWidgets.QMessageBox.question(
            self, self.tr("Delete profile"),
            self.tr("Are you sure you want to delete the profile “%s”?") % name)
        if res == QtWidgets.QMessageBox.StandardButton.Yes:
            self.profiles.delete_profile(name)
            self._refresh_profiles()
            self.status.showMessage(self.tr("Profile deleted: %s") % name)

    def _on_params_changed(self):
        self.settings.target_db = self.sp_target.value()
        self.settings.global_boost_db = self.sp_boost.value()
        self.settings.max_gain_db = self.sp_maxgain.value()
        self.settings.slow_s = self.sp_slow.value()
        self.settings.content_floor_db = self.sp_content.value()
        self.settings.ceiling_db = self.sp_ceiling.value()
        self.settings.limiter = self.chk_limiter.isChecked()
        self.settings.allow_attenuate = self.chk_att.isChecked()
        self.settings.max_atten_db = self.sp_att.value()
        if self.audio is not None and self.res is not None:
            self._save_history()
            self._recompute_gain(keep_segments=False)
            ymax = max(27.0, self.settings.max_gain_db + 2)
            self.gplot.setYRange(-1, ymax, padding=0)

    def _recompute_gain(self, keep_segments: bool):
        if self.res is None:
            return
        s = self._settings_copy()
        compute_stats(self.res, s)
        self.gain = compute_gain(self.res, s, self.res.segments or None)
        if not keep_segments:
            self.res.segments = detect_segments(self.res, self.gain, s)
            self._refresh_seg_table()
            self._render_segment_regions()
        self._render_dirty = True
        self._refresh_info()
        self._redraw()

    def _refresh_info(self):
        if self.res is None:
            return
        s = self._settings_copy()
        self.lbl_ref.setText("%.1f dBFS" % self.res.ref_db)
        self.lbl_target.setText(self.tr("%+.1f dB  →  voice at %.1f dBFS")
                                % (s.target_db, self.res.target_db))
        self.lbl_floor.setText("%.1f dBFS" % self.res.floor_db)
        pk = predicted_peak_db(self.res, self.gain, s) if self.gain is not None else -120.0
        self.lbl_peak.setText("%.1f dBFS" % pk)
        self.lbl_nseg.setText(str(len(self.res.segments)))

    # ------------------------------------------------------------------ playback
    def _load_source(self, src: str, t: float, play: bool):
        """Loads a source and jumps to the instant ``t`` as soon as it is ready.

        Qt ignores ``setPosition()`` while the media is loading, so the jump is
        postponed until the player reports that it is ready.
        """
        loaded = self.player.mediaStatus() not in (
            QMediaPlayer.MediaStatus.NoMedia,
            QMediaPlayer.MediaStatus.LoadingMedia,
            QMediaPlayer.MediaStatus.InvalidMedia,
        )
        if src == self._playing_path and loaded and self._pending_seek is None:
            self.player.setPosition(int(t * 1000))
            if play:
                self.player.play()
            return
        self._pending_seek = (t, play)
        self._playing_path = src
        self.player.setSource(QUrl.fromLocalFile(src))

    def _on_media_status(self, status):
        if self._pending_seek is None:
            return
        if status not in (QMediaPlayer.MediaStatus.LoadedMedia,
                          QMediaPlayer.MediaStatus.BufferedMedia):
            return
        t, play = self._pending_seek
        self._pending_seek = None
        self.player.setPosition(int(t * 1000))
        if play:
            self.player.play()
            self.act_play.setText(self.tr("Pause"))

    def _seek_to(self, t: float):
        """Moves to the instant clicked on the waveform and plays from there."""
        if self.audio is None:
            return
        t = max(0.0, min(self.audio.duration, t))
        for h in self.playheads:
            h.setPos(t)
        src = self._current_source()
        if src is None:
            return          # "_current_source" already explained why in the status bar
        self._load_source(src, t, True)
        self.act_play.setText(self.tr("Pause"))
        self.status.showMessage(self.tr("Playing from %s") % self._mmss(t))

    def toggle_play(self):
        if self.audio is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.act_play.setText(self.tr("Play"))
            return
        src = self._current_source()
        if src is None:
            return
        self._load_source(src, self.player.position() / 1000.0, True)
        self.act_play.setText(self.tr("Pause"))

    def stop_play(self):
        if getattr(self, "player", None) is None:
            return
        self.player.stop()
        self.act_play.setText(self.tr("Play"))

    def _current_source(self) -> str | None:
        if self.cmb_ab.currentIndex() == 0:
            return self.audio.path
        if self._render_path and os.path.exists(self._render_path) and not self._render_dirty:
            return self._render_path
        self.status.showMessage(self.tr(
            "The correction is not ready yet; use “Prepare correction”."))
        return None

    def _on_ab_changed(self, idx):
        if self.audio is None:
            return
        if idx == 1 and not (self._render_path and os.path.exists(self._render_path)
                             and not self._render_dirty):
            self.cmb_ab.blockSignals(True)
            self.cmb_ab.setCurrentIndex(0)
            self.cmb_ab.blockSignals(False)
            self.status.showMessage(self.tr(
                "Press “Prepare correction” before listening to the corrected version."))
            return
        was = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        src = self._current_source()
        if src is None:
            return
        self._load_source(src, self.player.position() / 1000.0, was)
        if was:
            self.act_play.setText(self.tr("Pause"))

    def _on_position(self, ms: int):
        t = ms / 1000.0
        for h in self.playheads:
            h.setPos(t)
        if self.audio is not None:
            x0, x1 = self._view_range()
            if t < x0 or t > x1 - 0.02 * (x1 - x0):
                span = x1 - x0
                self._set_range(t - span * 0.1, t + span * 0.9)

    def _on_state(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.act_play.setText(self.tr("Play"))

    def _on_player_error(self, err, msg):
        if err != QMediaPlayer.Error.NoError and msg:
            self.status.showMessage(self.tr("Player: ") + msg)

    # ------------------------------------------------------------------ render and export
    def prepare_render(self):
        if self.audio is None or self.gain is None:
            return
        out = os.path.join(self._tempdir, "corrected_preview.wav")
        audio, res, gain, s = self.audio, self.res, self.gain.copy(), self._settings_copy()

        def job(progress, status, cancel):
            status(self.tr("Rendering preview…"))
            export_corrected(audio, res, gain, s, out, bits=16,
                             progress=progress, cancel=cancel)
            self._render_path = out
            self._render_dirty = False

        self._start(job, self.tr("Preparing the corrected version for listening…"),
                    self._after_render)

    def _after_render(self):
        self.status.showMessage(self.tr(
            "Correction ready. Choose “Corrected” to listen to the A/B comparison."))
        if self.cmb_ab.currentIndex() == 1:
            self._on_ab_changed(1)

    def export_file(self):
        if self.audio is None or self.gain is None:
            return
        base = os.path.splitext(os.path.basename(self.audio.path))[0] + "_nivelado.wav"
        default = os.path.join(os.path.dirname(self.audio.path), base)
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, self.tr("Save corrected audio"), default, self.tr("WAV (*.wav)"))
        if not out:
            return
        audio, res, gain, s = self.audio, self.res, self.gain.copy(), self._settings_copy()

        def job(progress, status, cancel):
            status(self.tr("Exporting audio…"))
            export_corrected(audio, res, gain, s, out, bits=16,
                             progress=progress, cancel=cancel)

        self._start(job, self.tr("Exporting “%s”…") % os.path.basename(out), None)
        self.worker.finished.connect(lambda: self.status.showMessage(
            self.tr("Exported: %s (the original file was not modified).") % out))

    # ------------------------------------------------------------------ closing
    def closeEvent(self, ev):
        try:
            if self.worker is not None:
                self.worker.cancel()
                self.worker.wait(5000)
        finally:
            if self.audio is not None:
                self.audio.close()
            if getattr(self, "player", None) is not None:
                self.player.stop()
            shutil.rmtree(self._tempdir, ignore_errors=True)
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Leveler")
    translations.install(app, translations.preferred())
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        QtCore.QTimer.singleShot(200, lambda: win._open_path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
