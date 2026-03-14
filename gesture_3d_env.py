"""
========================================================================
GESTURE-CONTROLLED 3D CREATIVE ENVIRONMENT  v3
========================================================================
FIXES in this version:
  • Object MOVE: pinch-grab now properly tracks palm delta each frame
  • Drawing STOP: lifting index finger cleanly finalises the trail
  • Shape-to-object: recognised shape is placed and trail is cleared
  • ONBOARDING screen shows full reference before the app starts
  • Mode-switch timers are clearly separated and reset properly
  • Grab state is sticky (no jitter from gesture flicker)
  • Trail is drawn on screen so you can see what you're drawing

GESTURES (DRAW MODE)
  ✦ Index finger only      → draw trail in air
  ✦ Pinch (thumb+index)    → grab & move selected object
  ✦ Fist drag              → orbit camera
  ✦ Open hand 3 s          → switch to CREATE MODE
  ✦ Both hands spread/close→ scale entire scene
  ✦ Both hands rotate      → rotate entire scene
  ✦ Thumb-up               → cycle colour of selected object
  ✦ Peace-sign + shake     → delete selected object

GESTURES (CREATE MODE)
  ✦ 2 fingers → cube       ✦ 3 fingers → cylinder
  ✦ 4 fingers → sphere     ✦ pinch → grab/move block
  ✦ Both hands 3 s         → back to DRAW MODE

KEYBOARD
  ARROWS / mouse-drag  orbit camera
  scroll-wheel         zoom
  Z                    undo last object
  R                    reset scene
  Delete / X           delete selected
  ESC / Q              quit

pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy
========================================================================
"""

import sys, time, math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import (DOUBLEBUF, OPENGL, QUIT, KEYDOWN,
                            K_ESCAPE, K_q, K_z, K_r, K_x,
                            K_DELETE, K_UP, K_DOWN, K_LEFT, K_RIGHT,
                            MOUSEBUTTONDOWN, MOUSEMOTION, MOUSEBUTTONUP,
                            MOUSEWHEEL)
from OpenGL.GL  import *
from OpenGL.GLU import *

# ======================================================================
#  CONSTANTS
# ======================================================================

WIN_W, WIN_H         = 1280, 720
TARGET_FPS           = 60
COUNTDOWN_SEC        = 5
MODE_HOLD_SEC        = 3.0          # seconds to hold gesture for mode switch
DRAW_TRAIL_MAX       = 400
DRAW_TRAIL_MIN       = 20
GRAB_PINCH_THRESH    = 0.07         # normalised thumb-index dist for pinch
MOVE_SCALE           = 8.0          # how much screen-delta maps to world move
GRID_HALF            = 10
GRID_STEP            = 0.6
LERP                 = 0.20         # object position smoothing

# Per-shape colour palettes (8 colours each)
PALETTES = {
    "sphere":   [(0.20,0.60,1.00),(0.80,0.20,1.00),(0.20,0.90,0.80),
                 (1.00,0.40,0.40),(0.40,1.00,0.40),(1.00,0.80,0.20),
                 (0.60,0.60,1.00),(1.00,0.60,0.80)],
    "cube":     [(0.95,0.40,0.15),(0.15,0.70,0.95),(0.95,0.80,0.10),
                 (0.60,0.95,0.20),(0.95,0.20,0.60),(0.20,0.95,0.50),
                 (0.95,0.50,0.80),(0.50,0.20,0.95)],
    "cylinder": [(0.20,0.85,0.45),(0.85,0.20,0.45),(0.45,0.20,0.85),
                 (0.85,0.70,0.20),(0.20,0.85,0.85),(0.85,0.45,0.20),
                 (0.60,0.85,0.20),(0.20,0.45,0.85)],
    "pyramid":  [(0.90,0.85,0.10),(0.10,0.85,0.90),(0.90,0.10,0.85),
                 (0.50,0.90,0.10),(0.10,0.50,0.90),(0.90,0.10,0.50),
                 (0.70,0.70,0.10),(0.10,0.70,0.70)],
}

# ======================================================================
#  ENUMERATIONS
# ======================================================================

class AppState(Enum):
    ONBOARD    = auto()   # splash / reference screen
    WAITING    = auto()   # waiting for hand → countdown
    COUNTDOWN  = auto()   # 5-4-3-2-1 → START
    DRAW_MODE  = auto()
    CREATE_MODE = auto()

class ShapeType(Enum):
    SPHERE   = "sphere"
    CUBE     = "cube"
    CYLINDER = "cylinder"
    PYRAMID  = "pyramid"
    UNKNOWN  = "unknown"

class GestureType(Enum):
    NONE          = auto()
    ONE_FINGER    = auto()   # index only  → draw
    TWO_FINGERS   = auto()   # index + middle (V closed)
    THREE_FINGERS = auto()
    FOUR_FINGERS  = auto()
    OPEN_HAND     = auto()   # 5 fingers
    FIST          = auto()   # closed fist → orbit / grab
    PINCH         = auto()   # thumb + index close → grab / zoom
    THUMB_UP      = auto()
    PEACE         = auto()   # V spread → delete-shake

# ======================================================================
#  DATA CLASSES
# ======================================================================

@dataclass
class HandData:
    landmarks:   np.ndarray
    handedness:  str
    index_tip:   np.ndarray
    thumb_tip:   np.ndarray
    middle_tip:  np.ndarray
    wrist:       np.ndarray
    palm_center: np.ndarray
    gesture:     GestureType = GestureType.NONE
    pinch_dist:  float       = 0.0


_COUNTER = 0

@dataclass
class Object3D:
    shape:    ShapeType
    position: np.ndarray      # target (logical)
    _render:  np.ndarray      # smoothed render position
    rotation: np.ndarray      # Euler XYZ degrees
    scale:    float
    color:    tuple
    cidx:     int = 0
    oid:      int = 0

    def __post_init__(self):
        global _COUNTER
        _COUNTER += 1
        self.oid = _COUNTER

    def tick(self):
        self._render += (self.position - self._render) * LERP

    def cycle_color(self):
        pal = PALETTES.get(self.shape.value, [(1,1,1)])
        self.cidx = (self.cidx + 1) % len(pal)
        r,g,b = pal[self.cidx]
        self.color = (r,g,b,1.0)

    @property
    def pos(self) -> np.ndarray:
        return self._render

# ======================================================================
#  HAND TRACKER
# ======================================================================

class HandTracker:
    """MediaPipe Hands wrapper → list[HandData] per frame."""

    W=0;  T_CMC=1; T_MCP=2; T_IP=3;  T_TIP=4
    I_MCP=5; I_PIP=6; I_DIP=7;  I_TIP=8
    M_MCP=9; M_PIP=10; M_DIP=11; M_TIP=12
    R_MCP=13; R_PIP=14; R_DIP=15; R_TIP=16
    P_MCP=17; P_PIP=18; P_DIP=19; P_TIP=20

    def __init__(self, max_hands=2):
        self._mp  = mp.solutions.hands
        self._det = self._mp.Hands(
            static_image_mode=False, max_num_hands=max_hands,
            min_detection_confidence=0.65, min_tracking_confidence=0.55)

    def process(self, bgr: np.ndarray) -> List[HandData]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._det.process(rgb)
        out = []
        if not res.multi_hand_landmarks:
            return out
        for lmp, info in zip(res.multi_hand_landmarks, res.multi_handedness):
            lm = np.array([[p.x,p.y,p.z] for p in lmp.landmark], np.float32)
            pc = (lm[self.W] + lm[self.I_MCP] + lm[self.M_MCP] + lm[self.R_MCP]) / 4
            pd = float(np.linalg.norm(lm[self.T_TIP][:2] - lm[self.I_TIP][:2]))
            hd = HandData(
                landmarks=lm, handedness=info.classification[0].label,
                index_tip=lm[self.I_TIP], thumb_tip=lm[self.T_TIP],
                middle_tip=lm[self.M_TIP], wrist=lm[self.W],
                palm_center=pc, pinch_dist=pd)
            hd.gesture = self._classify(lm)
            out.append(hd)
        return out

    def _classify(self, lm) -> GestureType:
        # pinch first
        if np.linalg.norm(lm[self.T_TIP][:2]-lm[self.I_TIP][:2]) < GRAB_PINCH_THRESH:
            return GestureType.PINCH
        e = self._ext(lm)
        # thumb-up: only thumb extended
        if e[0] and sum(e[1:])==0:
            return GestureType.THUMB_UP
        # peace / two-fingers
        if e[1] and e[2] and not e[3] and not e[4]:
            spread = float(np.linalg.norm(lm[self.I_TIP][:2]-lm[self.M_TIP][:2]))
            return GestureType.PEACE if spread>0.07 else GestureType.TWO_FINGERS
        # fist
        if sum(e)==0:
            return GestureType.FIST
        # open hand
        if sum(e)>=4:
            return GestureType.OPEN_HAND
        n = sum(e[1:])
        if n==1 and e[1]:  return GestureType.ONE_FINGER
        if n==2:           return GestureType.TWO_FINGERS
        if n==3:           return GestureType.THREE_FINGERS
        if n==4:           return GestureType.FOUR_FINGERS
        return GestureType.NONE

    def _ext(self, lm) -> List[bool]:
        thumb = abs(lm[self.T_TIP][0]-lm[self.T_MCP][0]) > 0.04
        rest  = [(lm[t][1] < lm[p][1]-0.01)
                 for t,p in [(self.I_TIP,self.I_PIP),(self.M_TIP,self.M_PIP),
                             (self.R_TIP,self.R_PIP),(self.P_TIP,self.P_PIP)]]
        return [thumb]+rest

    def close(self): self._det.close()

# ======================================================================
#  GESTURE RECOGNIZER
# ======================================================================

class GestureRecognizer:
    """
    Converts per-frame HandData lists into high-level event dicts.

    Key design:
      - Pinch-grab is STICKY once started → stays until pinch opens
      - Grab delta computed as palm_center difference between frames
      - All deltas zeroed when gesture not active
    """

    def __init__(self):
        self._open_t: Optional[float] = None
        self._both_t: Optional[float] = None

        # Grab state
        self._grab_active   = False
        self._grab_prev_pos: Optional[np.ndarray] = None   # screen [0,1]

        # Orbit (fist drag)
        self._orbit_prev: Optional[np.ndarray] = None

        # Zoom (single pinch, no selection)
        self._zoom_prev_dist: Optional[float] = None

        # Two-hand
        self._two_prev_dist:  Optional[float] = None
        self._two_prev_angle: Optional[float] = None

        # Peace shake delete
        self._peace_xs  = deque(maxlen=24)
        self._peace_t:  Optional[float] = None

        # Thumb-up debounce
        self._thumb_fired = False

        # Create-mode spawn debounce
        self._last_create_g = None

    # ------------------------------------------------------------------
    def update(self, hands: List[HandData], has_sel: bool) -> dict:
        ev = dict(
            open_hold          = 0.0,    # seconds open hand held
            both_hold          = 0.0,    # seconds both hands visible
            is_drawing         = False,
            draw_pt            = None,   # (nx, ny) normalised
            grab_active        = False,
            grab_dx            = 0.0,    # world-unit delta
            grab_dy            = 0.0,
            zoom_delta         = 0.0,    # positive = zoom in
            scale_obj          = 1.0,    # factor for selected obj scale
            scene_scale        = 1.0,    # factor for global scale
            scene_rot_deg      = 0.0,    # degrees rotation
            orbit_dyaw         = 0.0,
            orbit_dpitch       = 0.0,
            create_gesture     = None,
            create_new         = False,  # True on first frame of new create gesture
            do_delete          = False,
            do_color_cycle     = False,
        )

        n = len(hands)

        # ── Hold timers ───────────────────────────────────────────────
        any_open = n>=1 and any(h.gesture==GestureType.OPEN_HAND for h in hands)
        if any_open:
            if self._open_t is None: self._open_t = time.time()
            ev["open_hold"] = time.time()-self._open_t
        else:
            self._open_t = None

        if n==2:
            if self._both_t is None: self._both_t = time.time()
            ev["both_hold"] = time.time()-self._both_t
        else:
            self._both_t = None

        if n==0:
            self._reset()
            return ev

        # ── SINGLE HAND ───────────────────────────────────────────────
        if n==1:
            self._two_prev_dist  = None
            self._two_prev_angle = None
            h = hands[0]

            if h.gesture == GestureType.ONE_FINGER:
                # Drawing mode
                ev["is_drawing"] = True
                ev["draw_pt"]    = (float(h.index_tip[0]), float(h.index_tip[1]))
                self._grab_active   = False
                self._grab_prev_pos = None
                self._orbit_prev    = None
                self._zoom_prev_dist = None

            elif h.gesture == GestureType.PINCH:
                # If there's a selected object → grab and move it
                # If no selection → zoom camera
                cur = h.palm_center[:2].copy()
                if has_sel:
                    self._grab_active = True
                    ev["grab_active"] = True
                    if self._grab_prev_pos is not None:
                        ev["grab_dx"] = float(cur[0]-self._grab_prev_pos[0]) * MOVE_SCALE
                        ev["grab_dy"] = float(-(cur[1]-self._grab_prev_pos[1])) * MOVE_SCALE
                    self._grab_prev_pos = cur
                else:
                    self._grab_active = False
                    self._grab_prev_pos = None
                    pd = h.pinch_dist
                    if self._zoom_prev_dist is not None and self._zoom_prev_dist>1e-4:
                        ev["zoom_delta"] = float((self._zoom_prev_dist-pd)*40.0)
                    self._zoom_prev_dist = pd
                self._orbit_prev = None

            elif h.gesture == GestureType.FIST:
                # Fist drag = orbit camera
                self._grab_active   = False
                self._grab_prev_pos = None
                self._zoom_prev_dist = None
                cur = h.palm_center[:2].copy()
                if self._orbit_prev is not None:
                    ev["orbit_dyaw"]   = float(cur[0]-self._orbit_prev[0]) * 220.0
                    ev["orbit_dpitch"] = float(cur[1]-self._orbit_prev[1]) * 110.0
                self._orbit_prev = cur

            elif h.gesture == GestureType.THUMB_UP:
                if not self._thumb_fired:
                    ev["do_color_cycle"] = True
                    self._thumb_fired = True
                self._grab_active = False
                self._grab_prev_pos = None
            else:
                self._thumb_fired = False
                if h.gesture != GestureType.PINCH:
                    self._grab_active = False
                    self._grab_prev_pos = None
                if h.gesture != GestureType.FIST:
                    self._orbit_prev = None

            # Peace shake → delete
            if h.gesture == GestureType.PEACE:
                self._peace_xs.append(float(h.index_tip[0]))
                if self._peace_t is None: self._peace_t = time.time()
                if len(self._peace_xs)==24 and time.time()-self._peace_t < 1.5:
                    if float(np.var(np.array(self._peace_xs))) > 0.0005:
                        ev["do_delete"] = True
                        self._peace_xs.clear(); self._peace_t = None
            else:
                self._peace_t = None

            # Create mode gesture emit
            ev["create_gesture"] = h.gesture
            new_g = h.gesture not in (GestureType.NONE, GestureType.OPEN_HAND,
                                       GestureType.FIST, GestureType.PINCH,
                                       GestureType.ONE_FINGER)
            if new_g and h.gesture != self._last_create_g:
                ev["create_new"] = True
            self._last_create_g = h.gesture
            return ev

        # ── TWO HANDS ─────────────────────────────────────────────────
        self._grab_active   = False
        self._grab_prev_pos = None
        self._orbit_prev    = None
        self._zoom_prev_dist = None

        h0, h1 = hands[0], hands[1]
        p0 = h0.palm_center[:2]
        p1 = h1.palm_center[:2]
        dist  = float(np.linalg.norm(p1-p0))
        angle = math.degrees(math.atan2(float(p1[1]-p0[1]), float(p1[0]-p0[0])))

        # Both pinch → scale selected object
        if h0.gesture==GestureType.PINCH and h1.gesture==GestureType.PINCH:
            if self._two_prev_dist is not None and self._two_prev_dist>1e-4:
                f = dist/self._two_prev_dist
                if 0.80<f<1.30: ev["scale_obj"] = f
            self._two_prev_dist  = dist
            self._two_prev_angle = None

        # Other two-hand → global scale + rotate
        else:
            if self._two_prev_dist is not None and self._two_prev_dist>1e-4:
                f = dist/self._two_prev_dist
                if 0.88<f<1.14: ev["scene_scale"] = f
            self._two_prev_dist = dist

            if self._two_prev_angle is not None:
                d = angle-self._two_prev_angle
                if d>180: d-=360
                if d<-180: d+=360
                if abs(d)<15: ev["scene_rot_deg"] = float(d*2.5)
            self._two_prev_angle = angle

        ev["create_gesture"] = h0.gesture
        return ev

    def _reset(self):
        self._grab_active    = False
        self._grab_prev_pos  = None
        self._orbit_prev     = None
        self._zoom_prev_dist = None
        self._two_prev_dist  = None
        self._two_prev_angle = None

    def reset_timers(self):
        self._open_t = None
        self._both_t = None

# ======================================================================
#  SHAPE RECOGNIZER
# ======================================================================

class ShapeRecognizer:
    """2-D finger trail → ShapeType via resampling + curvature analysis."""

    N = 64

    def recognise(self, pts: List[Tuple[float,float]]) -> ShapeType:
        if len(pts) < DRAW_TRAIL_MIN:
            return ShapeType.UNKNOWN
        arr = np.array(pts, np.float32)
        arr = self._smooth(arr, 5)
        arr = self._resample(arr, self.N)

        mn,mx = arr.min(0), arr.max(0)
        w = float(mx[0]-mn[0])+1e-6
        h = float(mx[1]-mn[1])+1e-6
        asp = w/h
        diag = math.hypot(w,h)
        end  = float(np.linalg.norm(arr[0]-arr[-1]))
        closed = (end/diag) < 0.30

        pts_i = ((arr-mn)/max(w,h)*600).astype(np.int32)
        hull  = cv2.convexHull(pts_i)
        fill  = float(cv2.contourArea(hull)+1) / float(600*600*(w/max(w,h))*(h/max(w,h))+1)
        cors  = self._corners(arr)
        totl  = float(np.sum(np.linalg.norm(np.diff(arr,axis=0),axis=1)))+1e-6
        straight = end/totl

        if not closed:
            return ShapeType.CYLINDER if straight>0.38 else ShapeType.UNKNOWN
        if cors==3 or (cors<=4 and fill<0.60 and asp<1.8):
            return ShapeType.PYRAMID
        if cors==4 and 0.3<asp<3.0 and fill>0.58:
            return ShapeType.CUBE
        if fill>0.48 and cors<=3:
            return ShapeType.SPHERE
        if cors>=4:
            return ShapeType.CUBE
        return ShapeType.SPHERE

    def _smooth(self,arr,k):
        kern=np.ones(k)/k
        return np.stack([np.convolve(arr[:,0],kern,'valid'),
                         np.convolve(arr[:,1],kern,'valid')],1)

    def _resample(self,arr,n):
        d=np.linalg.norm(np.diff(arr,axis=0),axis=1)
        c=np.concatenate([[0],np.cumsum(d)])
        t=np.linspace(0,c[-1],n)
        return np.stack([np.interp(t,c,arr[:,0]),np.interp(t,c,arr[:,1])],1)

    def _corners(self,arr,thr=32.0):
        v=np.diff(arr,axis=0); nv=np.linalg.norm(v,axis=1,keepdims=True)+1e-8
        u=v/nv; dots=np.clip(np.sum(u[:-1]*u[1:],axis=1),-1,1)
        angs=np.degrees(np.arccos(dots))
        c=0; pk=False
        for a in angs:
            if a>thr:
                if not pk: c+=1; pk=True
            else: pk=False
        return c

# ======================================================================
#  SCENE 3D
# ======================================================================

class Scene3D:
    """Owns object list, global transform, camera state, undo stack."""

    def __init__(self):
        self.objects:    List[Object3D] = []
        self.sel_id:     Optional[int]  = None
        self._undo:      List[Object3D] = []
        self.g_rot_y:    float = 0.0
        self.g_scale:    float = 1.0
        self.cam_yaw:    float = 30.0
        self.cam_pitch:  float = -28.0
        self.cam_dist:   float = 14.0

    # Object lifecycle
    def add(self, shape:ShapeType, pos:np.ndarray, scale:float=1.0) -> Object3D:
        pal=PALETTES.get(shape.value,[(1,1,1)]); r,g,b=pal[0]
        o=Object3D(shape=shape,position=pos.copy(),_render=pos.copy(),
                   rotation=np.zeros(3,float),scale=scale,color=(r,g,b,1.0))
        self.objects.append(o); self._undo.append(o); self.sel_id=o.oid
        return o

    def undo(self):
        if not self._undo: return
        v=self._undo.pop()
        self.objects=[o for o in self.objects if o.oid!=v.oid]
        self.sel_id=self.objects[-1].oid if self.objects else None

    def delete_sel(self):
        if self.sel_id is None: return
        self.objects=[o for o in self.objects if o.oid!=self.sel_id]
        self._undo  =[o for o in self._undo   if o.oid!=self.sel_id]
        self.sel_id =self.objects[-1].oid if self.objects else None

    def reset(self):
        self.objects.clear(); self._undo.clear()
        self.sel_id=None; self.g_rot_y=0.0; self.g_scale=1.0

    @property
    def selected(self) -> Optional[Object3D]:
        return next((o for o in self.objects if o.oid==self.sel_id), None)

    # Manipulation
    def move_sel(self,dx,dy,dz=0.0):
        o=self.selected
        if o: o.position += np.array([dx,dy,dz])

    def scale_sel(self,f):
        o=self.selected
        if o: o.scale=float(np.clip(o.scale*f,0.08,20.0))

    def color_cycle_sel(self):
        o=self.selected
        if o: o.cycle_color()

    def apply_g_scale(self,f):
        self.g_scale=float(np.clip(self.g_scale*f,0.05,30.0))

    def apply_g_rot_y(self,d):
        self.g_rot_y += d

    def orbit(self,dyaw,dpitch):
        self.cam_yaw=(self.cam_yaw+dyaw)%360
        self.cam_pitch=float(np.clip(self.cam_pitch+dpitch,-85,-3))

    def zoom(self,d):
        self.cam_dist=float(np.clip(self.cam_dist-d,2.0,80.0))

    def select_nearest(self, nx,ny,depth=4.0) -> bool:
        if not self.objects: return False
        pos=self._unp(nx,ny,depth)
        dists=[float(np.linalg.norm(o.pos-pos)) for o in self.objects]
        idx=int(np.argmin(dists))
        if dists[idx]<3.0:
            self.sel_id=self.objects[idx].oid; return True
        return False

    def _unp(self,nx,ny,depth):
        return np.array([(nx-0.5)*depth*2.2, -(ny-0.5)*depth*1.6, -depth*0.4])

    def s2w(self,nx,ny,depth=3.5):
        return self._unp(nx,ny,depth)

    def tick(self,dt):
        for o in self.objects: o.tick()

# ======================================================================
#  RENDERER
# ======================================================================

class Renderer:
    """All OpenGL draw calls (fixed-function GL 1.x + GLU)."""

    def __init__(self,W,H):
        self.W=W; self.H=H
        self._q=gluNewQuadric(); gluQuadricNormals(self._q, GLU_SMOOTH)
        self._dl_cube    = self._mk_cube()
        self._dl_pyramid = self._mk_pyramid()
        self._dl_grid    = self._mk_grid()
        self._dl_axes    = self._mk_axes()
        self._dl_sky     = self._mk_sky()
        self._hud_tex    = glGenTextures(1)
        self._gl_init()

    def _gl_init(self):
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE); glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT,[0.18,0.20,0.26,1.0])
        glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,[0.5,0.5,0.5,1.0])
        glMaterialf (GL_FRONT_AND_BACK,GL_SHININESS,56.0)

    # Frame
    def begin_frame(self, sc: Scene3D):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glCallList(self._dl_sky)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45.0, self.W/max(self.H,1), 0.1, 400.0)

        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glLightfv(GL_LIGHT0,GL_DIFFUSE, [1.0,0.95,0.85,1.0])
        glLightfv(GL_LIGHT0,GL_SPECULAR,[0.7,0.7, 0.7, 1.0])
        glLightfv(GL_LIGHT0,GL_POSITION,[10.0,22.0,14.0,1.0])
        glLightfv(GL_LIGHT1,GL_DIFFUSE, [0.22,0.28,0.48,1.0])
        glLightfv(GL_LIGHT1,GL_SPECULAR,[0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT1,GL_POSITION,[-12.0,8.0,-8.0,1.0])

        ry=math.radians(sc.cam_yaw); rp=math.radians(sc.cam_pitch)
        cx=sc.cam_dist*math.cos(rp)*math.sin(ry)
        cy=sc.cam_dist*math.sin(-rp)
        cz=sc.cam_dist*math.cos(rp)*math.cos(ry)
        gluLookAt(cx,cy,cz,0,0,0,0,1,0)

        glPushMatrix()
        glScalef(sc.g_scale,sc.g_scale,sc.g_scale)
        glRotatef(sc.g_rot_y,0,1,0)

    def end_frame(self): glPopMatrix()

    def draw_grid(self): glCallList(self._dl_grid)
    def draw_axes(self): glCallList(self._dl_axes)

    def draw_objects(self, objects, sel_id):
        for obj in objects:
            glPushMatrix()
            p=obj.pos
            glTranslatef(float(p[0]),float(p[1]),float(p[2]))
            glRotatef(float(obj.rotation[0]),1,0,0)
            glRotatef(float(obj.rotation[1]),0,1,0)
            glRotatef(float(obj.rotation[2]),0,0,1)
            s=float(obj.scale); glScalef(s,s,s)
            self._shadow(obj.shape)
            if obj.oid==sel_id: self._sel_ring()
            glColor4f(*obj.color)
            self._prim(obj.shape)
            glPopMatrix()

    # ── Primitives ────────────────────────────────────────────────────

    def _prim(self, sh):
        if sh==ShapeType.SPHERE:
            gluSphere(self._q,0.5,24,16)
        elif sh==ShapeType.CUBE:
            glCallList(self._dl_cube)
        elif sh==ShapeType.CYLINDER:
            glTranslatef(0,-0.5,0); glRotatef(-90,1,0,0)
            gluCylinder(self._q,0.3,0.3,1.0,20,4)
            glTranslatef(0,0,1.0); gluDisk(self._q,0,0.3,20,1)
            glTranslatef(0,0,-1.0); gluDisk(self._q,0,0.3,20,1)
        elif sh==ShapeType.PYRAMID:
            glCallList(self._dl_pyramid)
        else:
            gluSphere(self._q,0.5,16,12)

    def _shadow(self, sh):
        """Flat dark blob on y=-2 plane."""
        glPushMatrix()
        glTranslatef(0,-2.0,0); glScalef(1.0,0.0,1.0)
        glColor4f(0,0,0,0.25)
        glDisable(GL_LIGHTING)
        gluSphere(self._q,0.52,12,8)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def _sel_ring(self):
        glDisable(GL_LIGHTING); glLineWidth(2.5)
        glColor4f(1.0,1.0,0.2,1.0)
        s=0.62
        verts=[(-s,-s,-s),(s,-s,-s),(s,s,-s),(-s,s,-s),
               (-s,-s, s),(s,-s, s),(s,s, s),(-s,s, s)]
        edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
               (0,4),(1,5),(2,6),(3,7)]
        glBegin(GL_LINES)
        for a,b in edges:
            glVertex3fv(verts[a]); glVertex3fv(verts[b])
        glEnd()
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    # ── HUD texture quad ──────────────────────────────────────────────

    def draw_hud_surface(self, surf: pygame.Surface):
        data=pygame.image.tostring(surf,"RGBA",True)
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,self.W,self.H,0,
                     GL_RGBA,GL_UNSIGNED_BYTE,data)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0,self.W,0,self.H,-1,1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glColor4f(1,1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0)
        glTexCoord2f(1,0); glVertex2f(self.W,0)
        glTexCoord2f(1,1); glVertex2f(self.W,self.H)
        glTexCoord2f(0,1); glVertex2f(0,self.H)
        glEnd()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    # ── Display-list builders ─────────────────────────────────────────

    def _mk_cube(self):
        dl=glGenLists(1); glNewList(dl,GL_COMPILE); s=0.5
        faces=[([0,0,1],[(-s,-s,s),(s,-s,s),(s,s,s),(-s,s,s)]),
               ([0,0,-1],[(-s,-s,-s),(-s,s,-s),(s,s,-s),(s,-s,-s)]),
               ([0,1,0],[(-s,s,-s),(-s,s,s),(s,s,s),(s,s,-s)]),
               ([0,-1,0],[(-s,-s,-s),(s,-s,-s),(s,-s,s),(-s,-s,s)]),
               ([1,0,0],[(s,-s,-s),(s,s,-s),(s,s,s),(s,-s,s)]),
               ([-1,0,0],[(-s,-s,-s),(-s,-s,s),(-s,s,s),(-s,s,-s)])]
        glBegin(GL_QUADS)
        for n,vs in faces:
            glNormal3fv(n)
            for v in vs: glVertex3fv(v)
        glEnd(); glEndList(); return dl

    def _mk_pyramid(self):
        dl=glGenLists(1); glNewList(dl,GL_COMPILE)
        apex=(0,0.7,0)
        base=[(-0.5,-0.35,-0.5),(0.5,-0.35,-0.5),
              (0.5,-0.35,0.5),(-0.5,-0.35,0.5)]
        glBegin(GL_QUADS); glNormal3f(0,-1,0)
        for v in reversed(base): glVertex3fv(v)
        glEnd()
        glBegin(GL_TRIANGLES)
        for i in range(4):
            a=base[i]; b=base[(i+1)%4]
            va=np.array(b)-np.array(a); vb=np.array(apex)-np.array(a)
            nv=np.cross(va,vb); nv/=np.linalg.norm(nv)+1e-8
            glNormal3fv(nv); glVertex3fv(a); glVertex3fv(b); glVertex3fv(apex)
        glEnd(); glEndList(); return dl

    def _mk_grid(self):
        dl=glGenLists(1); glNewList(dl,GL_COMPILE)
        glDisable(GL_LIGHTING); glLineWidth(1.0)
        glColor4f(0.22,0.22,0.32,1.0)
        half=GRID_HALF*GRID_STEP
        glBegin(GL_LINES)
        for i in range(-GRID_HALF,GRID_HALF+1):
            x=i*GRID_STEP
            glVertex3f(x,-2.0,-half); glVertex3f(x,-2.0,half)
            glVertex3f(-half,-2.0,x); glVertex3f(half,-2.0,x)
        glEnd(); glEnable(GL_LIGHTING); glEndList(); return dl

    def _mk_axes(self):
        dl=glGenLists(1); glNewList(dl,GL_COMPILE)
        glDisable(GL_LIGHTING); glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1,0.2,0.2); glVertex3f(0,0,0); glVertex3f(1.5,0,0)
        glColor3f(0.2,1,0.2); glVertex3f(0,0,0); glVertex3f(0,1.5,0)
        glColor3f(0.2,0.5,1); glVertex3f(0,0,0); glVertex3f(0,0,1.5)
        glEnd(); glLineWidth(1.0); glEnable(GL_LIGHTING); glEndList(); return dl

    def _mk_sky(self):
        dl=glGenLists(1); glNewList(dl,GL_COMPILE)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0,1,0,1,-1,1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glBegin(GL_QUADS)
        glColor3f(0.06,0.06,0.14); glVertex2f(0,0); glVertex2f(1,0)
        glColor3f(0.10,0.12,0.22); glVertex2f(1,1); glVertex2f(0,1)
        glEnd()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glEndList(); return dl

# ======================================================================
#  APPLICATION
# ======================================================================

class Application:
    """
    Main loop + state machine.

    States: ONBOARD → WAITING → COUNTDOWN → DRAW_MODE ↔ CREATE_MODE
    """

    def __init__(self):
        pygame.init(); pygame.font.init()
        self.W, self.H = WIN_W, WIN_H
        self.screen = pygame.display.set_mode(
            (self.W,self.H), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Gesture 3D — v3")
        glViewport(0,0,self.W,self.H)

        # Fonts
        self.f_huge  = pygame.font.SysFont("monospace",90,bold=True)
        self.f_big   = pygame.font.SysFont("monospace",52,bold=True)
        self.f_med   = pygame.font.SysFont("monospace",34,bold=True)
        self.f_small = pygame.font.SysFont("monospace",22)
        self.f_tiny  = pygame.font.SysFont("monospace",18)

        # Subsystems
        self.tracker   = HandTracker(2)
        self.gesture_r = GestureRecognizer()
        self.shape_r   = ShapeRecognizer()
        self.scene     = Scene3D()
        self.renderer  = Renderer(self.W, self.H)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("[ERROR] No webcam found!"); sys.exit(1)

        # State
        self.state           = AppState.ONBOARD
        self.countdown_t:    Optional[float] = None
        self.draw_trail:     List[Tuple[float,float]] = []
        self.is_drawing      = False
        self.hands:          List[HandData] = []
        self.fps             = 0.0
        self.frame_n         = 0

        # Mouse orbit
        self._mouse_drag     = False
        self._mouse_prev     = (0,0)

        # Onboard page
        self._onboard_page   = 0   # 0=welcome, 1=draw, 2=create, 3=shortcuts

        # Create-mode spawn debounce timer
        self._create_debounce = 0.0

        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(TARGET_FPS)/1000.0
            self.frame_n += 1
            if self.frame_n%15==0: self.fps=self.clock.get_fps()

            # Events
            for ev in pygame.event.get():
                if ev.type==QUIT: running=False
                elif ev.type==KEYDOWN:
                    if ev.key in (K_ESCAPE,K_q): running=False
                    elif ev.key==K_z: self.scene.undo()
                    elif ev.key==K_r: self.scene.reset()
                    elif ev.key in (K_DELETE,K_x): self.scene.delete_sel()
                    elif ev.key==K_UP:    self.scene.orbit(0,-3)
                    elif ev.key==K_DOWN:  self.scene.orbit(0,3)
                    elif ev.key==K_LEFT:  self.scene.orbit(-5,0)
                    elif ev.key==K_RIGHT: self.scene.orbit(5,0)
                    # Onboard navigation: SPACE or ENTER → next page
                    elif ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                        if self.state==AppState.ONBOARD:
                            self._onboard_page += 1
                            if self._onboard_page > 3:
                                self.state = AppState.WAITING
                elif ev.type==MOUSEBUTTONDOWN:
                    if ev.button==1:
                        self._mouse_drag=True; self._mouse_prev=ev.pos
                    elif ev.button==4: self.scene.zoom(1.0)
                    elif ev.button==5: self.scene.zoom(-1.0)
                elif ev.type==MOUSEWHEEL:
                    self.scene.zoom(ev.y)
                elif ev.type==MOUSEMOTION and self._mouse_drag:
                    dx=ev.pos[0]-self._mouse_prev[0]
                    dy=ev.pos[1]-self._mouse_prev[1]
                    self.scene.orbit(dx*0.4, dy*0.2)
                    self._mouse_prev=ev.pos
                elif ev.type==MOUSEBUTTONUP and ev.button==1:
                    self._mouse_drag=False

            # Camera frame
            ret,frame=self.cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                self.hands = self.tracker.process(frame)
            else:
                self.hands = []

            has_sel = self.scene.selected is not None
            events  = self.gesture_r.update(self.hands, has_sel)

            # Update state machine
            self._update_state(events, dt)

            # Scene tick (lerp)
            self.scene.tick(dt)

            # Render
            self.renderer.begin_frame(self.scene)
            if self.state not in (AppState.ONBOARD,AppState.WAITING,AppState.COUNTDOWN):
                self.renderer.draw_grid()
                self.renderer.draw_axes()
                self.renderer.draw_objects(self.scene.objects, self.scene.sel_id)
            self.renderer.end_frame()

            # HUD
            hud = self._build_hud(events)
            self.renderer.draw_hud_surface(hud)

            pygame.display.flip()

        self._cleanup()

    # ------------------------------------------------------------------
    def _update_state(self, ev: dict, dt: float):
        if self.state == AppState.ONBOARD:
            return   # navigation handled in event loop

        elif self.state == AppState.WAITING:
            if len(self.hands)>0:
                self.state = AppState.COUNTDOWN
                self.countdown_t = time.time()

        elif self.state == AppState.COUNTDOWN:
            if len(self.hands)==0:
                self.state = AppState.WAITING
                self.countdown_t = None
            elif time.time()-self.countdown_t >= COUNTDOWN_SEC:
                self.state = AppState.DRAW_MODE
                self.gesture_r.reset_timers()

        elif self.state == AppState.DRAW_MODE:
            self._draw_mode_logic(ev, dt)

        elif self.state == AppState.CREATE_MODE:
            self._create_mode_logic(ev, dt)

    # ------------------------------------------------------------------
    def _draw_mode_logic(self, ev:dict, dt:float):
        n = len(self.hands)

        # ── Drawing trail ─────────────────────────────────────────────
        if ev["is_drawing"]:
            self.is_drawing = True
            self.draw_trail.append(ev["draw_pt"])
            if len(self.draw_trail) > DRAW_TRAIL_MAX:
                self.draw_trail.pop(0)
        else:
            if self.is_drawing:
                # Finger lifted → try to recognise shape
                shape = self.shape_r.recognise(self.draw_trail)
                if shape != ShapeType.UNKNOWN:
                    # Place object at centroid of trail
                    cx = float(np.mean([p[0] for p in self.draw_trail]))
                    cy = float(np.mean([p[1] for p in self.draw_trail]))
                    pos = self.scene.s2w(cx, cy, 3.5)
                    self.scene.add_object = lambda sh,p,s=1.0: self.scene.add(sh,p,s)
                    self.scene.add(shape, pos)
            # Always clear trail when not drawing
            self.draw_trail = []
            self.is_drawing = False

        # ── Grab / move ───────────────────────────────────────────────
        if ev["grab_active"] and (ev["grab_dx"]!=0 or ev["grab_dy"]!=0):
            self.scene.move_sel(ev["grab_dx"], ev["grab_dy"])

        # ── Zoom ──────────────────────────────────────────────────────
        if ev["zoom_delta"] != 0:
            self.scene.zoom(ev["zoom_delta"])

        # ── Object scale (two pinch) ──────────────────────────────────
        if ev["scale_obj"] != 1.0:
            self.scene.scale_sel(ev["scale_obj"])

        # ── Scene scale / rotate ──────────────────────────────────────
        if ev["scene_scale"] != 1.0:
            self.scene.apply_g_scale(ev["scene_scale"])
        if ev["scene_rot_deg"] != 0.0:
            self.scene.apply_g_rot_y(ev["scene_rot_deg"])

        # ── Camera orbit (gesture) ────────────────────────────────────
        if ev["orbit_dyaw"]!=0 or ev["orbit_dpitch"]!=0:
            self.scene.orbit(ev["orbit_dyaw"], ev["orbit_dpitch"])

        # ── Colour cycle ──────────────────────────────────────────────
        if ev["do_color_cycle"]:
            self.scene.color_cycle_sel()

        # ── Delete ────────────────────────────────────────────────────
        if ev["do_delete"]:
            self.scene.delete_sel()

        # ── Auto-select nearest hand ──────────────────────────────────
        if n>0 and not ev["is_drawing"] and not ev["grab_active"]:
            self.scene.select_nearest(
                float(self.hands[0].palm_center[0]),
                float(self.hands[0].palm_center[1]))

        # ── Mode switch ───────────────────────────────────────────────
        if ev["open_hold"] >= MODE_HOLD_SEC:
            self.state = AppState.CREATE_MODE
            self.gesture_r.reset_timers()
            self.draw_trail = []; self.is_drawing = False

    # ------------------------------------------------------------------
    def _create_mode_logic(self, ev:dict, dt:float):
        self._create_debounce = max(0.0, self._create_debounce-dt)

        # Spawn on new create gesture
        if ev["create_new"] and self._create_debounce<=0:
            g = ev["create_gesture"]
            shape_map = {
                GestureType.TWO_FINGERS:   ShapeType.CUBE,
                GestureType.THREE_FINGERS: ShapeType.CYLINDER,
                GestureType.FOUR_FINGERS:  ShapeType.SPHERE,
            }
            sh = shape_map.get(g)
            if sh and len(self.hands)>0:
                nx = float(self.hands[0].palm_center[0])
                ny = float(self.hands[0].palm_center[1])
                pos = self.scene.s2w(nx, ny, 2.8)
                pos[1] += 0.6
                self.scene.add(sh, pos, 0.85)
                self._create_debounce = 1.0   # 1 second cooldown

        # Grab / move (same as draw mode)
        if ev["grab_active"] and (ev["grab_dx"]!=0 or ev["grab_dy"]!=0):
            self.scene.move_sel(ev["grab_dx"], ev["grab_dy"])

        if ev["scale_obj"]!=1.0:
            self.scene.scale_sel(ev["scale_obj"])
        if ev["scene_scale"]!=1.0:
            self.scene.apply_g_scale(ev["scene_scale"])
        if ev["scene_rot_deg"]!=0.0:
            self.scene.apply_g_rot_y(ev["scene_rot_deg"])
        if ev["orbit_dyaw"]!=0 or ev["orbit_dpitch"]!=0:
            self.scene.orbit(ev["orbit_dyaw"], ev["orbit_dpitch"])
        if ev["do_color_cycle"]:
            self.scene.color_cycle_sel()
        if ev["do_delete"]:
            self.scene.delete_sel()

        if len(self.hands)>0 and not ev["grab_active"]:
            self.scene.select_nearest(
                float(self.hands[0].palm_center[0]),
                float(self.hands[0].palm_center[1]))

        # Switch back to draw mode
        if ev["both_hold"] >= MODE_HOLD_SEC:
            self.state = AppState.DRAW_MODE
            self.gesture_r.reset_timers()

    # ------------------------------------------------------------------
    #  HUD builder
    # ------------------------------------------------------------------
    def _build_hud(self, ev:dict) -> pygame.Surface:
        surf = pygame.Surface((self.W,self.H), pygame.SRCALPHA)
        surf.fill((0,0,0,0))

        if self.state == AppState.ONBOARD:
            self._hud_onboard(surf)

        elif self.state == AppState.WAITING:
            self._txt(surf,"Point your webcam at your hand",
                      self.f_med,(100,200,255),self.W//2,self.H//2-30,cx=True)
            self._txt(surf,"Raise one hand to begin",
                      self.f_small,(160,160,200),self.W//2,self.H//2+20,cx=True)

        elif self.state == AppState.COUNTDOWN:
            elapsed = time.time()-self.countdown_t
            rem = max(0, COUNTDOWN_SEC-int(elapsed))
            label = str(rem) if rem>0 else "START!"
            col = (255,80,80) if rem>0 else (80,255,120)
            self._txt(surf,label,self.f_huge,col,self.W//2,self.H//2,cx=True)
            self._txt(surf,"Keep your hand visible",
                      self.f_small,(180,180,180),self.W//2,self.H//2+90,cx=True)

        elif self.state == AppState.DRAW_MODE:
            self._hud_draw(surf, ev)

        elif self.state == AppState.CREATE_MODE:
            self._hud_create(surf, ev)

        # Always: hands + fps + obj count
        self._txt(surf,f"Hands: {len(self.hands)}",
                  self.f_tiny,(120,255,120),self.W-200,20)
        self._txt(surf,f"Objects: {len(self.scene.objects)}",
                  self.f_tiny,(120,200,255),self.W-200,44)
        self._txt(surf,f"FPS: {self.fps:.0f}",
                  self.f_tiny,(220,220,100),self.W-200,68)

        return surf

    # ------------------------------------------------------------------
    def _hud_onboard(self, surf):
        pages = [self._ob_page0, self._ob_page1,
                 self._ob_page2, self._ob_page3]
        pages[min(self._onboard_page, len(pages)-1)](surf)
        # nav dots
        total = len(pages); cx = self.W//2
        for i in range(total):
            col = (255,255,255) if i==self._onboard_page else (80,80,100)
            pygame.draw.circle(surf, col, (cx+(i-total//2)*30, self.H-40), 8)
        self._txt(surf,"SPACE / ENTER → Next    ESC → Quit",
                  self.f_tiny,(120,120,150),self.W//2,self.H-18,cx=True)

    def _ob_page0(self, surf):
        # Welcome / title
        overlay = pygame.Surface((self.W,self.H),pygame.SRCALPHA)
        overlay.fill((0,0,0,200)); surf.blit(overlay,(0,0))
        self._txt(surf,"✦ GESTURE 3D ✦",self.f_big,(80,200,255),
                  self.W//2,120,cx=True)
        self._txt(surf,"Webcam-controlled 3D creative environment",
                  self.f_small,(180,180,200),self.W//2,200,cx=True)
        lines=[
            ("How it works",""),
            ("1.","Raise one hand → 5-second countdown → app starts"),
            ("2.","Use your index finger to DRAW shapes in the air"),
            ("3.","Shapes are auto-detected and turned into 3D objects"),
            ("4.","PINCH (thumb+index) to grab and MOVE an object"),
            ("5.","Open hand for 3s → switch to block-building mode"),
        ]
        y=300
        for lbl,txt in lines:
            if txt=="":
                self._txt(surf,lbl,self.f_med,(255,200,60),140,y)
            else:
                self._txt(surf,lbl,self.f_small,(100,200,255),140,y)
                self._txt(surf,txt,self.f_small,(200,200,200),200,y)
            y+=42

    def _ob_page1(self, surf):
        overlay=pygame.Surface((self.W,self.H),pygame.SRCALPHA)
        overlay.fill((0,0,0,210)); surf.blit(overlay,(0,0))
        self._txt(surf,"DRAW MODE — Gestures",self.f_big,
                  (80,200,255),self.W//2,80,cx=True)
        rows=[
            ("☝  Index finger only","→  Draw trail  (lift to finish + detect shape)"),
            ("✊  Closed fist drag", "→  Orbit camera"),
            ("🤏  Pinch (near object)","→  Grab & MOVE selected object"),
            ("🤏  Pinch (no object)", "→  Zoom camera"),
            ("👍  Thumb-up",          "→  Cycle colour of selected object"),
            ("✌  Peace + shake",     "→  DELETE selected object"),
            ("🖐  Open hand 3 s",     "→  Switch to CREATE MODE"),
            ("Both hands spread",    "→  Scale entire scene"),
            ("Both hands rotate",    "→  Rotate entire scene"),
            ("Both pinch",           "→  Scale selected object"),
        ]
        y=170
        for g,a in rows:
            self._txt(surf,g,self.f_small,(255,180,80),100,y)
            self._txt(surf,a,self.f_small,(200,220,200),480,y)
            y+=44

    def _ob_page2(self, surf):
        overlay=pygame.Surface((self.W,self.H),pygame.SRCALPHA)
        overlay.fill((0,0,0,210)); surf.blit(overlay,(0,0))
        self._txt(surf,"CREATE MODE — Block Building",
                  self.f_big,(255,180,60),self.W//2,80,cx=True)
        rows=[
            ("✌  2 fingers",       "→  Spawn a CUBE at hand position"),
            ("🤟  3 fingers",       "→  Spawn a CYLINDER"),
            ("🖖  4 fingers",       "→  Spawn a SPHERE"),
            ("🤏  Pinch",          "→  Grab & MOVE nearest object"),
            ("Both hands spread",  "→  Scale entire scene"),
            ("Both hands rotate",  "→  Rotate entire scene"),
            ("Both hands 3 s",     "→  Switch BACK to Draw Mode"),
        ]
        y=180
        for g,a in rows:
            self._txt(surf,g,self.f_small,(255,180,80),100,y)
            self._txt(surf,a,self.f_small,(200,220,200),480,y)
            y+=50
        self._txt(surf,"Tip: objects spawn 1 per gesture (1s cooldown)",
                  self.f_tiny,(140,140,160),self.W//2,620,cx=True)

    def _ob_page3(self, surf):
        overlay=pygame.Surface((self.W,self.H),pygame.SRCALPHA)
        overlay.fill((0,0,0,210)); surf.blit(overlay,(0,0))
        self._txt(surf,"KEYBOARD SHORTCUTS",self.f_big,
                  (180,255,180),self.W//2,80,cx=True)
        rows=[
            ("Arrow keys",   "Orbit camera"),
            ("Scroll wheel", "Zoom in / out"),
            ("Mouse drag",   "Orbit camera"),
            ("Z",            "Undo last placed object"),
            ("R",            "Reset entire scene"),
            ("Delete / X",   "Delete selected object"),
            ("ESC / Q",      "Quit"),
        ]
        y=200
        for k,a in rows:
            self._txt(surf,k,self.f_med,(255,220,80),160,y)
            self._txt(surf,a,self.f_small,(200,220,200),400,y)
            y+=52
        self._txt(surf,"Press SPACE to start →",
                  self.f_big,(80,255,160),self.W//2,self.H-100,cx=True)

    # ------------------------------------------------------------------
    def _hud_draw(self, surf, ev):
        # Mode label
        self._txt(surf,"DRAW MODE",self.f_med,(60,210,255),16,16)

        # Draw trail on screen
        if len(self.draw_trail)>1:
            pts=[(int(p[0]*self.W),int(p[1]*self.H)) for p in self.draw_trail]
            # Gradient: fade older points
            for i in range(1,len(pts)):
                alpha=int(60+190*(i/len(pts)))
                col=(255,140,60,alpha)
                # pygame.draw.line doesn't support alpha — draw on separate surface
                pygame.draw.line(surf,col[:3],pts[i-1],pts[i],3)
            # Bright dot at tip
            pygame.draw.circle(surf,(255,255,100),pts[-1],7)

        # Drawing status
        if self.is_drawing:
            self._txt(surf,"● DRAWING – lift finger to place",
                      self.f_small,(255,120,60),16,60)
        else:
            self._txt(surf,"☝ Point index finger to draw",
                      self.f_small,(160,200,160),16,60)

        # Grab status
        if ev["grab_active"]:
            self._txt(surf,"✊ GRABBING – move hand to reposition",
                      self.f_small,(255,220,60),16,90)

        # Open-hand mode-switch progress bar
        hold = ev["open_hold"]
        if hold>0.2:
            pct=min(1.0,hold/MODE_HOLD_SEC)
            bw=int(340*pct)
            pygame.draw.rect(surf,(40,40,60,160),(16,118,340,22))
            pygame.draw.rect(surf,(255,200,60,200),(16,118,bw,22))
            rem=max(0,MODE_HOLD_SEC-hold)
            self._txt(surf,f"Open hand → CREATE MODE in {rem:.1f}s",
                      self.f_tiny,(255,240,140),16,142)

        # Selected object info
        obj=self.scene.selected
        if obj:
            info=f"Selected: {obj.shape.value}  scale={obj.scale:.2f}"
            self._txt(surf,info,self.f_tiny,(200,200,255),16,self.H-36)

    # ------------------------------------------------------------------
    def _hud_create(self, surf, ev):
        self._txt(surf,"CREATE MODE",self.f_med,(255,180,60),16,16)
        self._txt(surf,"2=cube  3=cylinder  4=sphere  (pinch to move)",
                  self.f_small,(200,200,200),16,58)

        # Cooldown bar
        if self._create_debounce>0:
            pct=self._create_debounce/1.0
            bw=int(200*pct)
            pygame.draw.rect(surf,(40,40,60,160),(16,92,200,16))
            pygame.draw.rect(surf,(80,200,255,200),(16,92,bw,16))
            self._txt(surf,"Spawning cooldown",self.f_tiny,(100,200,255),16,112)

        # Both-hand switch-back bar
        hold=ev["both_hold"]
        if hold>0.2:
            pct=min(1.0,hold/MODE_HOLD_SEC)
            bw=int(340*pct)
            pygame.draw.rect(surf,(40,40,60,160),(16,134,340,22))
            pygame.draw.rect(surf,(60,200,255,200),(16,134,bw,22))
            rem=max(0,MODE_HOLD_SEC-hold)
            self._txt(surf,f"Both hands → DRAW MODE in {rem:.1f}s",
                      self.f_tiny,(120,220,255),16,158)

        obj=self.scene.selected
        if obj:
            self._txt(surf,f"Selected: {obj.shape.value}  scale={obj.scale:.2f}",
                      self.f_tiny,(200,200,255),16,self.H-36)

    # ------------------------------------------------------------------
    @staticmethod
    def _txt(surf, text, font, color, x, y, cx=False):
        r=font.render(text,True,color[:3] if len(color)>3 else color)
        if cx: x-=r.get_width()//2
        surf.blit(r,(x,y))

    # ------------------------------------------------------------------
    def _cleanup(self):
        self.tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("[INFO] Closed.")


# ======================================================================
#  ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    Application().run()
