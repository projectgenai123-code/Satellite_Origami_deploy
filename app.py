"""
app.py — Satellite Origami Backend (Production Ready for Render)
CORS fully fixed — works for all users worldwide
STANDING satellite — origami folds clearly visible
"""

import os, io, base64, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from rag_pipeline import RAGPipeline

app = Flask(__name__)

# ── CORS — allow ALL origins ──────────────────────────────────────────────────
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type"],
     methods=["GET","POST","OPTIONS"])

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE, "cvae_model.pth")
KNOWLEDGE_PATH = os.path.join(BASE, "knowledge_base.json")
FRONTEND_PATH  = os.path.join(BASE, "chatbot.html")

LABEL_NAMES  = {0:"solar_panel", 1:"antenna", 2:"reflector", 3:"truss"}
NUM_CLASSES  = 4
INPUT_DIM    = 140
LATENT_DIM   = 16

# ── Body dimensions ───────────────────────────────────────────────────────────
BODY_X = 0.55
BODY_Y = 1.6
BODY_Z = 0.55

# ══════════════════════════════════════════════════════════════════════════════
# CVAE
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM+NUM_CLASSES,256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256,128),                   nn.LayerNorm(128), nn.LeakyReLU(0.2),
        )
        self.fc_mean    = nn.Linear(128, LATENT_DIM)
        self.fc_log_var = nn.Linear(128, LATENT_DIM)
    def forward(self,x,c):
        h = self.shared(torch.cat([x,c],dim=1))
        return self.fc_mean(h), self.fc_log_var(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM+NUM_CLASSES,128), nn.LayerNorm(128), nn.LeakyReLU(0.2),
            nn.Linear(128,256),                    nn.LayerNorm(256), nn.LeakyReLU(0.2),
            nn.Linear(256,INPUT_DIM), nn.Sigmoid()
        )
    def forward(self,z,c): return self.net(torch.cat([z,c],dim=1))

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def reparameterize(self,mean,log_var):
        return mean + torch.exp(0.5*log_var) * torch.randn_like(log_var)
    def forward(self,x,c):
        mean,log_var = self.encoder(x,c)
        return self.decoder(self.reparameterize(mean,log_var),c), mean, log_var

print("Loading CVAE...")
ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
cvae  = CVAE()
cvae.load_state_dict(ckpt["model_state"])
cvae.eval()
X_min   = ckpt["X_min"]
X_range = ckpt["X_range"]
print("CVAE loaded!")

print("Loading RAG...")
rag = RAGPipeline(KNOWLEDGE_PATH)
print("RAG loaded!")

def get_seed(part_name):
    label_idx = {v:k for k,v in LABEL_NAMES.items()}[part_name]
    with torch.no_grad():
        z = torch.randn(1,LATENT_DIM)
        c = torch.zeros(1,NUM_CLASSES); c[0,label_idx]=1.0
        out = cvae.decoder(z,c).numpy()[0]
    return out * X_range + X_min

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY — TRUE ORIGAMI AESTHETIC
# All shapes use flat angular facets + explicit crease/fold lines
# like real paper folding: Miura-ori panels, Yoshimura body, waterbomb antenna
# ══════════════════════════════════════════════════════════════════════════════

def build_body():
    """Yoshimura-pattern cylindrical body — diamond facet origami tube"""
    bx, by, bz = BODY_X, BODY_Y, BODY_Z
    n_rings  = 8   # horizontal rings of diamonds
    n_sides  = 6   # hexagonal cross-section
    verts, faces, mfolds, vfolds = [], [], [], []

    # Build diamond-faceted tube (Yoshimura pattern)
    for ring in range(n_rings + 1):
        y = -by + ring * (2*by / n_rings)
        offset = (np.pi / n_sides) * (ring % 2)
        for s in range(n_sides):
            angle = 2*np.pi*s/n_sides + offset
            verts.append([bx*np.cos(angle), y, bz*np.sin(angle)])

    verts = np.array(verts)

    # Quad faces between rings
    for ring in range(n_rings):
        for s in range(n_sides):
            a = ring*n_sides + s
            b = ring*n_sides + (s+1)%n_sides
            c = (ring+1)*n_sides + (s+1)%n_sides
            d = (ring+1)*n_sides + s
            # Split quad into 2 triangles to get origami diamond facets
            faces.append([a, b, c])
            faces.append([a, c, d])
            # Mountain folds = ring edges
            mfolds.append([verts[a].tolist(), verts[b].tolist()])
            # Valley folds = diagonal creases
            vfolds.append([verts[b].tolist(), verts[d].tolist()])

    # Top and bottom caps (flat polygons)
    top_y_verts = [ring*n_sides + s for s in range(n_sides) if True]  # last ring
    # Add centre cap verts
    top_idx = len(verts); verts = np.vstack([verts, [[0, by, 0]]])
    bot_idx = len(verts); verts = np.vstack([verts, [[0, -by, 0]]])

    for s in range(n_sides):
        t0 = n_rings*n_sides + s
        t1 = n_rings*n_sides + (s+1)%n_sides
        faces.append([top_idx, t0, t1])
        mfolds.append([verts[t0].tolist(), verts[t1].tolist()])

        b0 = s
        b1 = (s+1)%n_sides
        faces.append([bot_idx, b1, b0])
        mfolds.append([verts[b0].tolist(), verts[b1].tolist()])

    return verts, faces, mfolds, vfolds


def build_solar_panel(side, panel_index=0, n_panels_this_side=1):
    """Miura-ori solar panel — zigzag accordion fold, clearly angular"""
    nx = 5    # columns of Miura cells
    ny = 4    # rows of Miura cells
    cell_w = 0.55   # width of each cell
    cell_h = (2*BODY_Y*0.85) / max(n_panels_this_side, 1) / ny
    shear  = 0.18   # Miura parallelogram shear angle offset
    fold_z = 0.08   # z-depth of accordion fold

    x_base = BODY_X * side
    y_base = -BODY_Y*0.85 + panel_index * (ny * cell_h)

    verts = []
    # Build parallelogram grid — alternating rows offset for Miura look
    for row in range(ny + 1):
        y = y_base + row * cell_h
        x_shift = shear * (row % 2)
        for col in range(nx + 1):
            x = x_base + side * (col * cell_w + x_shift)
            # Accordion z: fold up/down alternating columns
            z = fold_z * (1 if col % 2 == 0 else -1) * (1 if row % 2 == 0 else -1)
            verts.append([x, y, z])

    verts = np.array(verts)
    faces, mfolds, vfolds = [], [], []

    for row in range(ny):
        for col in range(nx):
            a = row*(nx+1) + col
            b = row*(nx+1) + col+1
            c = (row+1)*(nx+1) + col+1
            d = (row+1)*(nx+1) + col
            # Each Miura cell = 2 triangles (angular facets)
            faces.append([a, b, c])
            faces.append([a, c, d])
            # Mountain folds — horizontal crease lines
            if row % 2 == 0:
                mfolds.append([verts[a].tolist(), verts[b].tolist()])
            else:
                vfolds.append([verts[a].tolist(), verts[b].tolist()])
            # Valley folds — vertical crease lines
            if col % 2 == 0:
                vfolds.append([verts[a].tolist(), verts[d].tolist()])
            else:
                mfolds.append([verts[a].tolist(), verts[d].tolist()])
            # Diagonal Miura crease
            mfolds.append([verts[b].tolist(), verts[d].tolist()])

    mid_y = y_base + (ny * cell_h) / 2
    attach = [x_base, mid_y, 0]
    return verts, faces, mfolds, vfolds, attach, attach


def build_antenna(index=0, n_total=1):
    """Waterbomb-base origami antenna — angular faceted dish from triangular folds"""
    n_petals = 8   # number of triangular waterbomb petals
    r_inner  = 0.15
    r_outer  = 0.75
    depth    = 0.35  # how deep the dish folds
    x_off    = (index - (n_total-1)/2) * 1.6
    y_base   = BODY_Y + 0.15
    mast_top = BODY_Y + 0.55

    verts, faces, mfolds, vfolds = [], [], [], []

    # Centre point of dish
    c_idx = 0
    verts.append([x_off, mast_top + depth, 0])  # dish centre (deepest point)

    # Inner ring — waterbomb inner fold ring
    for i in range(n_petals):
        angle = 2*np.pi*i/n_petals
        verts.append([x_off + r_inner*np.cos(angle),
                      mast_top + depth*0.4,
                      r_inner*np.sin(angle)])

    # Outer ring — dish rim
    for i in range(n_petals):
        angle = 2*np.pi*i/n_petals + np.pi/n_petals  # offset = waterbomb twist
        verts.append([x_off + r_outer*np.cos(angle),
                      mast_top,
                      r_outer*np.sin(angle)])

    verts = np.array(verts)

    for i in range(n_petals):
        inner_a = 1 + i
        inner_b = 1 + (i+1)%n_petals
        outer_a = 1 + n_petals + i
        outer_b = 1 + n_petals + (i+1)%n_petals

        # Inner petal triangle (mountain fold)
        faces.append([0, inner_a, inner_b])
        mfolds.append([verts[0].tolist(), verts[inner_a].tolist()])
        mfolds.append([verts[0].tolist(), verts[inner_b].tolist()])

        # Outer petal quad (valley fold)
        faces.append([inner_a, outer_a, outer_b])
        faces.append([inner_a, outer_b, inner_b])
        vfolds.append([verts[inner_a].tolist(), verts[outer_a].tolist()])
        vfolds.append([verts[outer_a].tolist(), verts[outer_b].tolist()])

    mast_base = [x_off, BODY_Y, 0]
    mast_tip  = [x_off, mast_top, 0]
    return verts, faces, mfolds, vfolds, mast_base, mast_tip


def build_reflector(index=0, n_total=1):
    """Kite-base origami reflector — angular concave bowl from fold triangles"""
    n_petals = 6
    r_outer  = 0.8
    depth    = 0.5
    x_off    = (index - (n_total-1)/2) * 1.8
    y_base   = -BODY_Y - 0.15
    mast_tip = -BODY_Y - 0.45

    verts, faces, mfolds, vfolds = [], [], [], []

    # Tip (deepest point of reflector bowl)
    verts.append([x_off, mast_tip - depth, 0])  # idx 0

    # Mid ring
    for i in range(n_petals):
        angle = 2*np.pi*i/n_petals
        verts.append([x_off + (r_outer*0.4)*np.cos(angle),
                      mast_tip - depth*0.3,
                      (r_outer*0.4)*np.sin(angle)])

    # Rim ring (offset for kite-base twist)
    for i in range(n_petals):
        angle = 2*np.pi*i/n_petals + np.pi/n_petals
        verts.append([x_off + r_outer*np.cos(angle),
                      y_base,
                      r_outer*np.sin(angle)])

    verts = np.array(verts)

    for i in range(n_petals):
        mid_a = 1 + i
        mid_b = 1 + (i+1)%n_petals
        rim_a = 1 + n_petals + i
        rim_b = 1 + n_petals + (i+1)%n_petals

        faces.append([0, mid_a, mid_b])
        mfolds.append([verts[0].tolist(), verts[mid_a].tolist()])
        mfolds.append([verts[mid_a].tolist(), verts[mid_b].tolist()])

        faces.append([mid_a, rim_a, rim_b])
        faces.append([mid_a, rim_b, mid_b])
        vfolds.append([verts[mid_a].tolist(), verts[rim_a].tolist()])
        vfolds.append([verts[rim_a].tolist(), verts[rim_b].tolist()])

    mast_base = [x_off, -BODY_Y, 0]
    mast_end  = [x_off, mast_tip, 0]
    return verts, faces, mfolds, vfolds, mast_base, mast_end


def build_truss(n_segments=1):
    """Yoshimura cylindrical truss — origami tube with diamond crease pattern"""
    length   = BODY_Y * 2 * 1.08
    n_rings  = 6 * n_segments
    n_sides  = 4   # square cross-section for clear diamond pattern
    r        = 0.09

    verts, faces, mfolds, vfolds, rings = [], [], [], [], []

    for ring in range(n_rings + 1):
        y = -length/2 + ring*(length/n_rings)
        rot = (np.pi/n_sides) * (ring % 2)  # alternate rotation = diamond pattern
        row = []
        for s in range(n_sides):
            angle = 2*np.pi*s/n_sides + rot
            verts.append([r*np.cos(angle), y, r*np.sin(angle)])
            row.append(len(verts)-1)
        rings.append(row)

    verts = np.array(verts)

    for ring in range(n_rings):
        r0, r1 = rings[ring], rings[ring+1]
        for s in range(n_sides):
            a = r0[s]; b = r0[(s+1)%n_sides]
            c = r1[(s+1)%n_sides]; d = r1[s]
            faces.append([a, b, c])
            faces.append([a, c, d])
            mfolds.append([verts[a].tolist(), verts[d].tolist()])   # vertical ridge
            vfolds.append([verts[b].tolist(), verts[d].tolist()])   # diagonal crease

    return verts, faces, mfolds, vfolds

# ══════════════════════════════════════════════════════════════════════════════
# PNG RENDER — STANDING satellite, front view shows origami folds clearly
# ══════════════════════════════════════════════════════════════════════════════
def _draw_poly(ax, v, f, fc, alpha):
    if not f: return
    ax.add_collection3d(Poly3DCollection([[v[vi] for vi in face] for face in f],
        alpha=alpha, facecolor=fc, edgecolor="#00b4ff", linewidth=0.2))

def _draw_folds(ax, folds, color, ls="-", lw=0.9):
    for f in folds:
        ax.plot([f[0][0],f[1][0]], [f[0][1],f[1][1]], [f[0][2],f[1][2]],
                color=color, lw=lw, ls=ls, alpha=0.95)

def _draw_rod(ax, p0, p1, color="#ffffff", lw=3):
    ax.plot([p0[0],p1[0]], [p0[1],p1[1]], [p0[2],p1[2]],
            color=color, lw=lw, alpha=1.0, solid_capstyle="round", zorder=10)

def render_satellite_png(config):
    ns=int(config.get("solar_panel",2)); na=int(config.get("antenna",1))
    nr=int(config.get("reflector",1));   nt=int(config.get("truss",1))
    BG="#010812"
    fig=plt.figure(figsize=(16,9),facecolor=BG)

    # Left: perspective showing 3D origami structure, Right: side view standing upright
    views = [
        (25, -60, "PERSPECTIVE VIEW"),
        (5,  -90, "SIDE VIEW — ORIGAMI FOLDS"),
    ]

    for idx,(elev,azim,title) in enumerate(views):
        ax=fig.add_subplot(1,2,idx+1,projection="3d")
        ax.set_facecolor(BG); ax.view_init(elev=elev,azim=azim)
        for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            pane.fill=False; pane.set_edgecolor(BG)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        ax.set_title(title,fontsize=10,fontweight="bold",color="#00dcff",fontfamily="monospace",pad=8)

        # Draw truss (vertical spine inside body)
        tv,tf,tmf,tvf=build_truss(nt)
        _draw_poly(ax,tv,tf,"#b87400",0.5)
        _draw_folds(ax,tmf,"#ffd700",lw=0.7)
        _draw_folds(ax,tvf,"#ffa500","--",lw=0.5)

        # Draw body
        bv,bf,bfolds,bvf=build_body()
        _draw_poly(ax,bv,bf,"#1a3a60",0.82)
        _draw_folds(ax,bfolds,"#00dcff",lw=1.4)
        _draw_folds(ax,bvf,"#00ffcc","--",lw=0.8)

        # Draw solar panels (wings left/right)
        lc=(ns+1)//2; rc=ns//2
        for i in range(lc):
            sv,sf,smf,svf,_,_=build_solar_panel(-1,i,lc)
            _draw_poly(ax,sv,sf,"#0d2d6e",0.85)
            _draw_folds(ax,smf,"#00dcff","-",1.6)
            _draw_folds(ax,svf,"#00ffcc","--",1.0)
        for i in range(rc):
            sv,sf,smf,svf,_,_=build_solar_panel(+1,i,rc)
            _draw_poly(ax,sv,sf,"#0d2d6e",0.85)
            _draw_folds(ax,smf,"#00dcff","-",1.6)
            _draw_folds(ax,svf,"#00ffcc","--",1.0)

        # Draw antennas (top of body)
        for i in range(na):
            av,af,amf,avf,mb,mt=build_antenna(i,na)
            _draw_poly(ax,av,af,"#006644",0.75)
            _draw_folds(ax,amf,"#00ffcc",lw=1.2)
            _draw_folds(ax,avf,"#aaffee","--")
            _draw_rod(ax,mb,mt,"#00ffcc",lw=2)

        # Draw reflectors (bottom of body)
        for i in range(nr):
            rv,rf,rmf,rvf,cb,cr=build_reflector(i,nr)
            _draw_poly(ax,rv,rf,"#5a1a8a",0.75)
            _draw_folds(ax,rmf,"#dd88ff",lw=1.2)
            _draw_folds(ax,rvf,"#cc66ff","--")
            _draw_rod(ax,cb,cr,"#bb66ff",lw=2)

        # Axis limits — standing satellite, tight fit
        span_x = BODY_X + 0.55*5 + 0.5
        ax.set_xlim(-span_x, span_x)
        ax.set_ylim(-BODY_Y - 1.2, BODY_Y + 1.5)
        ax.set_zlim(-1.0, 1.0)

    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D
    fig.legend(handles=[
        Patch(facecolor="#0d2d6e",edgecolor="#00b4ff",label=f"Solar Panels ({ns})"),
        Patch(facecolor="#006644",edgecolor="#00b4ff",label=f"Antenna ({na})"),
        Patch(facecolor="#5a1a8a",edgecolor="#00b4ff",label=f"Reflector ({nr})"),
        Patch(facecolor="#b87400",edgecolor="#00b4ff",label=f"Truss ({nt})"),
        Patch(facecolor="#1a3a60",edgecolor="#00b4ff",label="Body"),
        Line2D([0],[0],color="#00dcff",lw=1.5,label="Mountain Fold"),
        Line2D([0],[0],color="#00ffcc",lw=1.0,ls="--",label="Valley Fold"),
    ],loc="lower center",ncol=7,fontsize=9,frameon=True,bbox_to_anchor=(0.5,0.01),
      facecolor="#050f20",edgecolor="#00b4ff",labelcolor="#c8eeff")
    plt.suptitle(f"ORIGAMI SATELLITE  |  Solar:{ns}  Antenna:{na}  Reflector:{nr}  Truss:{nt}",
                 fontsize=12,fontweight="bold",color="#00dcff",fontfamily="monospace")
    plt.tight_layout()
    buf=io.BytesIO()
    plt.savefig(buf,format="png",dpi=130,bbox_inches="tight",facecolor=BG)
    plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# 3D FOR THREE.JS
# ══════════════════════════════════════════════════════════════════════════════
def f2tri(verts, faces):
    idx = []
    for f in faces:
        if len(f) == 3:
            idx += list(f)
        elif len(f) == 4:
            a,b,c,d = f; idx += [a,b,c, a,c,d]
    return idx

def to3js(verts,faces,mf,vf,color,opacity,lc):
    vl=verts.tolist() if hasattr(verts,"tolist") else verts
    lines=[]
    for p in mf+vf: lines+=[list(p[0]),list(p[1])]
    return {"vertices":vl,"indices":f2tri(vl,faces),"lines":lines,
            "color":color,"opacity":opacity,"line_color":lc}

def mast3d(p0,p1,color="#ffffff"):
    r=0.04; lines=[]
    for ox,oz in [(r,0),(-r,0),(0,r),(0,-r)]:
        lines+=[[p0[0]+ox,p0[1],p0[2]+oz],[p1[0]+ox,p1[1],p1[2]+oz]]
    lines+=[list(p0),list(p1)]
    return {"vertices":[list(p0),list(p1)],"indices":[],"lines":lines,
            "color":color,"opacity":1.0,"line_color":color}

def build_satellite_3d(config):
    ns=int(config.get("solar_panel",2)); na=int(config.get("antenna",1))
    nr=int(config.get("reflector",1));   nt=int(config.get("truss",1))
    meshes=[]
    tv,tf,tmf,tvf=build_truss(nt); meshes.append(to3js(tv,tf,tmf,tvf,"#b87400",0.5,"#ffd700"))
    bv,bf,bfolds,bvf=build_body(); meshes.append(to3js(bv,bf,bfolds+bvf,[],"#1a3a60",0.85,"#00dcff"))
    lc=(ns+1)//2; rc=ns//2
    for i in range(lc):
        sv,sf,smf,svf,_,_=build_solar_panel(-1,i,lc); meshes.append(to3js(sv,sf,smf,svf,"#0d2d6e",0.85,"#00dcff"))
    for i in range(rc):
        sv,sf,smf,svf,_,_=build_solar_panel(+1,i,rc); meshes.append(to3js(sv,sf,smf,svf,"#0d2d6e",0.85,"#00dcff"))
    for i in range(na):
        av,af,amf,avf,mb,mt=build_antenna(i,na)
        meshes.append(to3js(av,af,amf,avf,"#006644",0.75,"#00ffcc")); meshes.append(mast3d(mb,mt,"#00ffcc"))
    for i in range(nr):
        rv,rf,rmf,rvf,cb,cr=build_reflector(i,nr)
        meshes.append(to3js(rv,rf,rmf,rvf,"#5a1a8a",0.75,"#dd88ff")); meshes.append(mast3d(cb,cr,"#bb66ff"))
    return meshes

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET","OPTIONS"])
def health():
    return jsonify({"status":"ok","message":"Satellite Origami Backend Running"})

@app.route("/api/generate", methods=["POST","OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return "", 204
    data=request.json
    cfg={"solar_panel":data.get("solar_panel",2),"antenna":data.get("antenna",1),
         "reflector":data.get("reflector",1),"truss":data.get("truss",1)}
    return jsonify({"image":render_satellite_png(cfg),"config":cfg,"status":"generated"})

@app.route("/api/generate3d", methods=["POST","OPTIONS"])
def generate3d():
    if request.method == "OPTIONS":
        return "", 204
    data=request.json
    cfg={"solar_panel":data.get("solar_panel",2),"antenna":data.get("antenna",1),
         "reflector":data.get("reflector",1),"truss":data.get("truss",1)}
    return jsonify({"meshes":build_satellite_3d(cfg),"config":cfg,"status":"ok"})

@app.route("/api/chat", methods=["POST","OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204
    data=request.json
    message=data.get("message","").strip()
    answer,sources=rag.answer(message)
    return jsonify({"reply":answer,
                    "sources":[{"topic":s.get("topic",""),"score":s.get("score",0)} for s in sources]})

@app.route("/")
def index():
    return send_file(FRONTEND_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nSatellite Origami Backend — Production (Render)")
    print("CORS: ALL origins allowed")
    print("Satellite: STANDING orientation — origami folds visible from front")
    print("Routes: /  |  /api/health  |  /api/generate  |  /api/generate3d  |  /api/chat")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
