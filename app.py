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

# ── STANDING satellite body (tall, like a real comms satellite) ───────────────
BODY_X = 0.6   # narrow width
BODY_Y = 1.8   # TALL height  ← key change
BODY_Z = 0.6   # narrow depth

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
# GEOMETRY — STANDING orientation
# ══════════════════════════════════════════════════════════════════════════════

def build_body():
    bx,by,bz = BODY_X,BODY_Y,BODY_Z
    v = np.array([
        [-bx,-by,-bz],[ bx,-by,-bz],[ bx, by,-bz],[-bx, by,-bz],
        [-bx,-by, bz],[ bx,-by, bz],[ bx, by, bz],[-bx, by, bz],
    ])
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    # Origami fold lines on the body — horizontal and vertical creases
    folds = []
    # Vertical mountain folds on front face
    for xi in np.linspace(-bx, bx, 5)[1:-1]:
        folds.append([[xi, -by, bz], [xi, by, bz]])
    # Horizontal valley folds on front face
    for yi in np.linspace(-by, by, 7)[1:-1]:
        folds.append([[-bx, yi, bz], [bx, yi, bz]])
    # Diagonal folds
    folds.append([[-bx, -by, bz], [bx, by, bz]])
    folds.append([[bx, -by, bz], [-bx, by, bz]])
    return v, faces, folds

def build_solar_panel(side, panel_index=0, n_panels_this_side=1):
    # Solar panels extend LEFT/RIGHT horizontally, with Miura-ori fold pattern
    pw  = 2.8   # panel width (X direction)
    ph  = 0.04  # panel thickness
    ny  = 6     # fold divisions along Y (height)
    nx  = 5     # fold divisions along X (width)

    # Stack panels vertically if multiple
    panel_height = (2 * BODY_Y * 0.9) / max(n_panels_this_side, 1)
    y_start = -BODY_Y * 0.9 + panel_index * panel_height
    y_end   = y_start + panel_height * 0.92

    x_start = BODY_X       if side == 1 else -(BODY_X + pw)
    x_end   = BODY_X + pw  if side == 1 else -BODY_X

    verts = []
    for xi in np.linspace(x_start, x_end, nx+1):
        for yi in np.linspace(y_start, y_end, ny+1):
            # Miura-ori zigzag in Z
            z_offset = ph * (0.5 if (int((xi - x_start) / (pw/nx)) + int((yi - y_start) / (panel_height/ny))) % 2 == 0 else -0.5)
            verts.append([xi, yi, z_offset])
    verts = np.array(verts)

    faces = []
    for i in range(nx):
        for j in range(ny):
            a = i*(ny+1)+j; b = a+1
            c = (i+1)*(ny+1)+j+1; d = (i+1)*(ny+1)+j
            faces.append([a,b,c,d])

    # Mountain folds — horizontal lines
    mfolds = [[verts[j].tolist(), verts[nx*(ny+1)+j].tolist()] for j in range(1, ny+1, 2)]
    # Valley folds — vertical lines
    vfolds = [[verts[i*(ny+1)].tolist(), verts[i*(ny+1)+ny].tolist()] for i in range(1, nx+1, 2)]
    # Diagonal Miura folds
    for i in range(0, nx, 2):
        for j in range(0, ny, 2):
            a = i*(ny+1)+j; c = (i+1)*(ny+1)+j+1
            mfolds.append([verts[a].tolist(), verts[c].tolist()])

    mid_y = (y_start + y_end) / 2
    return verts, faces, mfolds, vfolds, [BODY_X*side, mid_y, 0], [BODY_X*side, mid_y, 0]

def build_antenna(index=0, n_total=1):
    # Antenna on TOP of body
    r_max=0.8; n_rings=5; n_segs=12; height=0.5
    x_off  = (index-(n_total-1)/2)*1.4
    dish_y = BODY_Y + height
    verts  = []
    for i in range(n_rings+1):
        r = r_max*i/n_rings
        y = dish_y + 0.25*(r/r_max)**2
        for j in range(n_segs):
            a = 2*np.pi*j/n_segs
            verts.append([x_off+r*np.cos(a), y, r*np.sin(a)])
    verts = np.array(verts)
    faces = []
    for i in range(n_rings):
        for j in range(n_segs):
            faces.append([i*n_segs+j, i*n_segs+(j+1)%n_segs,
                          (i+1)*n_segs+(j+1)%n_segs, (i+1)*n_segs+j])
    center = [x_off, dish_y, 0]
    mfolds = [[center, verts[n_rings*n_segs+j].tolist()] for j in range(n_segs)]
    vfolds = []
    for i in range(1, n_rings+1, 2):
        ring = [verts[i*n_segs+j].tolist() for j in range(n_segs)] + [verts[i*n_segs].tolist()]
        vfolds += [[ring[k], ring[k+1]] for k in range(len(ring)-1)]
    return verts, faces, mfolds, vfolds, [x_off, BODY_Y, 0], [x_off, dish_y, 0]

def build_reflector(index=0, n_total=1):
    # Reflector at BOTTOM of body
    size=0.9; depth=0.4; n=8
    x_off = (index-(n_total-1)/2)*1.5
    verts = []
    for xi in np.linspace(-size, size, n):
        for zi in np.linspace(-size, size, n):
            y = -BODY_Y - 0.3 - depth*(xi**2+zi**2)/(size**2)
            verts.append([x_off+xi, y, zi])
    verts = np.array(verts)
    faces = []
    for i in range(n-1):
        for j in range(n-1):
            faces.append([i*n+j, i*n+j+1, (i+1)*n+j+1, (i+1)*n+j])
    mfolds = []
    for i in range(n):
        row = [verts[i*n+j].tolist() for j in range(n)]
        mfolds += [[row[k], row[k+1]] for k in range(len(row)-1)]
    vfolds = [[verts[i*n+j].tolist(), verts[(i+1)*n+j+1].tolist()]
              for i in range(n-1) for j in range(n-1) if (i+j)%2==0]
    return verts, faces, mfolds, vfolds, [x_off, -BODY_Y, 0], [x_off, -BODY_Y-0.3, 0]

def build_truss(n_segments=1):
    # Truss runs vertically (Y axis) along body spine
    length = BODY_Y * 2 * 1.05
    radius = 0.10; n_rings = 8*n_segments; n_sides = 6
    verts=[]; faces=[]; mfolds=[]; vfolds=[]; rings=[]
    for i in range(n_rings+1):
        y = (-length/2) + i*(length/n_rings)
        rot = (np.pi/n_sides)*(i%2); ring=[]
        for j in range(n_sides):
            a = 2*np.pi*j/n_sides + rot
            r = radius + 0.012*(j%2)
            verts.append([r*np.cos(a), y, r*np.sin(a)])
            ring.append(len(verts)-1)
        rings.append(ring)
    verts = np.array(verts)
    for i in range(n_rings):
        r0,r1 = rings[i],rings[i+1]
        for j in range(n_sides):
            faces.append([r0[j], r0[(j+1)%n_sides], r1[(j+1)%n_sides], r1[j]])
            mfolds.append([verts[r0[j]].tolist(), verts[r1[j]].tolist()])
            vfolds.append([verts[r0[j]].tolist(), verts[r1[(j+1)%n_sides]].tolist()])
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

    # Left: Front view (shows folds clearly), Right: 3/4 perspective view
    views = [
        (20, -90, "FRONT VIEW — ORIGAMI FOLDS"),
        (30, -45, "PERSPECTIVE VIEW"),
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
        _draw_poly(ax,tv,tf,"#cc8800",0.4)
        _draw_folds(ax,tmf,"#ffcc44",lw=0.6)
        _draw_folds(ax,tvf,"#ffaa00","--",lw=0.5)

        # Draw body
        bv,bf,bfolds=build_body()
        _draw_poly(ax,bv,bf,"#1a3055",0.75)
        _draw_folds(ax,bfolds,"#00dcff",lw=1.2)

        # Draw solar panels (wings left/right)
        lc=(ns+1)//2; rc=ns//2
        for i in range(lc):
            sv,sf,smf,svf,_,_=build_solar_panel(-1,i,lc)
            _draw_poly(ax,sv,sf,"#1144aa",0.75)
            _draw_folds(ax,smf,"#00dcff","-",1.4)
            _draw_folds(ax,svf,"#00ffcc","--",0.9)
        for i in range(rc):
            sv,sf,smf,svf,_,_=build_solar_panel(+1,i,rc)
            _draw_poly(ax,sv,sf,"#1144aa",0.75)
            _draw_folds(ax,smf,"#00dcff","-",1.4)
            _draw_folds(ax,svf,"#00ffcc","--",0.9)

        # Draw antennas (top of body)
        for i in range(na):
            av,af,amf,avf,mb,mt=build_antenna(i,na)
            _draw_poly(ax,av,af,"#00cc99",0.6)
            _draw_folds(ax,amf,"#00ffcc")
            _draw_folds(ax,avf,"#aaffee","--")
            _draw_rod(ax,mb,mt,"#00ffcc",lw=2)

        # Draw reflectors (bottom of body)
        for i in range(nr):
            rv,rf,rmf,rvf,cb,cr=build_reflector(i,nr)
            _draw_poly(ax,rv,rf,"#9944cc",0.6)
            _draw_folds(ax,rmf,"#dd88ff")
            _draw_folds(ax,rvf,"#cc66ff","--")
            _draw_rod(ax,cb,cr,"#bb66ff",lw=2)

        # Axis limits — tall standing satellite
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_zlim(-2, 2)

    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D
    fig.legend(handles=[
        Patch(facecolor="#1144aa",edgecolor="#00b4ff",label=f"Solar Panels ({ns})"),
        Patch(facecolor="#00cc99",edgecolor="#00b4ff",label=f"Antenna ({na})"),
        Patch(facecolor="#9944cc",edgecolor="#00b4ff",label=f"Reflector ({nr})"),
        Patch(facecolor="#cc8800",edgecolor="#00b4ff",label=f"Truss ({nt})"),
        Patch(facecolor="#1a3055",edgecolor="#00b4ff",label="Body"),
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
def f2tri(verts,faces):
    idx=[]
    for f in faces:
        if len(f)==4: a,b,c,d=f; idx+=[a,b,c,a,c,d]
        elif len(f)==3: idx+=list(f)
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
    tv,tf,tmf,tvf=build_truss(nt); meshes.append(to3js(tv,tf,tmf,tvf,"#ffaa00",0.5,"#cc8800"))
    bv,bf,bfolds=build_body();      meshes.append(to3js(bv,bf,bfolds,[],"#2255aa",0.85,"#00dcff"))
    lc=(ns+1)//2; rc=ns//2
    for i in range(lc):
        sv,sf,smf,svf,_,_=build_solar_panel(-1,i,lc); meshes.append(to3js(sv,sf,smf,svf,"#2255cc",0.8,"#00dcff"))
    for i in range(rc):
        sv,sf,smf,svf,_,_=build_solar_panel(+1,i,rc); meshes.append(to3js(sv,sf,smf,svf,"#2255cc",0.8,"#00dcff"))
    for i in range(na):
        av,af,amf,avf,mb,mt=build_antenna(i,na)
        meshes.append(to3js(av,af,amf,avf,"#00ffcc",0.7,"#00aa88")); meshes.append(mast3d(mb,mt,"#00ffcc"))
    for i in range(nr):
        rv,rf,rmf,rvf,cb,cr=build_reflector(i,nr)
        meshes.append(to3js(rv,rf,rmf,rvf,"#bb66ff",0.7,"#8833cc")); meshes.append(mast3d(cb,cr,"#bb66ff"))
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
