"""
app.py — Satellite Origami Backend (Production Ready for Render)
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
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rag_pipeline import RAGPipeline

app = Flask(__name__)
CORS(app)

BASE           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE, "cvae_model.pth")
KNOWLEDGE_PATH = os.path.join(BASE, "knowledge_base.json")
FRONTEND_PATH  = os.path.join(BASE, "chatbot.html")

LABEL_NAMES  = {0:"solar_panel", 1:"antenna", 2:"reflector", 3:"truss"}
NUM_CLASSES  = 4
INPUT_DIM    = 140
LATENT_DIM   = 16
BODY_X, BODY_Y, BODY_Z = 1.0, 0.4, 0.7

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
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def build_body():
    bx,by,bz = BODY_X,BODY_Y,BODY_Z
    v = np.array([
        [-bx,-by,-bz],[ bx,-by,-bz],[ bx, by,-bz],[-bx, by,-bz],
        [-bx,-by, bz],[ bx,-by, bz],[ bx, by, bz],[-bx, by, bz],
    ])
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    folds = [[v[0].tolist(),v[6].tolist()],[v[1].tolist(),v[7].tolist()],
             [v[2].tolist(),v[4].tolist()],[v[3].tolist(),v[5].tolist()]]
    return v, faces, folds

def build_solar_panel(side, panel_index=0, n_panels_this_side=1):
    pw=3.0; ph=0.05; nz=3; nx=6
    z_spacing = (2*BODY_Z) / max(n_panels_this_side,1)
    z_start   = -BODY_Z + panel_index*z_spacing
    z_end     = min(z_start + z_spacing*0.92, BODY_Z)
    x_start   = BODY_X       if side==1 else -(BODY_X+pw)
    x_end     = BODY_X+pw    if side==1 else -BODY_X
    verts = []
    for xi in np.linspace(x_start,x_end,nx+1):
        for zi in np.linspace(z_start,z_end,nz+1):
            verts.append([xi,-ph/2,zi])
    verts = np.array(verts)
    faces = []
    for i in range(nx):
        for j in range(nz):
            a=i*(nz+1)+j; b=a+1; c=(i+1)*(nz+1)+j+1; d=(i+1)*(nz+1)+j
            faces.append([a,b,c,d])
    mfolds = [[verts[j].tolist(), verts[nx*(nz+1)+j].tolist()] for j in range(1,nz+1,2)]
    vfolds = [[verts[i*(nz+1)].tolist(), verts[i*(nz+1)+nz].tolist()] for i in range(1,nx+1,2)]
    mid_z  = (z_start+z_end)/2
    return verts,faces,mfolds,vfolds,[BODY_X*side,0,mid_z],[BODY_X*side,0,mid_z]

def build_antenna(index=0, n_total=1):
    r_max=0.9; n_rings=5; n_segs=12; height=0.6
    x_off  = (index-(n_total-1)/2)*1.5
    dish_y = BODY_Y+height
    verts  = []
    for i in range(n_rings+1):
        r = r_max*i/n_rings
        y = dish_y+0.3*(r/r_max)**2
        for j in range(n_segs):
            a = 2*np.pi*j/n_segs
            verts.append([x_off+r*np.cos(a),y,r*np.sin(a)])
    verts = np.array(verts)
    faces = []
    for i in range(n_rings):
        for j in range(n_segs):
            faces.append([i*n_segs+j, i*n_segs+(j+1)%n_segs,
                          (i+1)*n_segs+(j+1)%n_segs, (i+1)*n_segs+j])
    center = [x_off,dish_y,0]
    mfolds = [[center, verts[n_rings*n_segs+j].tolist()] for j in range(n_segs)]
    vfolds = []
    for i in range(1,n_rings+1,2):
        ring = [verts[i*n_segs+j].tolist() for j in range(n_segs)]+[verts[i*n_segs].tolist()]
        vfolds += [[ring[k],ring[k+1]] for k in range(len(ring)-1)]
    return verts,faces,mfolds,vfolds,[x_off,BODY_Y,0],[x_off,dish_y,0]

def build_reflector(index=0, n_total=1):
    size=1.0; depth=0.5; n=8
    x_off = (index-(n_total-1)/2)*1.6
    verts = []
    for xi in np.linspace(-size,size,n):
        for zi in np.linspace(-size,size,n):
            y = -BODY_Y-0.4-depth*(xi**2+zi**2)/(size**2)
            verts.append([x_off+xi,y,zi])
    verts = np.array(verts)
    faces = []
    for i in range(n-1):
        for j in range(n-1):
            faces.append([i*n+j,i*n+j+1,(i+1)*n+j+1,(i+1)*n+j])
    mfolds = []
    for i in range(n):
        row = [verts[i*n+j].tolist() for j in range(n)]
        mfolds += [[row[k],row[k+1]] for k in range(len(row)-1)]
    vfolds = [[verts[i*n+j].tolist(),verts[(i+1)*n+j+1].tolist()]
              for i in range(n-1) for j in range(n-1) if (i+j)%2==0]
    return verts,faces,mfolds,vfolds,[x_off,-BODY_Y,0],[x_off,-BODY_Y-0.4,0]

def build_truss(n_segments=1):
    length=BODY_Z*2*1.1; radius=0.12; n_rings=8*n_segments; n_sides=6
    verts=[]; faces=[]; mfolds=[]; vfolds=[]; rings=[]
    for i in range(n_rings+1):
        z=(-length/2)+i*(length/n_rings); rot=(np.pi/n_sides)*(i%2); ring=[]
        for j in range(n_sides):
            a=2*np.pi*j/n_sides+rot; r=radius+0.015*(j%2)
            verts.append([r*np.cos(a),r*np.sin(a),z]); ring.append(len(verts)-1)
        rings.append(ring)
    verts=np.array(verts)
    for i in range(n_rings):
        r0,r1=rings[i],rings[i+1]
        for j in range(n_sides):
            faces.append([r0[j],r0[(j+1)%n_sides],r1[(j+1)%n_sides],r1[j]])
            mfolds.append([verts[r0[j]].tolist(),verts[r1[j]].tolist()])
            vfolds.append([verts[r0[j]].tolist(),verts[r1[(j+1)%n_sides]].tolist()])
    return verts,faces,mfolds,vfolds

# ══════════════════════════════════════════════════════════════════════════════
# PNG RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render_satellite_png(config):
    ns=int(config.get("solar_panel",2)); na=int(config.get("antenna",1))
    nr=int(config.get("reflector",1));   nt=int(config.get("truss",1))
    BG="#010812"
    fig=plt.figure(figsize=(16,9),facecolor=BG)
    for idx,(elev,azim,title) in enumerate([(55,-60,"FLYING VIEW — ORBITAL PERSPECTIVE"),(10,-90,"SIDE VIEW")]):
        ax=fig.add_subplot(1,2,idx+1,projection="3d")
        ax.set_facecolor(BG); ax.view_init(elev=elev,azim=azim)
        for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            pane.fill=False; pane.set_edgecolor(BG)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
        ax.set_title(title,fontsize=10,fontweight="bold",color="#00dcff",fontfamily="monospace",pad=8)

        def poly(v,f,fc,alpha):
            if not f: return
            ax.add_collection3d(Poly3DCollection([[v[vi] for vi in face] for face in f],
                alpha=alpha,facecolor=fc,edgecolor="#00b4ff",linewidth=0.2))

        def fl(folds,color,ls="-",lw=0.9):
            for f in folds:
                ax.plot([f[0][0],f[1][0]],[f[0][1],f[1][1]],[f[0][2],f[1][2]],
                        color=color,lw=lw,ls=ls,alpha=0.95)

        def rod(p0,p1,color="#ffffff",lw=3):
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],
                    color=color,lw=lw,alpha=1.0,solid_capstyle="round",zorder=10)

        tv,tf,tmf,tvf=build_truss(nt); poly(tv,tf,"#cc8800",0.5); fl(tmf,"#ffcc44"); fl(tvf,"#ffaa00","--")
        bv,bf,bfolds=build_body();      poly(bv,bf,"#4a6080",0.85); fl(bfolds,"#88aacc")

        lc=(ns+1)//2; rc=ns//2
        for i in range(lc):
            sv,sf,smf,svf,_,_=build_solar_panel(-1,i,lc)
            poly(sv,sf,"#3366cc",0.7); fl(smf,"#00dcff",lw=1.2); fl(svf,"#00ffcc","--")
        for i in range(rc):
            sv,sf,smf,svf,_,_=build_solar_panel(+1,i,rc)
            poly(sv,sf,"#3366cc",0.7); fl(smf,"#00dcff",lw=1.2); fl(svf,"#00ffcc","--")
        for i in range(na):
            av,af,amf,avf,mb,mt=build_antenna(i,na)
            poly(av,af,"#00cc99",0.55); fl(amf,"#00ffcc"); fl(avf,"#aaffee","--"); rod(mb,mt,"#00ffcc")
        for i in range(nr):
            rv,rf,rmf,rvf,cb,cr=build_reflector(i,nr)
            poly(rv,rf,"#9944cc",0.55); fl(rmf,"#dd88ff"); fl(rvf,"#cc66ff","--"); rod(cb,cr,"#bb66ff")

        ax.set_xlim(-5,5); ax.set_ylim(-2,3); ax.set_zlim(-2,2)

    from matplotlib.patches import Patch
    from matplotlib.lines   import Line2D
    fig.legend(handles=[
        Patch(facecolor="#3366cc",edgecolor="#00b4ff",label=f"Solar Panels ({ns})"),
        Patch(facecolor="#00cc99",edgecolor="#00b4ff",label=f"Antenna ({na})"),
        Patch(facecolor="#9944cc",edgecolor="#00b4ff",label=f"Reflector ({nr})"),
        Patch(facecolor="#cc8800",edgecolor="#00b4ff",label=f"Truss ({nt})"),
        Patch(facecolor="#4a6080",edgecolor="#00b4ff",label="Body"),
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
    tv,tf,tmf,tvf=build_truss(nt); meshes.append(to3js(tv,tf,tmf,tvf,"#ffaa00",0.6,"#cc8800"))
    bv,bf,bfolds=build_body();      meshes.append(to3js(bv,bf,bfolds,[],"#778899",0.9,"#aabbcc"))
    lc=(ns+1)//2; rc=ns//2
    for i in range(lc):
        sv,sf,smf,svf,_,_=build_solar_panel(-1,i,lc); meshes.append(to3js(sv,sf,smf,svf,"#4488ff",0.8,"#2266dd"))
    for i in range(rc):
        sv,sf,smf,svf,_,_=build_solar_panel(+1,i,rc); meshes.append(to3js(sv,sf,smf,svf,"#4488ff",0.8,"#2266dd"))
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
@app.route("/api/health",methods=["GET"])
def health():
    return jsonify({"status":"ok","message":"Satellite Origami Backend Running"})

@app.route("/api/generate",methods=["POST"])
def generate():
    data=request.json
    cfg={"solar_panel":data.get("solar_panel",2),"antenna":data.get("antenna",1),
         "reflector":data.get("reflector",1),"truss":data.get("truss",1)}
    return jsonify({"image":render_satellite_png(cfg),"config":cfg,"status":"generated"})

@app.route("/api/generate3d",methods=["POST"])
def generate3d():
    data=request.json
    cfg={"solar_panel":data.get("solar_panel",2),"antenna":data.get("antenna",1),
         "reflector":data.get("reflector",1),"truss":data.get("truss",1)}
    return jsonify({"meshes":build_satellite_3d(cfg),"config":cfg,"status":"ok"})

@app.route("/api/chat",methods=["POST"])
def chat():
    data=request.json
    message=data.get("message","").strip()
    answer,sources=rag.answer(message)
    return jsonify({"reply":answer,
                    "sources":[{"topic":s.get("topic",""),"score":s.get("score",0)} for s in sources]})

@app.route("/")
def index():
    return send_file(FRONTEND_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# START — Render.com production
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nSatellite Origami Backend — Production")
    print("Routes: /  |  /api/generate  |  /api/generate3d  |  /api/chat  |  /api/health")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)