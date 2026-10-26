import sys
from pathlib import Path
if "OUTPUT_DIR" not in globals():
    raise SystemExit("Use bash scripts/reproduce.sh figures")
from map_gpah_paper1_common import topology_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

out_dir = Path(OUTPUT_DIR)
PDF = out_dir / 'fig_compat_topologies.pdf'
PNG = out_dir / 'fig_compat_topologies_preview.png'
SVG = out_dir / 'fig_compat_topologies_editable.svg'
P_COLOR='#71669e'; Q_COLOR='#8fd3e8'; BOTTLENECK='#cc70af'; EDGE='#7cb4cc'; HL='#e098c7'; TEXT='#71669e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig, axes = plt.subplots(2, 3, figsize=(14, 8.4)); axes=axes.ravel()

def setup(ax,title):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
def draw_edge(ax,a,b,color=EDGE,lw=1.6,alpha=.65,rad=0,z=1,ms=14,sa=16,sb=20):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,linewidth=lw,color=color,alpha=alpha,shrinkA=sa,shrinkB=sb,connectionstyle=f'arc3,rad={rad}',zorder=z))
def draw_nodes(ax,ppos,qpos,bottleneck=None):
    xs,ys=zip(*[ppos[i] for i in sorted(ppos)]); ax.scatter(xs,ys,s=380,marker='s',c=P_COLOR,edgecolors='white',linewidths=1.4,zorder=5)
    for i in sorted(qpos):
        ax.scatter([qpos[i][0]],[qpos[i][1]],s=430,marker='o',c=BOTTLENECK if i==bottleneck else Q_COLOR,edgecolors='white',linewidths=1.4,zorder=5)
def label(ax,x,y,s,ha='center',va='center',fs=12):
    ax.text(x,y,s,fontsize=fs,color=TEXT,ha=ha,va=va,zorder=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.98,boxstyle='round,pad=0.08'))
def label_nodes(ax,ppos,qpos,poff=None,qoff=None,fs=12):
    poff=poff or {}; qoff=qoff or {}
    for i,(x,y) in ppos.items():
        dx,dy,ha,va=poff.get(i,(0,-.105,'center','center')); label(ax,x+dx,y+dy,rf'$P_{{{i+1}}}$',ha,va,fs)
    for i,(x,y) in qpos.items():
        dx,dy,ha,va=qoff.get(i,(0,.085,'center','center')); label(ax,x+dx,y+dy,rf'$Q_{{{i+1}}}$',ha,va,fs)
def draw_from_compat(ax,compat,ppos,qpos,rad_default=0,special=None):
    special=special or {}
    for k in range(compat.shape[0]):
        for i in range(compat.shape[1]):
            if compat[k,i]:
                opts=special.get((k,i),{}); draw_edge(ax,ppos[k],qpos[i],**({'rad':rad_default}|opts))

ax=axes[0]; setup(ax,'chain')
ppos={0:(.10,.76),1:(.10,.48),2:(.45,.33),3:(.82,.48),4:(.82,.76)}
qpos={0:(.30,.83),1:(.30,.66),2:(.30,.49),3:(.30,.32),4:(.62,.32),5:(.62,.49),6:(.62,.66),7:(.62,.83)}
draw_from_compat(ax,topology_matrix('chain',8,5,.35,7),ppos,qpos,rad_default=.04); draw_nodes(ax,ppos,qpos)
label_nodes(ax,ppos,qpos,poff={0:(-.09,0,'right','center'),1:(-.09,0,'right','center'),2:(0,-.10,'center','center'),3:(.09,0,'left','center'),4:(.09,0,'left','center')},fs=12)

ax=axes[1]; setup(ax,'ring')
qpos={
    0:(.82,.55),1:(.68,.72),2:(.48,.77),3:(.28,.67),
    4:(.18,.47),5:(.30,.28),6:(.55,.23),7:(.76,.34),
}
ppos={0:(.93,.68),1:(.52,.91),2:(.10,.30),3:(.60,.06),4:(.96,.35)}
ring_edges={0:[7,0,1],1:[1,2,3],2:[2,3,4],3:[4,5,6],4:[5,6,7]}
ring_rad={
    (0,7):.10,(0,0):0,(0,1):-.08,
    (1,1):.10,(1,2):0,(1,3):-.08,
    (2,2):.10,(2,3):.04,(2,4):-.06,
    (3,4):.10,(3,5):0,(3,6):-.08,
    (4,5):.10,(4,6):.03,(4,7):-.06,
}
for k, targets in ring_edges.items():
    for i in targets:
        draw_edge(ax,ppos[k],qpos[i],color=EDGE,lw=1.25,alpha=.48,rad=ring_rad.get((k,i),0),z=1,ms=11,sa=18,sb=22)
draw_nodes(ax,ppos,qpos)
label_nodes(ax,ppos,qpos,
            poff={0:(.080,.070,'left','center'),1:(-.075,.030,'right','center'),2:(-.095,-.040,'right','center'),3:(.025,-.110,'left','center'),4:(.095,-.035,'left','center')},
            qoff={0:(.095,0,'left','center'),1:(.070,.090,'left','center'),2:(-.035,.110,'right','center'),3:(-.090,.085,'right','center'),4:(-.115,.000,'right','center'),5:(-.110,-.095,'right','center'),6:(.115,-.020,'left','center'),7:(.120,-.030,'left','center')},fs=12)

ax=axes[2]; setup(ax,'star')
ppos={0:(.50,.88),1:(.17,.08),2:(.39,.08),3:(.61,.08),4:(.83,.08)}
qpos={0:(.10,.55),1:(.30,.58),2:(.70,.58),3:(.90,.55),4:(.20,.34),5:(.40,.32),6:(.60,.32),7:(.80,.34)}
for k,targets in {1:[0,4],2:[1,5],3:[2,6],4:[3,7]}.items():
    for i in targets:
        draw_edge(ax,ppos[k],qpos[i],color=EDGE,lw=1.45,alpha=.56,rad=0,z=1,ms=11,sa=18,sb=22)
p1_sources={0:(.30,.80),1:(.39,.80),2:(.61,.80),3:(.70,.80),4:(.34,.78),5:(.45,.78),6:(.55,.78),7:(.66,.78)}
p1_rad={0:.20,1:.11,2:-.11,3:-.20,4:.28,5:.14,6:-.14,7:-.28}
for i in range(8):
    draw_edge(ax,p1_sources[i],qpos[i],color=HL,lw=1.8,alpha=.76,rad=p1_rad[i],z=2,ms=13,sa=0,sb=22)
draw_nodes(ax,ppos,qpos)
label_nodes(ax,ppos,qpos,
            poff={0:(.085,0,'left','center'),1:(0,-.105,'center','center'),2:(0,-.105,'center','center'),3:(0,-.105,'center','center'),4:(0,-.105,'center','center')},
            qoff={0:(-.090,.040,'right','center'),1:(-.155,.030,'right','center'),2:(.155,.030,'left','center'),3:(.090,.040,'left','center'),4:(-.135,.010,'right','center'),5:(.135,.010,'left','center'),6:(-.135,.010,'right','center'),7:(.135,.010,'left','center')},fs=12)

ax=axes[3]; setup(ax,'bridge bottleneck')
ppos={0:(.12,.82),1:(.28,.82),2:(.50,.82),3:(.72,.82),4:(.88,.82)}
qpos={0:(.08,.30),1:(.20,.30),2:(.32,.30),3:(.46,.18),4:(.58,.30),5:(.70,.30),6:(.82,.30),7:(.94,.30)}
compat=topology_matrix('bridge_bottleneck',8,5,.35,7)
for k in range(compat.shape[0]):
    for i in range(compat.shape[1]):
        if compat[k,i] and i != 3:
            draw_edge(ax,ppos[k],qpos[i],color=EDGE,lw=1.18,alpha=.38,rad=0,z=1,ms=10,sa=18,sb=22)
for k, rad in [(0,.18),(1,.10),(2,0),(3,-.10),(4,-.18)]:
    draw_edge(ax,ppos[k],qpos[3],color=HL,lw=2.0,alpha=.86,rad=rad,z=2,ms=14,sa=18,sb=24)
draw_nodes(ax,ppos,qpos,bottleneck=3)
label_nodes(ax,ppos,qpos,
            poff={i:(0,.095,'center','center') for i in ppos},
            qoff={i:(0,-.095,'center','center') for i in qpos},fs=12)

ax=axes[4]; setup(ax,'high overlap')
ppos={0:(.12,.82),1:(.31,.82),2:(.50,.82),3:(.69,.82),4:(.88,.82)}
qpos={0:(.08,.28),1:(.20,.20),2:(.32,.28),3:(.44,.20),4:(.56,.28),5:(.68,.20),6:(.80,.28),7:(.92,.20)}
# The dense topology is shown as shared coverage of central queues plus a few side links.
for k, i, rad in [(0,3,.10),(1,3,.05),(2,3,0),(3,4,-.05),(4,4,-.10)]:
    draw_edge(ax,ppos[k],qpos[i],color=HL,lw=1.9,alpha=.76,rad=rad,z=2,ms=12,sa=18,sb=22)
for k, targets in {0:[0],1:[1,2],2:[5],3:[6],4:[7]}.items():
    for i in targets:
        draw_edge(ax,ppos[k],qpos[i],color=EDGE,lw=1.2,alpha=.42,rad=0,z=1,ms=10,sa=18,sb=22)
draw_nodes(ax,ppos,qpos)
label_nodes(ax,ppos,qpos,poff={i:(0,.095,'center','center') for i in ppos},qoff={i:(0,-.095,'center','center') for i in qpos},fs=11)

ax=axes[5]; setup(ax,'random sparse')
ppos={0:(.12,.82),1:(.31,.82),2:(.50,.82),3:(.69,.82),4:(.88,.82)}
qpos={0:(.08,.28),1:(.20,.20),2:(.32,.28),3:(.44,.20),4:(.56,.28),5:(.68,.20),6:(.80,.28),7:(.92,.20)}
# A readable representative sparse graph; exact random seeds are reported in the scripts.
for k, targets in {0:[0,2],1:[1,4],2:[3,5],3:[4,6],4:[6,7]}.items():
    for i in targets:
        draw_edge(ax,ppos[k],qpos[i],color=EDGE,lw=1.35,alpha=.52,rad=0,z=1,ms=10,sa=18,sb=22)
draw_nodes(ax,ppos,qpos)
label_nodes(ax,ppos,qpos,poff={i:(0,.095,'center','center') for i in ppos},qoff={i:(0,-.095,'center','center') for i in qpos},fs=11)
plt.tight_layout(pad=1.2,w_pad=1.8,h_pad=2.2)
fig.savefig(PDF,bbox_inches='tight')
fig.savefig(PNG,dpi=220,bbox_inches='tight')
fig.savefig(SVG,bbox_inches='tight')
print(PDF); print(PNG); print(SVG)
