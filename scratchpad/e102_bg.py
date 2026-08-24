"""E102: does a gate that must CHOOSE stay selective, and does the benefit survive?"""
import os, time
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, TEST = int(30*60/CFG.dt), int(10*60/CFG.dt)
SEED0, SEEDS = int(os.environ.get("E102_SEED0", "0")), 8
BASE = dict(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)
NAMES = {spec.M_FORWARD:"FORWARD", spec.M_TURN_L:"TURN_L", spec.M_TURN_R:"TURN_R",
         spec.M_PECK:"PECK", spec.M_SCRATCH:"SCRATCH", spec.M_CROUCH:"CROUCH",
         spec.M_FLEE:"FLEE", spec.M_CALL_AERIAL:"CALL_AERIAL",
         spec.M_CALL_GROUND:"CALL_GROUND", spec.M_CALL_CONTACT:"CALL_CONTACT",
         spec.M_CALL_FOOD:"CALL_FOOD", spec.M_CALL_GAKEL:"CALL_GAKEL"}


def rear(pc, seed, do_rear=True):
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k,1), regions.DEFAULT_REGIONS, n_hens=16)
    ps = plasticity.initial_state(p, 16, pc)
    if do_rear:
        w = world.reset(k,CFG); x = brain.initial_state(p,16)
        _w,_x,p,ps,_k,_t = simulate.rollout(w,x,p,jax.random.fold_in(k,2),CFG,REAR,pc=pc,ps=ps)
    return p, ps, k


def predation(p, ps, k, pc):
    pc_off = plasticity.PlasticConfig(enabled=False, bg_gate=pc.bg_gate,
                                      bg_lateral=pc.bg_lateral,
                                      reflex_gate=pc.reflex_gate)
    w = world.reset(k,CFG); x = brain.initial_state(p,16)
    wf,*_ = simulate.rollout(w,x,p,jax.random.fold_in(k,7),CFG,TEST,pc=pc_off,ps=ps)
    d = float(jnp.sum(wf.n_dives)); c = float(jnp.sum(wf.n_caught_any))
    return c/max(d,1), float(jnp.mean(wf.hunger))


t0 = time.perf_counter()
BG  = plasticity.PlasticConfig(**BASE, bg_gate=True)
OFF = plasticity.PlasticConfig(**BASE)
print(f"E102 -- basal-ganglia competitive gate. seeds {SEED0}-{SEED0+SEEDS-1}\n")

# --- 2x2 -------------------------------------------------------------------
R = {}
print(f"{'cell':>22}{'caught/dive':>13}{'hunger':>9}")
for reared in (False, True):
    for gate in (False, True):
        pc = BG if gate else OFF
        cd, hu = [], []
        for s in range(SEED0, SEED0+SEEDS):
            p, ps, k = rear(pc, s, reared)
            a, b = predation(p, ps, k, pc)
            cd.append(a); hu.append(b)
        R[(reared,gate)] = np.array(cd)
        nm = f"{'reared' if reared else 'untrained'}, {'BG gate' if gate else 'no gate'}"
        print(f"{nm:>22}{np.mean(cd):>13.4f}{np.mean(hu):>9.3f}")

def paired(a,b,label,crit=2.365):
    d = b-a; se = d.std(ddof=1)/np.sqrt(len(d)); t = d.mean()/max(se,1e-12)
    print(f"  {label:<38}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}  "
          f"{'SIGNIFICANT' if abs(t)>crit else 'not significant'}")

print(f"\npaired, df=7, crit 2.365:")
paired(R[(True,False)], R[(True,True)], "BG gate effect, reared brain")
paired(R[(False,False)], R[(False,True)], "BG gate effect, untrained brain")
paired(R[(False,False)], R[(True,True)], "reared+BG vs untrained baseline")

# --- gate profile ----------------------------------------------------------
gates, sd = [], []
for s in range(SEED0, SEED0+SEEDS):
    p, ps, k = rear(BG, s, True)
    w = world.reset(k,CFG); x = brain.initial_state(p,16)
    obs = __import__("coop.sensing", fromlist=["observe"]).observe(w, CFG)
    seq = []
    for _ in range(300):
        x, motor, d = brain.step(x, obs, p, CFG.dt, bg_gate=True, bg_lateral=1.0)
        n_m = p.W_out.shape[-1]
        st = neurons.rate(x)[:, -n_m:]
        sdr = jnp.einsum("hmn,hn->hm", p.W_str, st)
        seq.append(np.asarray(jax.nn.sigmoid(4.0 + sdr - jnp.mean(sdr,-1,keepdims=True))))
    seq = np.stack(seq)
    gates.append(seq[-1].mean(0))
    # state-dependence: cosine of each step's gate vector against its own mean
    a = seq.reshape(-1, seq.shape[-1]); m = a.mean(0); m /= np.linalg.norm(m)+1e-12
    sd.append(float(((a@m)/(np.linalg.norm(a,axis=1)+1e-12)).mean()))
G = np.mean(gates,0)
print(f"\ngate per channel (E101 free gate closed 11 of 12 below 0.9):")
for i in np.argsort(G):
    print(f"  {NAMES.get(i,f'ch{i}'):>14}  {G[i]:.4f}{'   <-- closed' if G[i]<0.9 else ''}")
print(f"\n  channels below 0.9 : {int((G<0.9).sum())} of 12   (falsifier fires if >6)")
print(f"  mean gate          : {G.mean():.4f}   (hatch 0.982; falsifier fires if |d|>0.05)")
print(f"  spread (max-min)   : {G.max()-G.min():.4f}")
print(f"  gate state-dependence (1.0 = fixed vector): {np.mean(sd):.4f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
