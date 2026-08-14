import re, statistics as st
rows=[]
for line in open('/home/user/sentient-chicken/scratchpad/seed_dist.log'):
    m=re.match(r'(\d+) ([+-][\d.]+) fed=([\d.]+)% dend_min=([\d.]+) n_at_food=(\d+)', line)
    if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))))
ch=[r[1] for r in rows]; fed=[r[2] for r in rows]
n=len(rows)
print("n=%d seeds, fixed condition, 10 min, 16 hens" % n)
print("hunger change: mean=%+.4f sd=%.4f min=%+.3f max=%+.3f" % (st.mean(ch), st.stdev(ch), min(ch), max(ch)))
print("fed %%:         mean=%.2f sd=%.2f min=%.2f max=%.2f (max/min=%.1fx)" % (st.mean(fed), st.stdev(fed), min(fed), max(fed), max(fed)/min(fed)))
mx=st.mean(ch); my=st.mean(fed)
cov=sum((a-mx)*(b-my) for a,b in zip(ch,fed))/(n-1)
print("corr(hunger change, fed%%) = %+.3f" % (cov/(st.stdev(ch)*st.stdev(fed))))
pred=[(1/1800)/(0.03*f/100) for f in fed]
mp=st.mean(pred)
cov2=sum((a-mx)*(b-mp) for a,b in zip(ch,pred))/(n-1)
print("corr(hunger change, predicted equilibrium 0.0185/f) = %+.3f" % (cov2/(st.stdev(ch)*st.stdev(pred))))
for sd_d,lab in ((0.028,'E020 block'),(0.121,'E021 block')):
    for eff in (0.02,):
        print("  detect %.3f at 80%% power, sd_d=%.3f (%s): n = %.0f seeds" % (eff, sd_d, lab, 8*(sd_d/eff)**2))
