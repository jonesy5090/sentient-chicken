import re, ast, statistics as st
rows=[]
for line in open('/home/user/sentient-chicken/scratchpad/pairing.log'):
    m=re.match(r'(\d+) (\{.*\}) \[', line)
    if m: rows.append(ast.literal_eval(m.group(2)))
n=len(rows)
print("n=%d matched seeds, 10 min, 16 hens, primary metric = within-run hunger change" % n)
for treat in ('learn','noise'):
    d=[r[treat][0]-r['fixed'][0] for r in rows]
    mean=st.mean(d); sd=st.stdev(d); se=sd/n**0.5
    print("  %-6s - fixed: mean %+0.4f  sd %.4f  SE %.4f  t=%.2f" % (treat, mean, sd, se, abs(mean)/se))
    print("           per-seed: %s" % " ".join("%+0.3f"%x for x in d))
fx=[r['fixed'][0] for r in rows]
print("  fixed alone: mean %+0.4f sd %.4f   fed%% range %.2f-%.2f" %
      (st.mean(fx), st.stdev(fx), min(r['fixed'][1] for r in rows), max(r['fixed'][1] for r in rows)))
# correlation between conditions across seeds = how much the pairing buys
for treat in ('learn','noise'):
    a=[r['fixed'][0] for r in rows]; b=[r[treat][0] for r in rows]
    ma,mb=st.mean(a),st.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))/(n-1)
    r=cov/(st.stdev(a)*st.stdev(b))
    print("  corr(fixed, %s) across seeds = %+.3f   (pairing only helps if this is near 1)" % (treat, r))
