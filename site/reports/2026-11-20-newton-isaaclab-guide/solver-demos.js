(() => {
'use strict';
// ---------- shared helpers ----------
const $ = s => document.querySelector(s);
const COL = {};
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const players = new Map();
const viewport = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) players.get(entry.target)?.stop();
  }
});

function registerPlayback(id, stop, { keep = false } = {}) {
  const button = document.getElementById(id);
  players.set(button, { stop, keep });
  viewport.observe(button);
}

function syncPlayback() {
  for (const [button, { stop, keep }] of players) {
    const hide = motion.matches && !keep;
    if (hide || document.hidden || !button.getClientRects().length) stop();
    if (button.hidden !== hide) button.hidden = hide;
  }
}

let threeReady;
function loadThree() {
  return threeReady ??= new Promise(resolve => {
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
    script.addEventListener('load', resolve, { once: true });
    script.addEventListener('error', resolve, { once: true });
    document.head.append(script);
  });
}
function hexRgb(h){ h=h.trim().replace('#',''); if(h.length===3) h=h.split('').map(c=>c+c).join(''); const n=parseInt(h,16); return [n>>16&255,n>>8&255,n&255]; }
function readCols(){ const cs=getComputedStyle(document.documentElement); for(const k of ['bg','surface','ink','muted','line','grid','particle','node','strain','collider']){ const v=cs.getPropertyValue('--'+k).trim(); COL[k]=v; COL[k+'RGB']=hexRgb(v); } }
function rgba(k,a){ const c=COL[k+'RGB']; return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }
function mix(k1,k2,t){ const a=COL[k1+'RGB'], b=COL[k2+'RGB']; return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]; }
function fit(cv, aspect){ const w=Math.max(200, cv.clientWidth || cv.parentElement.clientWidth); const h=Math.round(w*aspect); const d=window.devicePixelRatio||1; cv.style.height=h+'px'; cv.width=Math.round(w*d); cv.height=Math.round(h*d); const ctx=cv.getContext('2d'); ctx.setTransform(d,0,0,d,0,0); return {ctx,w,h}; }
function arrow(ctx,x0,y0,x1,y1,color,lw){ const dx=x1-x0, dy=y1-y0, L=Math.hypot(dx,dy); if(L<1) return; ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=lw; ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke(); const hh=Math.min(6,L*.45), a=Math.atan2(dy,dx); ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x1-hh*Math.cos(a-.45),y1-hh*Math.sin(a-.45)); ctx.lineTo(x1-hh*Math.cos(a+.45),y1-hh*Math.sin(a+.45)); ctx.closePath(); ctx.fill(); }
function rng(seed){ let a=seed; return () => { a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function seg(id, onChange){ const el=$(id); el.addEventListener('click', e => { const b=e.target.closest('button'); if(!b) return; el.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed', x===b?'true':'false')); onChange(b.dataset.v); }); }
const FONT_MONO = "11px 'JetBrains Mono', ui-monospace, Menlo, monospace";
function solveDense(A, b, n){ // Gaussian elimination with partial pivoting, returns null if singular
  const M=A.map((r,i)=>r.slice(0,n).concat([b[i]]));
  let scale=0; for(let i=0;i<n;i++) scale=Math.max(scale,Math.abs(M[i][i]));
  for(let c=0;c<n;c++){ let p=c; for(let r=c+1;r<n;r++) if(Math.abs(M[r][c])>Math.abs(M[p][c])) p=r;
    if(Math.abs(M[p][c]) < 1e-7*Math.max(scale,1e-12)) return null; [M[c],M[p]]=[M[p],M[c]];
    for(let r=c+1;r<n;r++){ const f=M[r][c]/M[c][c]; if(f) for(let k=c;k<=n;k++) M[r][k]-=f*M[c][k]; } }
  const x=new Array(n).fill(0); for(let r=n-1;r>=0;r--){ let s=M[r][n]; for(let k=r+1;k<n;k++) s-=M[r][k]*x[k]; x[r]=s/M[r][r]; } return x; }
const GH='https://github.com/newton-physics/newton/blob/bcca9058890a25601b9ec93b0c4115be96f6a019/newton/_src/solvers/implicit_mpm/';
const link=(f,l,t)=>`<a href="${GH}${f}#L${l}" target="_blank" rel="noopener">${t}</a>`;

// =====================================================================
// A. Toy implicit MPM (2D, Q1 velocity, P0 strain, nodal contacts)
// =====================================================================
const SIM = (() => {
  const NX=20, NY=12, W1=NX+1, NN=W1*(NY+1), NC=NX*NY;
  const dt=1/30, g=-30, mp=0.25, drag=0.01, crit=0.5, vmax=25;
  const m=new Float64Array(NN), vx=new Float64Array(NN), vy=new Float64Array(NN), sx=new Float64Array(NN), sy=new Float64Array(NN);
  const cnx=new Float64Array(NN), cny=new Float64Array(NN), rn=new Float64Array(NN), rtx=new Float64Array(NN), rty=new Float64Array(NN);
  const isC=new Uint8Array(NN), cAct=new Uint8Array(NC), phi=new Float64Array(NC), lam=new Float64Array(NC*3);
  const S={P:[],stage:0,residual:0,steps:0,mu:0.6,iters:40,playing:false,hover:-1,cv:null,view:null};
  const idx=(i,j)=>j*W1+i;
  function clearGrid(){ m.fill(0);vx.fill(0);vy.fill(0);sx.fill(0);sy.fill(0);isC.fill(0);cAct.fill(0);phi.fill(0);lam.fill(0);rn.fill(0);rtx.fill(0);rty.fill(0); }
  function reset(){ const r=rng(11); S.P=[]; for(let a=0;a<10;a++) for(let b=0;b<12;b++){ const x=2.5+(a+.5)*.5+(r()-.5)*.1, y=4.5+(b+.5)*.5+(r()-.5)*.1; S.P.push({x,y,vx:7,vy:0,C:[0,0,0,0],ox:x,oy:y}); } S.stage=0; S.steps=0; S.residual=0; clearGrid(); }
  function cellOf(p){ return [Math.min(NX-1,Math.max(0,Math.floor(p.x))), Math.min(NY-1,Math.max(0,Math.floor(p.y)))]; }
  function bin(){ clearGrid(); for(const p of S.P){ const [i,j]=cellOf(p); cAct[j*NX+i]=1; } }
  function weights(p){ const [i,j]=cellOf(p); const fx=p.x-i, fy=p.y-j; const wx=[1-fx,fx], wy=[1-fy,fy]; const out=[]; for(let b=0;b<2;b++) for(let a=0;a<2;a++) out.push({n:idx(i+a,j+b),i:i+a,j:j+b,w:wx[a]*wy[b],gx:(a?1:-1)*wy[b],gy:wx[a]*(b?1:-1)}); return out; }
  function p2g(){
    for(const p of S.P){ for(const q of weights(p)){ const dx=q.i-p.x, dy=q.j-p.y; const ux=p.vx+p.C[0]*dx+p.C[1]*dy, uy=p.vy+dt*g+p.C[2]*dx+p.C[3]*dy; m[q.n]+=q.w*mp; vx[q.n]+=q.w*mp*ux; vy[q.n]+=q.w*mp*uy; } }
    for(let n=0;n<NN;n++){ if(m[n]>0){ vx[n]/=(m[n]+drag); vy[n]/=(m[n]+drag); sx[n]=vx[n]; sy[n]=vy[n];
      const i=n%W1, j=(n/W1)|0; let d=j-1, nx=0, ny=1; if(i-1<d){d=i-1;nx=1;ny=0;} if(NX-1-i<d){d=NX-1-i;nx=-1;ny=0;}
      if(d<=0){ isC[n]=1; cnx[n]=nx; cny[n]=ny; } } }
    for(let c=0;c<NC;c++) if(cAct[c]){ const i=c%NX, j=(c/NX)|0, n=idx(i,j); phi[c]=Math.min(1.2,(m[n]+m[n+1]+m[n+W1]+m[n+W1+1])/4); }
  }
  function cellRates(c){ const i=c%NX, j=(c/NX)|0; const n00=idx(i,j), n10=n00+1, n01=n00+W1, n11=n01+1;
    const ax=vx[n10]-vx[n00]+vx[n11]-vx[n01], by=vy[n01]-vy[n00]+vy[n11]-vy[n10], ay=vx[n01]-vx[n00]+vx[n11]-vx[n10], bx=vy[n10]-vy[n00]+vy[n11]-vy[n01];
    return [0.5*(ax+by), 0.5*(ax-by), 0.5*(ay+bx), n00,n10,n01,n11]; }
  function solve(){
    const cells=[]; for(let c=0;c<NC;c++) if(cAct[c] && phi[c]>=crit) cells.push(c);
    const contacts=[]; for(let n=0;n<NN;n++) if(isC[n]) contacts.push(n);
    const im=n=>1/(m[n]+drag);
    for(let it=0; it<S.iters; it++){
      for(const c of cells){ const f=phi[c]; const [ev,ed,es,n00,n10,n01,n11]=cellRates(c);
        const i00=im(n00),i10=im(n10),i01=im(n01),i11=im(n11); const W=f*f*0.5*(i00+i10+i01+i11); if(W<=0) continue;
        const L=c*3, ln=lam[L], ld=lam[L+1], ls=lam[L+2];
        const nn=Math.max(0, ln - f*ev/W); let td=ld - f*ed/W, ts=ls - f*es/W; const tn=Math.hypot(td,ts), lim=S.mu*nn; if(tn>lim){ const k=tn>0?lim/tn:0; td*=k; ts*=k; }
        const dn=nn-ln, dd=td-ld, ds=ts-ls; lam[L]=nn; lam[L+1]=td; lam[L+2]=ts; const a=0.5*f, P=dn+dd, Q=dn-dd;
        vx[n00]+=i00*a*(-P-ds); vx[n10]+=i10*a*(P-ds); vx[n01]+=i01*a*(-P+ds); vx[n11]+=i11*a*(P+ds);
        vy[n00]+=i00*a*(-Q-ds); vy[n10]+=i10*a*(-Q+ds); vy[n01]+=i01*a*(Q-ds); vy[n11]+=i11*a*(Q+ds); }
      for(const n of contacts){ const nx=cnx[n], ny=cny[n]; const ux=vx[n]-(rn[n]*nx+rtx[n]), uy=vy[n]-(rn[n]*ny+rty[n]);
        const vn=ux*nx+uy*ny, r=Math.max(0,-vn); const ux2=ux+r*nx, uy2=uy+r*ny; const vn2=ux2*nx+uy2*ny; const tx=ux2-vn2*nx, ty=uy2-vn2*ny, tl=Math.hypot(tx,ty);
        const lim=0.5*r, k=tl>lim?lim/tl:1; rn[n]=r; rtx[n]=-tx*k; rty[n]=-ty*k; vx[n]=ux2+rtx[n]; vy[n]=uy2+rty[n]; }
    }
    let res=0; for(const c of cells){ const ev=cellRates(c)[0]; res=Math.max(res,-ev); } S.residual=res;
  }
  function g2p(){
    for(const p of S.P){ let ux=0,uy=0; const C=[0,0,0,0]; for(const q of weights(p)){ ux+=q.w*vx[q.n]; uy+=q.w*vy[q.n]; C[0]+=vx[q.n]*q.gx; C[1]+=vx[q.n]*q.gy; C[2]+=vy[q.n]*q.gx; C[3]+=vy[q.n]*q.gy; }
      const sp=Math.hypot(ux,uy); if(sp>vmax){ ux*=vmax/sp; uy*=vmax/sp; }
      p.ox=p.x; p.oy=p.y; p.x+=dt*ux; p.y+=dt*uy;
      if(p.y<1.02){p.y=1.02; if(uy<0) uy=0;} if(p.x<1.02){p.x=1.02; if(ux<0) ux=0;} if(p.x>NX-1.02){p.x=NX-1.02; if(ux>0) ux=0;} if(p.y>NY-.02) p.y=NY-.02;
      p.vx=ux; p.vy=uy; p.C=C; }
  }
  // Convergence traces for Part 5b: freeze one landing step, run three iteration styles on it.
  function traces(N){
    reset(); for(let s=0;s<16;s++){ bin(); p2g(); solve(); g2p(); }
    bin(); p2g();
    const bx=Float64Array.from(vx), by=Float64Array.from(vy);
    const cells=[]; for(let c=0;c<NC;c++) if(cAct[c] && phi[c]>=crit) cells.push(c);
    const contacts=[]; for(let n=0;n<NN;n++) if(isC[n]) contacts.push(n);
    const im=n=>1/(m[n]+drag);
    const resid=()=>{ let r=0; for(const c of cells) r=Math.max(r,-cellRates(c)[0]); for(const n of contacts) r=Math.max(r,-(vx[n]*cnx[n]+vy[n]*cny[n])); return r; };
    function cellStep(c,w,X,Y){ const f=phi[c]; const [ev,ed,es,n00,n10,n01,n11]=cellRates(c);
      const i00=im(n00),i10=im(n10),i01=im(n01),i11=im(n11); const W=f*f*0.5*(i00+i10+i01+i11); if(W<=0) return;
      const L=c*3, ln=lam[L], ld=lam[L+1], ls=lam[L+2];
      const nn=Math.max(0, ln - f*ev/W); let td=ld - f*ed/W, ts=ls - f*es/W; const tn=Math.hypot(td,ts), lim=S.mu*nn; if(tn>lim){ const k=tn>0?lim/tn:0; td*=k; ts*=k; }
      const dn=w*(nn-ln), dd=w*(td-ld), ds=w*(ts-ls); lam[L]+=dn; lam[L+1]+=dd; lam[L+2]+=ds; const a=0.5*f, P=dn+dd, Q=dn-dd;
      X[n00]+=i00*a*(-P-ds); X[n10]+=i10*a*(P-ds); X[n01]+=i01*a*(-P+ds); X[n11]+=i11*a*(P+ds);
      Y[n00]+=i00*a*(-Q-ds); Y[n10]+=i10*a*(-Q+ds); Y[n01]+=i01*a*(Q-ds); Y[n11]+=i11*a*(Q+ds); }
    function contactStep(n,w,X,Y){ const nx=cnx[n], ny=cny[n]; const ox=rn[n]*nx+rtx[n], oy=rn[n]*ny+rty[n]; const ux=vx[n]-ox, uy=vy[n]-oy;
      const vn=ux*nx+uy*ny, r=Math.max(0,-vn); const ux2=ux+r*nx, uy2=uy+r*ny; const vn2=ux2*nx+uy2*ny; const tx=ux2-vn2*nx, ty=uy2-vn2*ny, tl=Math.hypot(tx,ty);
      const lim=.5*r, k=tl>lim?lim/tl:1; const nrn=r, nrx=-tx*k, nry=-ty*k;
      const drn=w*(nrn-rn[n]), drx=w*(nrx-rtx[n]), dry=w*(nry-rty[n]); rn[n]+=drn; rtx[n]+=drx; rty[n]+=dry;
      X[n]+=drn*nx+drx; Y[n]+=drn*ny+dry; }
    const start=()=>{ vx.set(bx); vy.set(by); lam.fill(0); rn.fill(0); rtx.fill(0); rty.fill(0); };
    const out={gs:[],jac:[],nes:[]};
    start(); out.gs.push(resid()); for(let it=0;it<N;it++){ for(const c of cells) cellStep(c,1,vx,vy); for(const n of contacts) contactStep(n,1,vx,vy); out.gs.push(resid()); }
    start(); out.jac.push(resid()); const dvx=new Float64Array(NN), dvy=new Float64Array(NN);
    for(let it=0;it<N;it++){ dvx.fill(0); dvy.fill(0); for(const c of cells) cellStep(c,.35,dvx,dvy); for(const n of contacts) contactStep(n,.35,dvx,dvy); for(let n=0;n<NN;n++){ vx[n]+=dvx[n]; vy[n]+=dvy[n]; } out.jac.push(resid()); }
    start(); out.nes.push(resid());
    const snap=()=>[Float64Array.from(lam),Float64Array.from(rn),Float64Array.from(rtx),Float64Array.from(rty),Float64Array.from(vx),Float64Array.from(vy)];
    const arrs=()=>[lam,rn,rtx,rty,vx,vy];
    let prev=snap(), k=0, last=Infinity;
    for(let it=0;it<N;it++){ for(const c of cells) cellStep(c,1,vx,vy); for(const n of contacts) contactStep(n,1,vx,vy);
      const res=resid(); out.nes.push(res); const cur=snap();
      if(res>last){ k=0; } else { k++; const beta=Math.max(0,(k-1)/(k+2)); arrs().forEach((A,q)=>{ for(let i=0;i<A.length;i++) A[i]=cur[q][i]+beta*(cur[q][i]-prev[q][i]); }); }
      prev=cur; last=res; }
    out.cells=cells.length; out.contacts=contacts.length;
    reset(); return out;
  }
  function next(){ if(S.stage===0||S.stage===4){ bin(); S.stage=1; } else if(S.stage===1){ p2g(); S.stage=2; } else if(S.stage===2){ solve(); S.stage=3; } else { g2p(); S.steps++; S.stage=4; } }
  function full(){ if(S.stage!==0 && S.stage!==4){ while(S.stage!==4) next(); } else { bin(); p2g(); solve(); g2p(); S.steps++; S.stage=4; } }

  const TEXT=[
    {h:'0 \u00B7 Particles carry the material', b:`<p>Each particle stores position, velocity, its velocity gradient <code>C</code> (for APIC), elastic strain, a hardening variable <code>Jp</code> and last step's stress. The grid keeps nothing between steps except warm-start data.</p><p>Arrows show particle velocity. Press <b>Next stage</b>.</p>`},
    {h:'1 \u00B7 Bin particles, build the grid', b:`<p>Find the cell under each particle and activate only those cells. Every active cell gets velocity nodes (here <code>Q1</code> corners, teal), one strain node (<code>P0</code>, violet cell) and contact points.</p><p class="src mono">${link('solver_implicit_mpm.py',2360,'step()')} \u00B7 ${link('solver_implicit_mpm.py',229,'_make_grid_basis_space')}</p>`},
    {h:'2 \u00B7 Particle \u2192 grid', b:`<div class="formula">m_i  = \u03A3 w_ip m_p
v*_i = \u03A3 w_ip m_p (v_p + \u0394t g + C_p(x_i\u2212x_p)) / m_i</div><p>Circle size is node mass; arrows are the free velocity v*, which already includes one step of gravity. Nothing stops the material yet. It could fall through the floor or squeeze into itself.</p><p class="src mono">${link('implicit_mpm_solver_kernels.py',88,'integrate_mass')} \u00B7 ${link('implicit_mpm_solver_kernels.py',108,'integrate_velocity')} \u00B7 ${link('implicit_mpm_solver_kernels.py',163,'free_velocity')}</p>`},
    {h:'3 \u00B7 Implicit solve: fix the grid velocities', b:`<p>After stage 2, every grid node (the corners of the cells) has one velocity, the grey arrows. It is just "free fall", and it breaks two rules:</p><p><b>Cells get squeezed.</b> A cell's four corner velocities say how the cell moves. If its top corners move down faster than its bottom corners, the cell gets flatter and its area shrinks. Sand can't be packed tighter, so the solver gives the cell a <b>pressure</b> that pushes its corners apart until the shrinking stops. The same pressure lets the cell resist sliding (shear), but only up to \u03BC \u00D7 pressure.</p><p><b>Corners move into walls.</b> Corners on the floor (the rose band) or the side walls are heading into the solid. The solver removes that part of their velocity and slows their sliding with friction.</p><p>Each fix disturbs the others, so the solver repeats them all until nothing changes. Result: teal arrows; violet shading = pressure; rose rings = wall corners. <b>Zoom into one cell below \u2193</b> to watch this happen step by step.</p><p class="src mono">${link('solve_rheology.py',2222,'solve_rheology')} \u00B7 ${link('solve_rheology.py',1150,'_GaussSeidelSolver')}</p>`},
    {h:'4 \u00B7 Grid \u2192 particle', b:`<div class="formula">v_p  = \u03A3 w_ip v_i
C_p  = \u03A3 v_i \u2297 \u2207w_ip
x_p \u2190 x_p + \u0394t v_p</div><p>Particles read the solved velocity with the same weights, keep the gradient for the next APIC transfer and for their strain update, then move. Faint lines show this step's motion. The grid is thrown away.</p><p class="src mono">${link('implicit_mpm_solver_kernels.py',342,'advect_particles')} \u00B7 ${link('implicit_mpm_solver_kernels.py',391,'update_particle_strains')}</p>`},
  ];

  function draw(){
    const cv=S.cv; const {ctx,w,h}=fit(cv, NY/NX); const s=w/NX; const X=x=>x*s, Y=y=>(NY-y)*s;
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle=COL.grid; ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=NX;i++){ ctx.moveTo(X(i)+.5,0); ctx.lineTo(X(i)+.5,h);} for(let j=0;j<=NY;j++){ ctx.moveTo(0,Y(j)+.5); ctx.lineTo(w,Y(j)+.5);} ctx.stroke();
    // walls
    ctx.fillStyle=rgba('collider',.13); ctx.fillRect(0,Y(1),w,s); ctx.fillRect(0,0,s,Y(1)); ctx.fillRect(X(NX-1),0,s,Y(1));
    ctx.strokeStyle=COL.collider; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(X(1),0); ctx.lineTo(X(1),Y(1)); ctx.lineTo(X(NX-1),Y(1)); ctx.lineTo(X(NX-1),0); ctx.stroke();
    const st = S.playing ? 'play' : S.stage;
    const ak=0.06*s;
    if(st===1){ for(let c=0;c<NC;c++) if(cAct[c]){ const i=c%NX, j=(c/NX)|0; ctx.fillStyle=rgba('strain',.12); ctx.fillRect(X(i),Y(j+1),s,s); ctx.strokeStyle=rgba('strain',.45); ctx.lineWidth=1; ctx.strokeRect(X(i)+.5,Y(j+1)+.5,s-1,s-1);
        ctx.fillStyle=COL.node; for(const [a,b] of [[0,0],[1,0],[0,1],[1,1]]){ ctx.beginPath(); ctx.arc(X(i+a),Y(j+b),2.4,0,7); ctx.fill(); } } }
    if(st===3 || st==='play' || st===4){ if(st!=='play' || $('#cPress').checked){ let mx=1e-9; for(let c=0;c<NC;c++) mx=Math.max(mx,lam[c*3]); for(let c=0;c<NC;c++) if(cAct[c]){ const i=c%NX, j=(c/NX)|0; const t=Math.min(1,lam[c*3]/mx); ctx.fillStyle=rgba('strain', (st===4?.06:.08)+(st===4?.25:.6)*Math.sqrt(t)); ctx.fillRect(X(i)+1,Y(j+1)+1,s-2,s-2); } } }
    if(st===2 || st===3){ for(let n=0;n<NN;n++) if(m[n]>0){ const i=n%W1, j=(n/W1)|0; const x0=X(i), y0=Y(j);
        if(st===3) arrow(ctx,x0,y0,x0+sx[n]*ak,y0-sy[n]*ak,rgba('muted',.55),1.2);
        ctx.fillStyle=rgba('node',.22); ctx.beginPath(); ctx.arc(x0,y0,2+s*.22*Math.sqrt(Math.min(1.3,m[n])),0,7); ctx.fill();
        const ux=st===3?vx[n]:sx[n], uy=st===3?vy[n]:sy[n]; arrow(ctx,x0,y0,x0+ux*ak,y0-uy*ak,COL.node,1.6);
        if(st===3 && isC[n]){ ctx.strokeStyle=COL.collider; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(x0,y0,5,0,7); ctx.stroke(); } } }
    // particles
    for(const p of S.P){ if(st===4){ ctx.strokeStyle=rgba('particle',.45); ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(X(p.ox),Y(p.oy)); ctx.lineTo(X(p.x),Y(p.y)); ctx.stroke(); }
      ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(X(p.x),Y(p.y),Math.max(2.2,s*.14),0,7); ctx.fill(); }
    if(st===0 || st===4){ for(const p of S.P) arrow(ctx,X(p.x),Y(p.y),X(p.x)+p.vx*ak,Y(p.y)-p.vy*ak,rgba('particle',.8),1.2); }
    // hover weights
    if(!S.playing && S.hover>=0 && S.P[S.hover]){ const p=S.P[S.hover]; const ws=weights(p); ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='middle';
      for(const q of ws){ ctx.strokeStyle=rgba('ink',.25+.6*q.w); ctx.lineWidth=1+5*q.w; ctx.beginPath(); ctx.moveTo(X(p.x),Y(p.y)); ctx.lineTo(X(q.i),Y(q.j)); ctx.stroke();
        ctx.fillStyle=COL.node; ctx.beginPath(); ctx.arc(X(q.i),Y(q.j),4+10*Math.sqrt(q.w),0,7); ctx.fill(); }
      ctx.strokeStyle=COL.ink; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(X(p.x),Y(p.y),Math.max(3,s*.18),0,7); ctx.stroke();
      $('#simHover').textContent=`particle at (${p.x.toFixed(2)}, ${p.y.toFixed(2)}) \u2192 corner weights ${ws.map(q=>(q.w*100).toFixed(0)+'%').join(' \u00B7 ')}  (sum 100%)`; }
    // rail + text + stats
    document.querySelectorAll('#rail div').forEach((d,k)=>{ const cur=S.playing?-1:S.stage; d.className = k===cur?'on':(k<cur?'done':''); });
    let act=0, nodes=0, cont=0; for(let c=0;c<NC;c++) act+=cAct[c]; for(let n=0;n<NN;n++){ if(m[n]>0) nodes++; if(isC[n]) cont++; }
    $('#simStats').innerHTML=`<span>step <b>${S.steps}</b></span><span>active cells <b>${act}</b></span><span>velocity nodes <b>${S.stage>=2||S.playing?nodes:'\u2013'}</b></span><span>contact nodes <b>${S.stage>=2||S.playing?cont:'\u2013'}</b></span><span>max compression after solve <b>${(S.stage>=3||S.playing)?S.residual.toFixed(3):'\u2013'}</b> /s</span>`;
  }
  function setText(){ const t=TEXT[S.playing?3:S.stage]; $('#stageText').innerHTML=`<h3>${S.playing?'Playing: all stages every frame':t.h}</h3>${S.playing?'<p>Each frame runs bin \u2192 P\u2192G \u2192 solve \u2192 G\u2192P. Violet cells show pressure from the solve. Lower friction to see the pile flatten. Set iterations to 0 to see what happens without the implicit solve.</p>':t.b}`; }
  let last=0, raf=0;
  function loop(t){ if(!S.playing) return; if(t-last>=1000/30){ last=t; full(); draw(); } raf=requestAnimationFrame(loop); }
  function init(){
    registerPlayback('bPlay', stop);
    S.cv=$('#simCv'); reset(); setText();
    $('#bNext').onclick=()=>{ stop(); next(); setText(); draw(); };
    $('#bStep').onclick=()=>{ stop(); full(); setText(); draw(); };
    $('#bReset').onclick=()=>{ stop(); reset(); setText(); draw(); $('#simHover').textContent='Hover a particle to see how its mass splits over its 4 grid corners.'; };
    $('#bPlay').onclick=()=>{ if(S.playing){ stop(); } else { S.playing=true; $('#bPlay').textContent='Pause'; setText(); raf=requestAnimationFrame(loop); } };
    $('#sMu').oninput=e=>{ S.mu=+e.target.value; $('#oMu').textContent=S.mu.toFixed(2); };
    $('#sIt').oninput=e=>{ S.iters=+e.target.value; $('#oIt').textContent=S.iters; };
    $('#cPress').onchange=draw;
    S.cv.addEventListener('pointermove', e=>{ const r=S.cv.getBoundingClientRect(); const s=r.width/NX; const x=(e.clientX-r.left)/s, y=NY-(e.clientY-r.top)/s; let best=-1, bd=0.4; S.P.forEach((p,k)=>{ const d=Math.hypot(p.x-x,p.y-y); if(d<bd){bd=d;best=k;} }); if(best!==S.hover){ S.hover=best; if(!S.playing) draw(); } });
    S.cv.addEventListener('pointerleave', ()=>{ S.hover=-1; if(!S.playing) draw(); });
  }
  function stop(){ if(S.playing){ S.playing=false; cancelAnimationFrame(raf); $('#bPlay').textContent='Play'; setText(); } }
  return {init, draw, traces};
})();

// =====================================================================
// B. Velocity basis lab
// =====================================================================
const vbHooks=[];
const VB = (() => {
  const L=6; const S={x:2.72,y:3.35,basis:'Q1',mode:'value',drag:false};
  const K={
    Q1:{off:0, w:t=>{const a=Math.abs(t); return a<1?1-a:0;}, d:t=>{const a=Math.abs(t); return a<1?-Math.sign(t):0;}},
    B2:{off:.5, w:t=>{const a=Math.abs(t); return a<.5?.75-a*a:a<1.5?.5*(1.5-a)**2:0;}, d:t=>{const a=Math.abs(t); return a<.5?-2*t:a<1.5?-(1.5-a)*Math.sign(t):0;}},
    B3:{off:0, w:t=>{const a=Math.abs(t); return a<1?.5*a*a*a-a*a+2/3:a<2?(2-a)**3/6:0;}, d:t=>{const a=Math.abs(t); return a<1?(1.5*a*a-2*a)*Math.sign(t):a<2?-.5*(2-a)**2*Math.sign(t):0;}},
  };
  const FACTS={
    Q1:{name:'Q1 \u00B7 trilinear (Lagrange)', touch:'2 nodes per axis: 4 in 2D, <b>8 in 3D</b>', smooth:'Value continuous. Gradient constant inside a cell, jumps at every cell face (C\u2070).', good:'Cheapest, tightest stencil, default. Interpolating: the field equals the node value at a node. <code>solver="auto"</code> \u2192 <code>gs</code>.', cost:'Cell-crossing noise: stress built from gradients jumps as particles move between cells.'},
    B2:{name:'B2 \u00B7 quadratic B-spline', touch:'3 nodes per axis: 9 in 2D, <b>27 in 3D</b>', smooth:'Gradient continuous (C\u00B9). Slopes change linearly as the particle moves.', good:'Smoother transfers and strain rates, much less cell-crossing noise. <code>solver="auto"</code> \u2192 <code>gs-batched</code>.', cost:'Wider coupling. Newton reserves 64 velocity nodes per active cell for B2/B3. Not interpolating, and a bit more smoothing.'},
    B3:{name:'B3 \u00B7 cubic B-spline', touch:'4 nodes per axis: 16 in 2D, <b>64 in 3D</b>', smooth:'C\u00B2: even the gradient\u2019s slope is continuous.', good:'Smoothest fields and gradients.', cost:'Widest stencil, most memory and solver coupling. Features blur over about 4 cells.'},
  };
  const nodes=b=>{ const o=K[b].off, out=[]; for(let k=0;k<=L;k++){ const v=k+o; if(v<=L+1e-9) out.push(v);} return out; };
  function active(){ const k=K[S.basis], ns=nodes(S.basis), list=[]; for(const xi of ns) for(const yj of ns){ const wx=k.w(S.x-xi), wy=k.w(S.y-yj); const w=wx*wy; if(w>0) list.push({xi,yj,w,gx:k.d(S.x-xi)*wy,gy:wx*k.d(S.y-yj)}); } return list; }
  function draw(){
    const cv=$('#vbCv'); const {ctx,w,h}=fit(cv,1); const s=w/L, X=x=>x*s, Y=y=>(L-y)*s;
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const ci=Math.floor(S.x), cj=Math.floor(S.y); ctx.fillStyle=rgba('particle',.07); ctx.fillRect(X(ci),Y(cj+1),s,s);
    ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=L;i++){ ctx.moveTo(X(i)+.5,0); ctx.lineTo(X(i)+.5,h); ctx.moveTo(0,Y(i)+.5); ctx.lineTo(w,Y(i)+.5);} ctx.stroke();
    const ns=nodes(S.basis); ctx.fillStyle=rgba('muted',.45); for(const a of ns) for(const b of ns){ ctx.beginPath(); ctx.arc(X(a),Y(b),2,0,7); ctx.fill(); }
    const list=active(); const px=X(S.x), py=Y(S.y);
    for(const q of list){ ctx.strokeStyle=rgba('ink',.08+.5*q.w); ctx.lineWidth=.5+5*q.w; ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(X(q.xi),Y(q.yj)); ctx.stroke(); }
    ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='middle';
    for(const q of list){ const x0=X(q.xi), y0=Y(q.yj);
      if(S.mode==='value'){ ctx.fillStyle=rgba('node',.85); ctx.beginPath(); ctx.arc(x0,y0,3+s*.28*Math.sqrt(q.w),0,7); ctx.fill(); if(q.w>=.03){ ctx.fillStyle=COL.ink; ctx.fillText((q.w*100).toFixed(0)+'%', x0, y0 - (3+s*.28*Math.sqrt(q.w)) - 8); } }
      else { ctx.fillStyle=rgba('node',.35); ctx.beginPath(); ctx.arc(x0,y0,3.5,0,7); ctx.fill(); arrow(ctx,x0,y0,x0+q.gx*s*.85,y0-q.gy*s*.85,COL.node,2); } }
    ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(px,py,8,0,7); ctx.fill(); ctx.strokeStyle=COL.surface; ctx.lineWidth=2; ctx.stroke();
    const sum=list.reduce((a,q)=>a+q.w,0); const sg=list.reduce((a,q)=>[a[0]+q.gx,a[1]+q.gy],[0,0]);
    $('#vbCap').textContent = S.mode==='value' ? `${list.length} nodes touched in 2D \u00B7 weights sum to ${(sum*100).toFixed(1)}% \u00B7 drag the particle or use arrow keys` : `arrows = \u2207w_i (how fast each weight changes) \u00B7 they sum to (${sg[0].toFixed(2)}, ${sg[1].toFixed(2)}), so a uniform velocity gives zero strain rate`;
    drawPlot();
    const f=FACTS[S.basis]; $('#vbFacts').innerHTML=`<h3>${f.name}</h3><dl><dt>Touches</dt><dd>${f.touch}</dd><dt>Smoothness</dt><dd>${f.smooth}</dd><dt>Good</dt><dd>${f.good}</dd><dt>Cost</dt><dd>${f.cost}</dd></dl>`;
    vbHooks.forEach(f=>f());
  }
  function drawPlot(){
    const cv=$('#vbPlot'); const {ctx,w,h}=fit(cv,.5); const k=K[S.basis], ns=nodes(S.basis); const g=S.mode==='grad';
    const pad=18, X=x=>pad+(x/L)*(w-2*pad), ymin=g?-1.15:-.05, ymax=g?1.15:1.08, Y=y=>8+(ymax-y)/(ymax-ymin)*(h-26);
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle=COL.grid; ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=L;i++){ ctx.moveTo(X(i)+.5,Y(ymax)); ctx.lineTo(X(i)+.5,Y(ymin)); } ctx.stroke();
    ctx.strokeStyle=COL.line; ctx.beginPath(); ctx.moveTo(X(0),Y(0)); ctx.lineTo(X(L),Y(0)); ctx.stroke();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top'; for(let i=0;i<=L;i++) ctx.fillText(String(i), X(i), h-15);
    for(const xi of ns){ const on=Math.abs(k.w(S.x-xi))>1e-6; ctx.strokeStyle=on?COL.node:rgba('muted',.35); ctx.lineWidth=on?2.2:1.2; ctx.beginPath(); let prev=null; for(let px=0;px<=w-2*pad;px+=1){ const x=px/(w-2*pad)*L; const v=g?k.d(x-xi):k.w(x-xi); const sx=X(x), sy=Y(v); if(prev===null || Math.abs(v-prev)>0.4) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy); prev=v; } ctx.stroke(); }
    ctx.strokeStyle=COL.particle; ctx.setLineDash([4,4]); ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(X(S.x),Y(ymax)); ctx.lineTo(X(S.x),Y(ymin)); ctx.stroke(); ctx.setLineDash([]);
    for(const xi of ns){ const v=g?k.d(S.x-xi):k.w(S.x-xi); if(Math.abs(k.w(S.x-xi))>1e-6){ ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(X(S.x),Y(v),4,0,7); ctx.fill(); } }
    $('#vbPlotCap').textContent = g ? 'dN_i/dx along x: Q1 is a step that flips at cell lines; B2 is continuous; B3 is smooth' : 'one bump per node along x; orange dots = this particle\u2019s weights';
  }
  function init(){
    seg('#vbBasis', v=>{S.basis=v; draw();}); seg('#vbMode', v=>{S.mode=v; draw();});
    const cv=$('#vbCv'); const toW=e=>{ const r=cv.getBoundingClientRect(); const s=r.width/L; return [(e.clientX-r.left)/s, L-(e.clientY-r.top)/s]; };
    const set=([x,y])=>{ S.x=Math.min(4.5,Math.max(1.5,x)); S.y=Math.min(4.5,Math.max(1.5,y)); draw(); };
    cv.addEventListener('pointerdown', e=>{ S.drag=true; cv.setPointerCapture(e.pointerId); set(toW(e)); });
    cv.addEventListener('pointermove', e=>{ if(S.drag) set(toW(e)); });
    cv.addEventListener('pointerup', ()=>{ S.drag=false; });
    cv.addEventListener('keydown', e=>{ const d=.05; const m={ArrowLeft:[-d,0],ArrowRight:[d,0],ArrowUp:[0,d],ArrowDown:[0,-d]}[e.key]; if(m){ e.preventDefault(); set([S.x+m[0], S.y+m[1]]); } });
  }
  return {init, draw, S, K};
})();

// =====================================================================
// B2. Velocity basis in 3D (three.js)
// =====================================================================
const V3 = (() => {
  const L=5, OFF=.5, MAXN=216, MAXA=64;
  let inited=false, ok=false, visible=false, R, scene, cam, mats={}, grey=[], act=[], arrows=[], links, linkPos, linkCol, cellBox, supBox, part, dragging=false, lx=0, ly=0, anim=0, t0=0;
  const sph={theta:.55, phi:1.12, r:12.5};
  const st={z:2.4};
  const clampP=v=>Math.min(3.98,Math.max(1.02,v));
  function setup(){
    if(inited) return ok; inited=true;
    if(!window.THREE){ $('#v3Msg').hidden=false; return false; }
    try{
      const cv=$('#v3Cv'); R=new THREE.WebGLRenderer({canvas:cv, antialias:true}); R.setPixelRatio(Math.min(2, window.devicePixelRatio||1));
      scene=new THREE.Scene(); cam=new THREE.PerspectiveCamera(36,1,.1,100);
      scene.add(new THREE.AmbientLight(0xffffff,.7)); const dl=new THREE.DirectionalLight(0xffffff,.5); dl.position.set(6,10,8); scene.add(dl);
      mats.grey=new THREE.MeshLambertMaterial(); mats.node=new THREE.MeshLambertMaterial(); mats.part=new THREE.MeshLambertMaterial();
      mats.grid=new THREE.LineBasicMaterial({transparent:true, opacity:.5}); mats.cell=new THREE.LineBasicMaterial(); mats.sup=new THREE.LineBasicMaterial({transparent:true, opacity:.55}); mats.link=new THREE.LineBasicMaterial({vertexColors:true});
      const gp=[]; for(let a=0;a<=L;a++) for(let b=0;b<=L;b++){ gp.push(0,a,b, L,a,b, a,0,b, a,L,b, a,b,0, a,b,L); }
      const gg=new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.Float32BufferAttribute(gp,3)); scene.add(new THREE.LineSegments(gg, mats.grid));
      const sg=new THREE.SphereGeometry(1,18,12);
      for(let i=0;i<MAXN;i++){ const m=new THREE.Mesh(sg, mats.grey); m.visible=false; scene.add(m); grey.push(m); }
      for(let i=0;i<MAXA;i++){ const m=new THREE.Mesh(sg, mats.node); m.visible=false; scene.add(m); act.push(m);
        const a=new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(), 1, 0x08798a, .16, .1); a.visible=false; scene.add(a); arrows.push(a); }
      linkPos=new Float32Array(MAXA*6); linkCol=new Float32Array(MAXA*6); const lg=new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(linkPos,3)); lg.setAttribute('color', new THREE.BufferAttribute(linkCol,3)); links=new THREE.LineSegments(lg, mats.link); scene.add(links);
      const eg=new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)); cellBox=new THREE.LineSegments(eg, mats.cell); scene.add(cellBox); supBox=new THREE.LineSegments(eg, mats.sup); scene.add(supBox);
      part=new THREE.Mesh(sg, mats.part); part.scale.setScalar(.12); scene.add(part);
      cv.addEventListener('pointerdown', e=>{ dragging=true; lx=e.clientX; ly=e.clientY; cv.setPointerCapture(e.pointerId); });
      cv.addEventListener('pointermove', e=>{ if(!dragging) return; sph.theta-=(e.clientX-lx)*.008; sph.phi=Math.min(2.8,Math.max(.25,sph.phi-(e.clientY-ly)*.008)); lx=e.clientX; ly=e.clientY; render(); });
      cv.addEventListener('pointerup', ()=>{ dragging=false; });
      ok=true;
    }catch(err){ $('#v3Msg').hidden=false; ok=false; }
    return ok;
  }
  function size(){ const cv=$('#v3Cv'); const w=Math.max(220, cv.clientWidth || cv.parentElement.clientWidth); const h=Math.round(Math.min(w*.85, 620)); R.setSize(w,h,false); cv.style.height=h+'px'; cam.aspect=w/h; cam.updateProjectionMatrix(); }
  function render(){ if(!ok) return; const c=L/2; cam.position.set(c+sph.r*Math.sin(sph.phi)*Math.sin(sph.theta), c+sph.r*Math.cos(sph.phi), c+sph.r*Math.sin(sph.phi)*Math.cos(sph.theta)); cam.lookAt(c,c,c); R.render(scene,cam); }
  function update(){
    if(!visible || !ok) return;
    const S=VB.S, k=VB.K[S.basis], g=S.mode==='grad';
    const x=S.x-OFF, y=S.y-OFF, z=st.z;
    for(const [id,v] of [['x',x],['y',y],['z',z]]){ $('#v3'+id).value=v.toFixed(2); $('#o3'+id).textContent=v.toFixed(2); }
    scene.background=new THREE.Color(COL.surface); mats.grey.color.set(COL.muted); mats.node.color.set(COL.node); mats.part.color.set(COL.particle); mats.grid.color.set(COL.line); mats.cell.color.set(COL.particle); mats.sup.color.set(COL.node);
    const ns=[]; for(let i=0;i<=L;i++){ const v=i+k.off; if(v<=L+1e-9) ns.push(v); }
    let gi=0; const list=[];
    for(const a of ns) for(const b of ns) for(const c of ns){
      const wx=k.w(x-a), wy=k.w(y-b), wz=k.w(z-c), w=wx*wy*wz;
      if(w>0) list.push({a,b,c,w,g:[k.d(x-a)*wy*wz, wx*k.d(y-b)*wz, wx*wy*k.d(z-c)]});
      else if(gi<MAXN){ const m=grey[gi++]; m.visible=true; m.position.set(a,b,c); m.scale.setScalar(.035); } }
    for(;gi<MAXN;gi++) grey[gi].visible=false;
    const bg=COL.surfaceRGB, ink=COL.inkRGB; let mn=[9,9,9], mx=[-9,-9,-9];
    list.forEach((q,i)=>{ const m=act[i]; m.visible=true; m.position.set(q.a,q.b,q.c); m.scale.setScalar(g ? .07 : .05+.3*Math.sqrt(q.w));
      const t=Math.min(1,.15+q.w*2.6); const col=[0,1,2].map(j=>(bg[j]+(ink[j]-bg[j])*t)/255);
      linkPos.set([x,y,z,q.a,q.b,q.c], i*6); linkCol.set([...col,...col], i*6);
      const ar=arrows[i]; if(g){ const len=Math.hypot(q.g[0],q.g[1],q.g[2]); if(len>1e-4){ ar.visible=true; ar.position.set(q.a,q.b,q.c); ar.setDirection(new THREE.Vector3(q.g[0]/len,q.g[1]/len,q.g[2]/len)); const l2=Math.min(1.4,len*1.15); ar.setLength(l2, Math.min(.18,l2*.4), Math.min(.11,l2*.25)); ar.setColor(new THREE.Color(COL.node)); } else ar.visible=false; } else ar.visible=false;
      mn=[Math.min(mn[0],q.a),Math.min(mn[1],q.b),Math.min(mn[2],q.c)]; mx=[Math.max(mx[0],q.a),Math.max(mx[1],q.b),Math.max(mx[2],q.c)]; });
    for(let i=list.length;i<MAXA;i++){ act[i].visible=false; arrows[i].visible=false; }
    links.geometry.setDrawRange(0, list.length*2); links.geometry.attributes.position.needsUpdate=true; links.geometry.attributes.color.needsUpdate=true;
    cellBox.position.set(Math.floor(x)+.5, Math.floor(y)+.5, Math.floor(z)+.5);
    supBox.position.set((mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2); supBox.scale.set(Math.max(.01,mx[0]-mn[0]),Math.max(.01,mx[1]-mn[1]),Math.max(.01,mx[2]-mn[2]));
    part.position.set(x,y,z);
    const ax=['a','b','c'].map(key=>new Set(list.map(q=>q[key])).size).join('\u00D7'), sum=list.reduce((s,q)=>s+q.w,0), top=list.slice().sort((p,q)=>q.w-p.w).slice(0,4).map(q=>(q.w*100).toFixed(1)+'%').join(', ');
    $('#vbCap').textContent = g ? `3D: ${list.length} nodes, each with a 3D gradient arrow \u2207w_i. Drag to orbit. Watch the arrows flip direction for Q1 as the particle crosses a cell face.`
      : `3D: ${list.length} nodes touched (${ax}) \u00B7 weights sum to ${(sum*100).toFixed(1)}% \u00B7 largest ${top} \u00B7 orange box = particle's cell, teal box = everything it touches. Drag to orbit.`;
    render();
  }
  async function show(on){
    visible=on; $('#vbCv').hidden=on; $('#v3Wrap').hidden=!on;
    if(on){
      await loadThree();
      if(visible && setup()) size();
    } else stopAnim();
    VB.draw();
  }
  function stopAnim(){ if(anim){ cancelAnimationFrame(anim); anim=0; $('#v3Anim').textContent='Move the particle'; } }
  function step(ts){ if(!t0) t0=ts; const t=(ts-t0)/1000*.55; VB.S.x=clampP(2.5+1.35*Math.sin(t*.9))+OFF; VB.S.y=clampP(2.5+1.3*Math.sin(t*.67+1.1))+OFF; st.z=clampP(2.5+1.3*Math.sin(t*.51+2.3)); VB.draw(); anim=requestAnimationFrame(step); }
  function init(){
    registerPlayback('v3Anim', stopAnim);
    vbHooks.push(update);
    seg('#vbView', v=>show(v==='3d'));
    $('#v3x').oninput=e=>{ VB.S.x=clampP(+e.target.value)+OFF; VB.draw(); };
    $('#v3y').oninput=e=>{ VB.S.y=clampP(+e.target.value)+OFF; VB.draw(); };
    $('#v3z').oninput=e=>{ st.z=clampP(+e.target.value); VB.draw(); };
    $('#v3Anim').onclick=()=>{ if(anim) stopAnim(); else { t0=0; $('#v3Anim').textContent='Stop'; anim=requestAnimationFrame(step); } };
    $('#v3Reset').onclick=()=>{ sph.theta=.55; sph.phi=1.12; render(); };
  }
  function resize(){ if(visible && ok){ size(); update(); } }
  return {init, resize};
})();

// =====================================================================
// B3. One node's bump as a 3D surface (three.js)
// =====================================================================
const VS = (() => {
  const RES=120, WRES=18, HALF=2.25;
  let inited=false, ok=false, visible=false, R, scene, cam, geo, wgeo, mesh, wire, floor, nodeMk, partMk, partLine, dragging=false, lx=0, ly=0;
  const sph={theta:.75, phi:1.0, r:8.5};
  function setup(){
    if(inited) return ok; inited=true;
    if(!window.THREE){ $('#vbSurfMsg').hidden=false; return false; }
    try{
      const cv=$('#vbSurf'); R=new THREE.WebGLRenderer({canvas:cv, antialias:true}); R.setPixelRatio(Math.min(2, window.devicePixelRatio||1));
      scene=new THREE.Scene(); cam=new THREE.PerspectiveCamera(35,1,.1,100);
      scene.add(new THREE.AmbientLight(0xffffff,.62)); const dl=new THREE.DirectionalLight(0xffffff,.62); dl.position.set(4,8,6); scene.add(dl);
      geo=new THREE.PlaneGeometry(2*HALF,2*HALF,RES,RES); geo.rotateX(-Math.PI/2);
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count*3),3));
      mesh=new THREE.Mesh(geo, new THREE.MeshLambertMaterial({vertexColors:true, side:THREE.DoubleSide})); scene.add(mesh);
      wgeo=new THREE.PlaneGeometry(2*HALF,2*HALF,WRES,WRES); wgeo.rotateX(-Math.PI/2);
      wire=new THREE.Mesh(wgeo, new THREE.MeshBasicMaterial({wireframe:true, transparent:true, opacity:.22})); scene.add(wire);
      floor=new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({transparent:true, opacity:.7})); scene.add(floor);
      const sg=new THREE.SphereGeometry(1,16,12);
      nodeMk=new THREE.Mesh(sg, new THREE.MeshLambertMaterial()); nodeMk.scale.setScalar(.07); scene.add(nodeMk);
      partMk=new THREE.Mesh(sg, new THREE.MeshLambertMaterial()); partMk.scale.setScalar(.08); scene.add(partMk);
      partLine=new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial()); scene.add(partLine);
      cv.addEventListener('pointerdown', e=>{ dragging=true; lx=e.clientX; ly=e.clientY; cv.setPointerCapture(e.pointerId); });
      cv.addEventListener('pointermove', e=>{ if(!dragging) return; sph.theta-=(e.clientX-lx)*.008; sph.phi=Math.min(1.55,Math.max(.2,sph.phi-(e.clientY-ly)*.008)); lx=e.clientX; ly=e.clientY; render(); });
      cv.addEventListener('pointerup', ()=>{ dragging=false; });
      ok=true;
    }catch(err){ $('#vbSurfMsg').hidden=false; ok=false; }
    return ok;
  }
  function size(){ const cv=$('#vbSurf'); const w=Math.max(220, cv.clientWidth || cv.parentElement.clientWidth); const h=Math.round(w*.78); R.setSize(w,h,false); cv.style.height=h+'px'; cam.aspect=w/h; cam.updateProjectionMatrix(); }
  let target=.4;
  function render(){ if(!ok) return; cam.position.set(sph.r*Math.sin(sph.phi)*Math.sin(sph.theta), target+sph.r*Math.cos(sph.phi), sph.r*Math.sin(sph.phi)*Math.cos(sph.theta)); cam.lookAt(0,target,0); R.render(scene,cam); }
  function update(){
    if(!visible || !ok) return;
    const S=VB.S, k=VB.K[S.basis], g=S.mode==='grad';
    // strongest node for this particle (2D slice, same as the lab on the left)
    let best=null; for(let i=0;i<=6;i++) for(let j=0;j<=6;j++){ const xi=i+k.off, yj=j+k.off; if(xi>6||yj>6) continue; const w=k.w(S.x-xi)*k.w(S.y-yj); if(!best||w>best.w) best={xi,yj,w}; }
    const cx=best.xi, cy=best.yj;
    const f=(x,y)=> g ? k.d(x-cx)*k.w(y-cy) : k.w(x-cx)*k.w(y-cy);
    const hs = g ? 1.1 : 2.4; target = g ? 0 : .45;
    const pos=geo.attributes.position, col=geo.attributes.color; let mx=1e-9;
    for(let i=0;i<pos.count;i++){ const v=f(cx+pos.getX(i), cy-pos.getZ(i)); mx=Math.max(mx,Math.abs(v)); }
    for(let i=0;i<pos.count;i++){ const v=f(cx+pos.getX(i), cy-pos.getZ(i)); pos.setY(i, hs*v);
      const c=mix('surface', v>=0?'node':'particle', Math.min(1,.12+.88*Math.abs(v)/mx)); col.setXYZ(i,c[0]/255,c[1]/255,c[2]/255); }
    pos.needsUpdate=true; col.needsUpdate=true; geo.computeVertexNormals();
    const wp_=wgeo.attributes.position; for(let i=0;i<wp_.count;i++) wp_.setY(i, hs*f(cx+wp_.getX(i), cy-wp_.getZ(i))+.004); wp_.needsUpdate=true;
    // floor: cell lines in the shown square
    const L=[]; for(let v=Math.ceil(cx-HALF); v<=cx+HALF; v++){ L.push(v-cx,-.002,-HALF, v-cx,-.002,HALF); } for(let v=Math.ceil(cy-HALF); v<=cy+HALF; v++){ L.push(-HALF,-.002,-(v-cy), HALF,-.002,-(v-cy)); }
    floor.geometry.dispose(); floor.geometry=new THREE.BufferGeometry(); floor.geometry.setAttribute('position', new THREE.Float32BufferAttribute(L,3));
    const X=S.x-cx, Z=-(S.y-cy), val=f(S.x,S.y), hp=hs*val;
    partLine.geometry.dispose(); partLine.geometry=new THREE.BufferGeometry(); partLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute([X,0,Z, X,hp,Z],3));
    partMk.position.set(X,hp,Z); nodeMk.position.set(0,0,0);
    scene.background=new THREE.Color(COL.surface); wire.material.color.set(COL.ink); floor.material.color.set(COL.muted);
    nodeMk.material.color.set(COL.ink); partMk.material.color.set(COL.particle); partLine.material.color.set(COL.particle);
    const where=`node at (${cx}, ${cy})`;
    $('#vbPlotCap').textContent = g
      ? `Slope in x (\u2202w/\u2202x) of the bump of the ${where}. Teal = positive, orange = negative. Q1: vertical cliffs at the cell lines, where the slope jumps (the cause of cell-crossing noise). B2: no cliffs, only creases. B3: smooth everywhere. Orange stick = this particle: ${val.toFixed(3)}. Drag to orbit.`
      : `Bump of the ${where} (the node this particle weights most). Height = weight. Orange stick = this particle: ${(val*100).toFixed(1)}%. Q1: a tent with straight edges. B2/B3: a smooth hill, lower and wider. Drag to orbit.`;
    render();
  }
  async function show(on){
    visible=on; $('#vbPlot').hidden=on; $('#vbSurf').hidden=!on;
    if(on){
      await loadThree();
      if(visible && setup()) size();
    }
    VB.draw();
  }
  function init(){ vbHooks.push(update); seg('#vbPlotView', v=>show(v==='3d')); }
  function resize(){ if(visible && ok){ size(); update(); } }
  return {init, resize};
})();

// =====================================================================
// C. Strain basis lab
// =====================================================================
const SB = (() => {
  const NX=6, NY=4; const S={basis:'P0', field:'linear', view:'fit', cmp:'none', ys:1.35};
  const BASES=['P0','P1d','Q1','Q1d','pic'], NAME={P0:'P0',P1d:'P1d',Q1:'Q1',Q1d:'Q1d',pic:'pic27'};
  const top=x=>2.75+.45*Math.sin(1.3*x+.4)-.1*x;
  const inside=(x,y)=> y>0.02 && y<top(x) && x>0.15+0.55*Math.max(0,y-1.4) && x<5.9;
  const r=rng(5); const P=[]; for(let i=0;i<NX*3;i++) for(let j=0;j<NY*3;j++){ const x=(i+.5)/3+(r()-.5)*.16, y=(j+.5)/3+(r()-.5)*.16; if(inside(x,y)) P.push({x,y}); }
  const cellOf=p=>[Math.min(NX-1,Math.floor(p.x)), Math.min(NY-1,Math.floor(p.y))];
  const byCell=Array.from({length:NX*NY},()=>[]); P.forEach(p=>{ const [i,j]=cellOf(p); byCell[j*NX+i].push(p); });
  const FIELDS={ uniform:()=>.6, linear:(x,y)=>.95-.6*y, twist:(x,y)=>.3*(x-3)*(y-1.5), curved:(x,y)=>.85*Math.cos(1.5*(x-1))+.1*y, band:(x,y)=>Math.exp(-(((x-.8*y-1.5)/.3)**2)), rot:()=>0 };
  const NOTES={
    uniform:'<b>Uniform:</b> the same strain rate everywhere. Every basis holds a constant, so every error is 0.',
    linear:'<b>Linear:</b> strain rate changes steadily with height, like pressure growing with depth in a pile. P0 can only make a staircase. P1d, Q1 and Q1d all contain straight slopes, so all three are exact and look identical.',
    twist:'<b>Twist:</b> strain rate varies like x\u00B7y (a saddle). P1d can only tilt, so it misses a little of the twist in every cell (switch the right panel to <b>error</b>). Q1 and Q1d contain the x\u00B7y term and are both exact.',
    curved:'<b>Curved:</b> the field bends inside cells, and no grid basis contains a bend. Q1d fits each cell on its own, so its pieces jump slightly at cell lines. Q1 must agree at shared corners, so it stays continuous but fits worse (compare their bars). In the slice, pick Q1d and compare with Q1.',
    band:'<b>Shear band:</b> a thin band of sliding, narrower than one cell. Grid bases smear it over whole cells, and Q1 also leaks it into neighboring cells. pic follows it at the particles.',
    rot:'<b>Rotation:</b> the velocity gradient is not zero, but its symmetric part is. Every basis shows zero strain rate and zero stress. Newton uses the rotation only to turn each particle\u2019s stored elastic strain.',
  };
  const FACTS={
    P0:{name:'P0 \u00B7 one constant stress per cell', modes:'1 per cell (\u00D76 tensor components in 3D)', what:'One push-back value per cell. Default, cheapest and most robust. Pairs stably with Q1 velocity.', miss:'Anything that varies inside a cell becomes a staircase.'},
    P1d:{name:'P1d \u00B7 linear per cell, discontinuous', modes:'3 per cell in 2D, <b>4 in 3D</b> (constant + x, y, z slopes)', what:'Each cell can tilt its stress, so gradients like hydrostatic pressure are exact.', miss:'Twist and curvature inside a cell. Newton orthogonalizes the modes against the particle mass in each cell, and nearly empty cells drop modes they can\u2019t support ("reduced" below).'},
    Q1:{name:'Q1 \u00B7 continuous trilinear', modes:'1 per grid corner, shared by the cells around it', what:'Tilt and twist, and no jumps between cells. The purple squares are its corner values.', miss:'Sharp features smear into neighbors, because neighbors must agree at shared corners. Unknowns couple cells, so the solve is less local.'},
    Q1d:{name:'Q1d \u00B7 trilinear per cell, discontinuous', modes:'4 per cell in 2D, <b>8 in 3D</b>, at Gauss\u2013Legendre points', what:'Same shapes as Q1 inside a cell, but each cell is fitted on its own and may jump at borders.', miss:'The most strain unknowns per velocity node, so the highest locking risk with Q1 velocity. Partially filled cells often lose modes.'},
    pic:{name:'pic27 \u00B7 one stress value per particle', modes:'1 per particle, at most N per cell. This demo has 3\u00D73 = 9 particles per cell, the 2D version of 27 in 3D.', what:'Stress unknowns sit exactly at the particles, so detail follows the material. The colored patches only show the nearest particle. In Newton the values exist only at the particles.', miss:'Unknown count grows with particle count, which can over-constrain the velocity field. Choose N at least as large as your particles per cell.'},
  };
  const PB={ P0:[()=>1], P1d:[()=>1,(dx)=>dx,(dx,dy)=>dy], Q1d:[()=>1,(dx)=>dx,(dx,dy)=>dy,(dx,dy)=>dx*dy] };
  const activeCell=(x,y)=>{ const i=Math.max(0,Math.min(NX-1,Math.floor(x))), j=Math.max(0,Math.min(NY-1,Math.floor(y))); return [i,j,j*NX+i]; };
  const q1Nodes=()=>{ let n=0; for(let i=0;i<=NX;i++) for(let j=0;j<=NY;j++){ if([[i-1,j-1],[i,j-1],[i-1,j],[i,j]].some(([a,b])=>a>=0&&b>=0&&a<NX&&b<NY&&byCell[b*NX+a].length)) n++; } return n; };
  function fitBasis(b){
    if(b==='pic') return {modes:P.length, reduced:0, ev:(x,y)=>{ const [,,c]=activeCell(x,y); if(!byCell[c].length) return null; let best=null, bd=1e9; for(const p of byCell[c]){ const d=(p.x-x)**2+(p.y-y)**2; if(d<bd){bd=d;best=p;} } return best.f; }};
    if(b==='Q1'){ const W=NX+1, n=W*(NY+1); const A=Array.from({length:n},()=>new Array(n).fill(0)), rhs=new Array(n).fill(0);
      for(const p of P){ const [i,j]=cellOf(p); const fx=p.x-i, fy=p.y-j; const ids=[j*W+i,j*W+i+1,(j+1)*W+i,(j+1)*W+i+1], ws=[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy];
        for(let a=0;a<4;a++){ rhs[ids[a]]+=ws[a]*p.f; for(let c2=0;c2<4;c2++) A[ids[a]][ids[c2]]+=ws[a]*ws[c2]; } }
      for(let k=0;k<n;k++) A[k][k]+= A[k][k]>0 ? 1e-6 : 1; const q=solveDense(A,rhs,n);
      return {modes:q1Nodes(), reduced:0, ev:(x,y)=>{ const [i,j,c]=activeCell(x,y); if(!byCell[c].length) return null; const fx=x-i, fy=y-j; return q[j*W+i]*(1-fx)*(1-fy)+q[j*W+i+1]*fx*(1-fy)+q[(j+1)*W+i]*(1-fx)*fy+q[(j+1)*W+i+1]*fx*fy; }}; }
    const cells={}; let reduced=0, modes=0;
    for(let c=0;c<NX*NY;c++){ const ps=byCell[c]; if(!ps.length) continue; const i=c%NX, j=(c/NX)|0; let fns=PB[b], coef=null;
      while(!coef){ const n=fns.length; if(ps.length>=n){ const A=Array.from({length:n},()=>new Array(n).fill(0)), rhs=new Array(n).fill(0);
          for(const p of ps){ const dx=p.x-i-.5, dy=p.y-j-.5; const v=fns.map(fn=>fn(dx,dy)); for(let a=0;a<n;a++){ rhs[a]+=v[a]*p.f; for(let bb=0;bb<n;bb++) A[a][bb]+=v[a]*v[bb]; } }
          coef=solveDense(A,rhs,n);
          if(!coef){ reduced++; let tr=0; for(let a=0;a<n;a++) tr+=A[a][a]; for(let a=0;a<n;a++) A[a][a]+=1e-7*tr+1e-12; coef=solveDense(A,rhs,n); } }
        else { reduced++; const A=Array.from({length:n},()=>new Array(n).fill(0)), rhs=new Array(n).fill(0);
          for(const p of ps){ const dx=p.x-i-.5, dy=p.y-j-.5; const v=fns.map(fn=>fn(dx,dy)); for(let a=0;a<n;a++){ rhs[a]+=v[a]*p.f; for(let bb=0;bb<n;bb++) A[a][bb]+=v[a]*v[bb]; } }
          let tr=0; for(let a=0;a<n;a++) tr+=A[a][a]; for(let a=0;a<n;a++) A[a][a]+=1e-7*tr+1e-12; coef=solveDense(A,rhs,n); }
        if(!coef){ fns=PB.P0; coef=[ps.reduce((a,p)=>a+p.f,0)/ps.length]; } }
      cells[c]={fns,coef}; modes+=fns.length; }
    return {modes, reduced, ev:(x,y)=>{ const [i,j,c]=activeCell(x,y); const e=cells[c]; if(!e) return null; const dx=x-i-.5, dy=y-j-.5; return e.fns.reduce((a,fn,k)=>a+fn(dx,dy)*e.coef[k],0); }};
  }
  function color(v){ const t=Math.min(1,Math.abs(v)); const c=mix('surface', v>=0?'strain':'node', t*.92); return `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`; }
  function paint(cv, fn, mask){ const {ctx,w,h}=fit(cv, NY/NX); const s=w/NX, X=x=>x*s, Y=y=>(NY-y)*s; ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const B=3; for(let py=0;py<h;py+=B) for(let px=0;px<w;px+=B){ const x=(px+B/2)/s, y=NY-(py+B/2)/s; if(mask && !mask(x,y)) continue; const v=fn(x,y); if(v===null) continue; ctx.fillStyle=color(v); ctx.fillRect(px,py,B,B); }
    ctx.strokeStyle=rgba('ink',.18); ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=NX;i++){ ctx.moveTo(X(i)+.5,0); ctx.lineTo(X(i)+.5,h);} for(let j=0;j<=NY;j++){ ctx.moveTo(0,Y(j)+.5); ctx.lineTo(w,Y(j)+.5);} ctx.stroke();
    ctx.fillStyle=rgba('ink',.55); for(const p of P){ ctx.beginPath(); ctx.arc(X(p.x),Y(p.y),1.6,0,7); ctx.fill(); }
    ctx.strokeStyle=COL.ink; ctx.setLineDash([6,4]); ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(0,Y(S.ys)); ctx.lineTo(w,Y(S.ys)); ctx.stroke(); ctx.setLineDash([]);
    return {ctx,s,X,Y}; }
  function drawProfile(f, fits){
    const cv=$('#sbProf'); const {ctx,w,h}=fit(cv,.5); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const pad=10, X=x=>pad+x/NX*(w-2*pad); const xs=[]; for(let px=0;px<=w-2*pad;px++) xs.push(px/(w-2*pad)*NX);
    const series=[]; const tv=xs.map(x=>inside(x,S.ys)?f(x,S.ys):null); series.push(tv);
    const sv=xs.map(x=>fits[S.basis].ev(x,S.ys)); const cvv=S.cmp!=='none'&&S.cmp!==S.basis?xs.map(x=>fits[S.cmp].ev(x,S.ys)):null;
    let lo=-.2, hi=.2; for(const arr of [tv,sv,cvv]) if(arr) for(const v of arr) if(v!==null){ lo=Math.min(lo,v); hi=Math.max(hi,v); }
    const m=(hi-lo)*.12; lo-=m; hi+=m; const Y=v=>8+(hi-v)/(hi-lo)*(h-24);
    ctx.strokeStyle=COL.grid; ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=NX;i++){ ctx.moveTo(X(i)+.5,4); ctx.lineTo(X(i)+.5,h-14); } ctx.stroke();
    ctx.strokeStyle=COL.line; ctx.beginPath(); ctx.moveTo(pad,Y(0)); ctx.lineTo(w-pad,Y(0)); ctx.stroke();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top'; for(let i=0;i<=NX;i++) ctx.fillText(String(i),X(i),h-13);
    const line=(arr,color,lw,dash)=>{ ctx.strokeStyle=color; ctx.lineWidth=lw; ctx.setLineDash(dash||[]); ctx.beginPath(); let pen=false, pc=-1;
      arr.forEach((v,k)=>{ const c=Math.floor(xs[k]); if(v===null){ pen=false; return; } if(!pen || c!==pc) ctx.moveTo(X(xs[k]),Y(v)); else ctx.lineTo(X(xs[k]),Y(v)); pen=true; pc=c; }); ctx.stroke(); ctx.setLineDash([]); };
    if(cvv) line(cvv, COL.node, 2);
    line(sv, COL.strain, 2.6);
    line(tv, COL.ink, 1.5, [5,4]);
    $('#sbProfSel').textContent=NAME[S.basis];
  }
  function drawErr(errs){
    const cv=$('#sbErr'); const {ctx,w,h}=fit(cv,.5); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const mx=Math.max(.05,...BASES.map(b=>errs[b])); const bw=(w-20)/BASES.length; const base=h-20, top=18;
    ctx.font=FONT_MONO; ctx.textAlign='center';
    BASES.forEach((b,i)=>{ const x=10+bw*i+bw/2, bh=(base-top)*errs[b]/mx; const sel=b===S.basis;
      ctx.fillStyle=sel?COL.strain:rgba('muted',.45); ctx.fillRect(x-bw*.3, base-bh, bw*.6, Math.max(1,bh));
      ctx.fillStyle=sel?COL.ink:COL.muted; ctx.textBaseline='bottom'; ctx.fillText(errs[b]<5e-4?'0':errs[b].toFixed(3), x, base-bh-3);
      ctx.textBaseline='top'; ctx.fillText(NAME[b], x, base+4); });
    ctx.strokeStyle=COL.line; ctx.beginPath(); ctx.moveTo(6,base+.5); ctx.lineTo(w-6,base+.5); ctx.stroke();
  }
  function draw(){
    const f=FIELDS[S.field]; P.forEach(p=>p.f=f(p.x,p.y));
    const fits={}, errs={}; for(const b of BASES){ fits[b]=fitBasis(b); let se=0; P.forEach(p=>{ const v=fits[b].ev(p.x,p.y); se+=(v-p.f)**2; }); errs[b]=Math.sqrt(se/P.length); }
    const F=fits[S.basis];
    paint($('#sbTrue'), (x,y)=>f(x,y), inside);
    const g = S.view==='err' ? paint($('#sbFit'), (x,y)=>{ const v=F.ev(x,y); return v===null?null:5*(v-f(x,y)); }, inside) : paint($('#sbFit'), F.ev, null);
    if(S.basis==='Q1' && S.view==='fit'){ g.ctx.fillStyle=COL.strain; for(let i=0;i<=NX;i++) for(let j=0;j<=NY;j++){ const touches=[[i-1,j-1],[i,j-1],[i-1,j],[i,j]].some(([a,b])=>a>=0&&b>=0&&a<NX&&b<NY&&byCell[b*NX+a].length); if(touches) g.ctx.fillRect(g.X(i)-3,g.Y(j)-3,6,6); } }
    $('#sbFitCap').textContent = S.view==='err' ? `error of ${NAME[S.basis]} (fit \u2212 true, \u00D75 so small errors show). White = exact.` : `what ${NAME[S.basis]} can hold`;
    $('#sbNote').innerHTML=NOTES[S.field];
    drawProfile(f, fits); drawErr(errs);
    const act=byCell.filter(c=>c.length).length;
    $('#sbStats').innerHTML=`<span>active cells <b>${act}</b></span><span>particles <b>${P.length}</b></span><span>stress values (${NAME[S.basis]}) <b>${F.modes}</b></span>${(S.basis==='P1d'||S.basis==='Q1d')?`<span>cells with too few particles for every mode <b>${F.reduced}</b></span>`:''}<span>fit error (RMS) <b>${errs[S.basis].toFixed(3)}</b></span>`;
    const D=FACTS[S.basis]; $('#sbFacts').innerHTML=`<h3>${D.name}</h3><dl><dt>Values</dt><dd>${D.modes}</dd><dt>Captures</dt><dd>${D.what}</dd><dt>Misses</dt><dd>${D.miss}</dd></dl>`;
  }
  function init(){
    seg('#sbBasis', v=>{S.basis=v; draw();}); seg('#sbField', v=>{S.field=v; draw();}); seg('#sbView', v=>{S.view=v; draw();}); seg('#sbCmp', v=>{S.cmp=v; draw();});
    $('#sbY').oninput=e=>{ S.ys=+e.target.value; $('#oSbY').textContent=S.ys.toFixed(2); draw(); };
  }
  return {init, draw};
})();

// =====================================================================
// D. Collider basis lab
// =====================================================================
const CB = (() => {
  const NX=6, NY=4, h=1, rp=.25; const S={basis:'S2', cx:3.05, cy:.95, R:1.15, drag:false};
  const FACTS={
    Q1:{name:'Q1 \u00B7 cell corners', pts:'4 per cell in 2D, <b>8 in 3D</b>, the same nodes as Q1 velocity', how:'The wall is the bilinear blend of distances at corners. Curved walls look faceted. Material can stop short of the wall or catch on a corner that pokes inside.'},
    S2:{name:'S2 \u00B7 quadratic serendipity', pts:'Corners + edge midpoints: 8 per cell in 2D, <b>20 in 3D</b>', how:'Quadratic along each edge, so curved walls are followed much more closely at modest cost. This is the default. With Q1 velocity, a midpoint\u2019s velocity is the average of its two corners, so one midpoint constraint links two velocity nodes.'},
    pic:{name:'pic8 \u00B7 particle centers', pts:'One per particle, at most 8 per cell', how:'Contact uses the exact collider distance at each particle, and velocity is read there with the velocity-basis weights. Contacts activate within one voxel of the particle surface and stop centers at max(radius, 0.25 voxel). Options: <code>collider_recovery_time</code> pushes out penetration, and contact shielding skips particles hidden behind the first layer.'},
  };
  let P=[];
  function genParticles(){ const r=rng(3); P=[]; for(let i=0;i<NX*2;i++) for(let j=0;j<NY*2;j++){ const x=(i+.5)/2+(r()-.5)*.08, y=(j+.5)/2+(r()-.5)*.08; if(y<3.3 && Math.hypot(x-S.cx,y-S.cy)>S.R+rp) P.push({x,y}); } }
  const sd=(x,y)=>Math.hypot(x-S.cx,y-S.cy)-S.R;
  function activeCells(){ const set=new Set(); for(const p of P) set.add(Math.min(NY-1,Math.floor(p.y))*NX+Math.min(NX-1,Math.floor(p.x))); return [...set]; }
  function interp(c,x,y){ const i=c%NX, j=(c/NX)|0; const xi=2*(x-i)-1, eta=2*(y-j)-1;
    if(S.basis==='Q1'){ const fx=x-i, fy=y-j; return sd(i,j)*(1-fx)*(1-fy)+sd(i+1,j)*fx*(1-fy)+sd(i,j+1)*(1-fx)*fy+sd(i+1,j+1)*fx*fy; }
    let v=0; for(const [a,b] of [[-1,-1],[1,-1],[-1,1],[1,1]]) v+=.25*(1+xi*a)*(1+eta*b)*(xi*a+eta*b-1)*sd(i+(a+1)/2,j+(b+1)/2);
    for(const b of [-1,1]) v+=.5*(1-xi*xi)*(1+eta*b)*sd(i+.5,j+(b+1)/2);
    for(const a of [-1,1]) v+=.5*(1+xi*a)*(1-eta*eta)*sd(i+(a+1)/2,j+.5);
    return v; }
  function draw(){
    genParticles(); const cv=$('#cbCv'); const {ctx,w,h:H}=fit(cv,NY/NX); const s=w/NX, X=x=>x*s, Y=y=>(NY-y)*s;
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,H);
    const cells=activeCells();
    for(const c of cells){ const i=c%NX, j=(c/NX)|0; ctx.fillStyle=rgba('node',.05); ctx.fillRect(X(i),Y(j+1),s,s); }
    if(S.basis!=='pic'){ const B=3; ctx.fillStyle=rgba('collider',.30); for(const c of cells){ const i=c%NX, j=(c/NX)|0; for(let py=Y(j+1);py<Y(j);py+=B) for(let px=X(i);px<X(i+1);px+=B){ const x=(px+B/2)/s, y=NY-(py+B/2)/s; if(interp(c,x,y)<0) ctx.fillRect(px,py,B,B); } } }
    else { ctx.fillStyle=rgba('collider',.18); ctx.beginPath(); ctx.arc(X(S.cx),Y(S.cy),S.R*s,0,7); ctx.fill(); }
    ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=NX;i++){ ctx.moveTo(X(i)+.5,0); ctx.lineTo(X(i)+.5,H);} for(let j=0;j<=NY;j++){ ctx.moveTo(0,Y(j)+.5); ctx.lineTo(w,Y(j)+.5);} ctx.stroke();
    ctx.strokeStyle=COL.ink; ctx.setLineDash([5,5]); ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(X(S.cx),Y(S.cy),S.R*s,0,7); ctx.stroke(); ctx.setLineDash([]);
    for(const p of P){ ctx.fillStyle=rgba('particle', S.basis==='pic'?.35:.75); ctx.beginPath(); ctx.arc(X(p.x),Y(p.y),Math.max(2,s*.07),0,7); ctx.fill(); }
    let pts=[]; if(S.basis==='pic') pts=P.map(p=>({x:p.x,y:p.y})); else { const seen=new Set(); for(const c of cells){ const i=c%NX, j=(c/NX)|0; const loc=[[0,0],[1,0],[0,1],[1,1]]; if(S.basis==='S2') loc.push([.5,0],[.5,1],[0,.5],[1,.5]); for(const [a,b] of loc){ const k=(i+a)*2+','+(j+b)*2; if(!seen.has(k)){ seen.add(k); pts.push({x:i+a,y:j+b}); } } } }
    let nAct=0, nPen=0;
    for(const q of pts){ const d=sd(q.x,q.y); const nx=(q.x-S.cx)/(d+S.R), ny=(q.y-S.cy)/(d+S.R);
      let active, pen; if(S.basis==='pic'){ active=d-rp<h; pen=d<Math.max(rp,.25*h); } else { active=d<.25*h; pen=d<0; }
      const x0=X(q.x), y0=Y(q.y);
      if(active){ nAct++; if(pen) nPen++; arrow(ctx,x0,y0,x0+nx*s*.32,y0-ny*s*.32,rgba('collider',.85),1.4);
        ctx.fillStyle=pen?COL.collider:COL.surface; ctx.strokeStyle=COL.collider; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(x0,y0,S.basis==='pic'?3.5:4.5,0,7); ctx.fill(); ctx.stroke(); }
      else { ctx.fillStyle=rgba('muted',.55); ctx.beginPath(); ctx.arc(x0,y0,S.basis==='pic'?1.8:2.6,0,7); ctx.fill(); } }
    ctx.fillStyle=rgba('ink',.6); ctx.beginPath(); ctx.arc(X(S.cx),Y(S.cy),4,0,7); ctx.fill();
    $('#cbCap').textContent = S.basis==='pic' ? 'filled = within stop distance \u00B7 ring = active, approaching \u00B7 grey = inactive \u00B7 arrows = contact normal' : 'shaded = solid as the solver sees it \u00B7 filled = inside wall \u00B7 ring = within 0.25 voxel \u00B7 grey = free';
    $('#cbStats').innerHTML=`<span>active cells <b>${cells.length}</b></span><span>contact points <b>${pts.length}</b></span><span>active <b>${nAct}</b></span><span>${S.basis==='pic'?'at stop distance':'inside wall'} <b>${nPen}</b></span>`;
    const F=FACTS[S.basis]; $('#cbFacts').innerHTML=`<h3>${F.name}</h3><dl><dt>Points</dt><dd>${F.pts}</dd><dt>Behavior</dt><dd>${F.how}</dd></dl>`;
  }
  function init(){
    seg('#cbBasis', v=>{S.basis=v; draw();});
    $('#sR').oninput=e=>{ S.R=+e.target.value; $('#oR').textContent=S.R.toFixed(2); draw(); };
    const cv=$('#cbCv'); const toW=e=>{ const r=cv.getBoundingClientRect(); const s=r.width/NX; return [(e.clientX-r.left)/s, NY-(e.clientY-r.top)/s]; };
    const set=([x,y])=>{ S.cx=Math.min(5.5,Math.max(.5,x)); S.cy=Math.min(3,Math.max(-.5,y)); draw(); };
    cv.addEventListener('pointerdown', e=>{ S.drag=true; cv.setPointerCapture(e.pointerId); set(toW(e)); });
    cv.addEventListener('pointermove', e=>{ if(S.drag) set(toW(e)); });
    cv.addEventListener('pointerup', ()=>{ S.drag=false; });
    cv.addEventListener('keydown', e=>{ const d=.05; const m={ArrowLeft:[-d,0],ArrowRight:[d,0],ArrowUp:[0,d],ArrowDown:[0,-d]}[e.key]; if(m){ e.preventDefault(); set([S.cx+m[0], S.cy+m[1]]); } });
  }
  return {init, draw};
})();

// =====================================================================
// A2. Zoom: the solve on one cell
// =====================================================================
const ZOOM = (() => {
  const SHORT=['BL','BR','TL','TR'];
  const PART=[[.25,.25],[.75,.25],[.25,.75],[.75,.75]], PNAME=['bottom-left particle','bottom-right particle'];
  const wts=(fx,fy)=>[(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy];
  const PRE={
    land:{floor:true, v:[[.6,-1.5],[.6,-1.5],[.6,-1.5],[.6,-1.5]], note:'Start: all four corners fall at 1.5 and slide right at 0.6. The cell sits on the floor. Velocities are in cells per second.'},
    squeeze:{floor:false, v:[[1,0],[-1,0],[1,0],[-1,0]], note:'Start: left corners move right, right corners move left. The cell is squeezed from the sides.'},
    shear:{floor:false, v:[[0,0],[0,0],[1.2,0],[1.2,0]], note:'Start: top corners slide right, bottom corners stay. Layers slide over each other without squeezing.'},
    pull:{floor:false, v:[[-1,0],[1,0],[-1,0],[1,0]], note:'Start: corners move apart. The cell grows.'},
    spin:{floor:false, v:[[.6,-.6],[.6,.6],[-.6,-.6],[-.6,.6]], note:'Start: the corners circle the center. The cell rotates without changing size or shape.'},
  };
  const CNOTE={grid:'Contacts at the cell corners (Q1 collider basis): the bottom corners themselves may not move into the floor.',
    pic:'Contacts at the particles (pic collider basis). The two bottom particles sit at their stop distance above the floor (0.25 voxel, also their radius), so they may not approach any further. The two top particles are shielded: a particle right below them takes the contact first.'};
  const S={pre:'land', cmode:'grid', v:null, v0:null, lam:[0,0,0], rn:[0,0], rt:[0,0], it:0, phase:'cell', mu:.6, floor:true, hist:[], last:null, itStart:null, timer:0, drag:-1, done:false};
  const f2=x=>(Math.abs(x)<5e-3?0:x).toFixed(2);
  const cp=v=>v.map(a=>a.slice());
  const pvel=k=>{ const w=wts(...PART[k]); let x=0,y=0; for(let c=0;c<4;c++){ x+=w[c]*S.v[c][0]; y+=w[c]*S.v[c][1]; } return [x,y]; };
  function load(k){ const p=PRE[k]; S.pre=k; S.floor=p.floor; $('#zFloor').checked=p.floor; S.v=cp(p.v); S.v0=cp(p.v); restart(p.note); }
  function restart(note){ stopTimer(); S.v0=cp(S.v); S.lam=[0,0,0]; S.rn=[0,0]; S.rt=[0,0]; S.it=0; S.phase='cell'; S.done=false; S.hist=[]; S.last=null; S.mc=null; S.mf=null; if(note) $('#zNote').textContent=note+(S.floor?' '+CNOTE[S.cmode]:''); }
  function rates(v){ const ax=v[1][0]-v[0][0]+v[3][0]-v[2][0], by=v[2][1]-v[0][1]+v[3][1]-v[1][1], ay=v[2][0]-v[0][0]+v[3][0]-v[1][0], bx=v[1][1]-v[0][1]+v[3][1]-v[2][1];
    return {vol:.5*(ax+by), dev:.5*(ax-by), shr:.5*(ay+bx)}; }
  function push(l){ const a=.5, P=l[0]+l[1], Q=l[0]-l[1], s=l[2]; return [[a*(-P-s),a*(-Q-s)],[a*(P-s),a*(-Q+s)],[a*(-P+s),a*(Q-s)],[a*(P+s),a*(Q+s)]]; }

  function cellFix(){
    S.it++; S.itStart=cp(S.v); const before=cp(S.v);
    const r=rates(S.v), W=2, ln=S.lam[0], ld=S.lam[1], ls=S.lam[2];
    const nn=Math.max(0, ln - r.vol/W); let td=ld - r.dev/W, ts=ls - r.shr/W; const tn=Math.hypot(td,ts), cap=S.mu*nn; let capped=false;
    const tdT=td, tsT=ts; if(tn>cap+1e-9){ const k=tn>0?cap/tn:0; td*=k; ts*=k; capped=true; }
    S.mc={it:S.it, ln, vol:r.vol, nn, tdT, tsT, td, ts, cap, capped};
    const d=[nn-ln, td-ld, ts-ls]; S.lam=[nn,td,ts]; const pd=push(d); for(let c=0;c<4;c++){ S.v[c][0]+=pd[c][0]; S.v[c][1]+=pd[c][1]; }
    const sr=Math.hypot(r.dev,r.shr), tau=Math.hypot(td,ts); let t;
    if(r.vol<-5e-3) t=`The corners are squeezing the cell: its area is shrinking at <b>${f2(-r.vol)}</b>. Pressure goes up <b>${f2(ln)} \u2192 ${f2(nn)}</b> and pushes the corners apart (violet).`;
    else if(nn<ln-5e-3) t=`The cell is now growing slightly (area rate +${f2(r.vol)}), so pressure relaxes <b>${f2(ln)} \u2192 ${f2(nn)}</b>.`;
    else t=`The cell is not being squeezed (area rate ${f2(r.vol)}), so pressure stays at <b>${f2(nn)}</b>.`;
    let fr='\u2013';
    if(sr>5e-3){
      if(nn<5e-3){ t+=` It is changing shape (rate ${f2(sr)}), but with zero pressure friction has no grip, so nothing resists.`; fr='no grip'; }
      else if(capped){ t+=` It is also changing shape (rate ${f2(sr)}). Friction resists with its maximum, \u03BC \u00D7 pressure = <b>${f2(cap)}</b>. That is not enough to stop it, so the material keeps sliding.`; fr='sliding'; }
      else { t+=` Friction fully stops its shape change (uses ${f2(tau)} of the ${f2(cap)} available).`; fr='holding'; }
    }
    S.last={kind:'cell', before, title:`Iteration ${S.it} \u00B7 cell fix`, text:t};
    S.hist.push({it:S.it, sq:Math.max(0,-r.vol), p:nn, fr, bl:null, br:null});
    if(S.floor) S.phase='floor'; else endIter();
  }
  function mathHTML(){
    const v=S.v, r=rates(v), n=x=>(Math.abs(x)<5e-4?0:x).toFixed(2), mc=S.mc, mf=S.mf;
    const sx=.5*((v[1][0]-v[0][0])+(v[3][0]-v[2][0])), sy=.5*((v[2][1]-v[0][1])+(v[3][1]-v[1][1]));
    const tbl=`<table class="zvt"><tr><th></th><th>x</th><th>y</th></tr>${SHORT.map((s,c)=>`<tr><td>${s}</td><td>${n(v[c][0])}</td><td>${n(v[c][1])}</td></tr>`).join('')}</table>`;
    const c0=`<div class="zm"><h4>0 \u00B7 Corner velocities now</h4>${tbl}<p>Every number below is computed from these four arrows. Units: cells per second. Corner mass m = 1.</p></div>`;
    const c1=`<div class="zm"><h4>1 \u00B7 Squeeze now</h4><div class="formula">x-stretch = \u00BD[(BR.x \u2212 BL.x) + (TR.x \u2212 TL.x)]
          = \u00BD[(${n(v[1][0])} \u2212 ${n(v[0][0])}) + (${n(v[3][0])} \u2212 ${n(v[2][0])})] = ${n(sx)}
y-stretch = \u00BD[(TL.y \u2212 BL.y) + (TR.y \u2212 BR.y)]
          = \u00BD[(${n(v[2][1])} \u2212 ${n(v[0][1])}) + (${n(v[3][1])} \u2212 ${n(v[1][1])})] = ${n(sy)}
area rate = x-stretch + y-stretch = ${n(r.vol)}
squeeze   = max(0, \u2212area rate) = ${n(Math.max(0,-r.vol))}</div><p>How fast the cell's area changes. Negative = shrinking. Right side moving away from left side, and top moving away from bottom, both grow the area.</p></div>`;
    const c2 = mc ? `<div class="zm"><h4>2 \u00B7 Pressure (last cell fix, iteration ${mc.it})</h4><div class="formula">W = \u00BD \u00D7 (1/m_BL + 1/m_BR + 1/m_TL + 1/m_TR) = \u00BD \u00D7 4 = 2
p_new = max(0, p_old \u2212 area rate \u00F7 W)
      = max(0, ${n(mc.ln)} \u2212 (${n(mc.vol)}) \u00F7 2) = ${n(mc.nn)}
\u0394p = ${n(mc.nn-mc.ln)}  \u2192  pressure moves every corner ${n(.5*(mc.nn-mc.ln))}
     outward in x and in y (\u00BD\u00B7\u0394p each)</div><p>W (the Delassus number) says how much one unit of pressure changes this cell's area rate. Dividing the squeeze by W gives exactly the extra pressure that cancels it. Pressure can only push, hence max(0, \u2026).</p></div>`
      : `<div class="zm"><h4>2 \u00B7 Pressure</h4><p>Press <b>Next fix</b>. After the first cell fix this card shows the pressure update with its numbers.</p></div>`;
    const sr=Math.hypot(r.dev,r.shr);
    const c3 = `<div class="zm"><h4>3 \u00B7 Shape change and friction</h4><div class="formula">now:
  stretch difference = x-stretch \u2212 y-stretch = ${n(r.dev)}
  shear = \u00BD[(TL.x\u2212BL.x)+(TR.x\u2212BR.x)] + \u00BD[(BR.y\u2212BL.y)+(TR.y\u2212TL.y)] = ${n(r.shr)}
  shape rate = \u221A(difference\u00B2 + shear\u00B2) = ${n(sr)}${mc?`
last cell fix (iteration ${mc.it}), rates measured before it:
  trial \u03C4 = \u03C4_old \u2212 rates \u00F7 W = (${n(mc.tdT)}, ${n(mc.tsT)}),  |\u03C4| = ${n(Math.hypot(mc.tdT,mc.tsT))}
  limit   = \u03BC \u00D7 p = ${n(S.mu)} \u00D7 ${n(mc.nn)} = ${n(mc.cap)}
  ${mc.capped?`|\u03C4| > limit \u2192 shrunk to the limit: material slides`:`|\u03C4| \u2264 limit \u2192 accepted: material holds`}
  \u03C4 = (${n(mc.td)}, ${n(mc.ts)})`:''}</div><p>"Friction used" in the readout = |\u03C4| \u00F7 (\u03BC \u00D7 p). The trial \u03C4 is the shear stress that would stop the shape change completely. Friction may only supply up to \u03BC \u00D7 pressure.</p></div>`;
    let c4;
    if(!S.floor) c4=`<div class="zm"><h4>4 \u00B7 Floor</h4><p>The floor is switched off.</p></div>`;
    else if(!mf) c4=`<div class="zm"><h4>4 \u00B7 Floor</h4><div class="formula">into the floor (readout) = max(0, \u2212lowest bottom ${S.cmode==='pic'?'particle':'corner'} y-velocity) = ${n(Math.max(0,-Math.min(S.cmode==='pic'?pvel(0)[1]:v[0][1], S.cmode==='pic'?pvel(1)[1]:v[1][1])))}</div><p>Press <b>Next fix</b> twice: the floor fix comes right after each cell fix.</p></div>`;
    else c4=`<div class="zm"><h4>4 \u00B7 Floor (last floor fix, iteration ${mf.it})</h4><div class="formula">${mf.rows.map(q=> mf.mode==='grid' ? `${q.name}: before (${n(q.v0[0])}, ${n(q.v0[1])})
  remove the floor's earlier push:  u = (${n(q.u[0])}, ${n(q.u[1])})
  normal push   r_n = max(0, \u2212u_y) = ${n(q.rn)}
  friction budget   = 0.5 \u00D7 r_n = ${n(q.lim)}
  friction      r_t = \u2212sign(u_x)\u00B7min(|u_x|, budget) = ${n(q.rt)}
  after = (u_x + r_t, u_y + r_n) = (${n(q.v1[0])}, ${n(q.v1[1])})` : `${q.name}: velocity = \u03A3 w\u00B7corner = (${n(q.v0[0])}, ${n(q.v0[1])})
  W = \u03A3 w\u00B2 = ${q.W.toFixed(3)}   (weights 0.56, 0.19, 0.19, 0.06)
  remove the floor's earlier push:  u = v \u2212 W\u00B7(r_t, r_n) = (${n(q.u[0])}, ${n(q.u[1])})
  normal push   r_n = max(0, \u2212u_y \u00F7 W) = ${n(q.rn)}
  friction budget   = 0.5 \u00D7 r_n = ${n(q.lim)}
  friction      r_t = clamp(\u2212u_x \u00F7 W, \u00B1budget) = ${n(q.rt)}
  every corner += its weight \u00D7 (\u0394r_t, \u0394r_n)  \u2192  particle (${n(q.v1[0])}, ${n(q.v1[1])})`).join('\n')}</div><p>r_n and r_t are running totals for the whole solve, which is why the floor's earlier push is removed first and then recomputed. 0.5 is the floor's friction coefficient (fixed in this demo; the \u03BC slider is the sand's).</p></div>`;
    return c0+c1+c2+c3+c4;
  }
  function floorFixGrid(){
    const before=cp(S.v), into=[0,0], s0=[0,0], s1=[0,0]; const rows=[];
    for(const c of [0,1]){ const cur=S.v[c][1]; into[c]=Math.max(0,-cur); s0[c]=S.v[c][0]; const v0=S.v[c].slice();
      const ux=S.v[c][0]-S.rt[c], uy=S.v[c][1]-S.rn[c]; const rr=Math.max(0,-uy), lim=.5*rr; const rt=-Math.sign(ux)*Math.min(Math.abs(ux),lim);
      S.rn[c]=rr; S.rt[c]=rt; S.v[c]=[ux+rt, uy+rr]; s1[c]=S.v[c][0]; rows.push({name:SHORT[c]+' corner', v0, u:[ux,uy], rn:rr, lim, rt, v1:S.v[c].slice()}); }
    S.mf={mode:'grid', it:S.it, rows};
    let t;
    if(Math.max(into[0],into[1])>5e-3) t=`The bottom corners are moving into the floor (down at <b>${f2(into[0])}</b> and <b>${f2(into[1])}</b>)${S.it>1?', because the cell\u2019s pressure just pushed them down':''}. The floor cancels that (rose).`;
    else t='Nothing is moving into the floor, so the floor does nothing.';
    if(Math.abs(s1[0]-s0[0])>5e-3 || Math.abs(s1[1]-s0[1])>5e-3) t+=` Floor friction changes their sliding speed from ${f2(s0[0])} and ${f2(s0[1])} to ${f2(s1[0])} and ${f2(s1[1])}.`;
    return {before, into, t};
  }
  function floorFixPic(){
    const before=cp(S.v), into=[0,0], s0=[0,0], s1=[0,0]; const rows=[];
    for(const k of [0,1]){ const w=wts(...PART[k]); const W=w.reduce((a,x)=>a+x*x,0); const [vx,vy]=pvel(k); into[k]=Math.max(0,-vy); s0[k]=vx;
      const ux=vx-W*S.rt[k], uy=vy-W*S.rn[k]; const rr=Math.max(0,-uy/W), lim=.5*rr; let rt=-ux/W; if(Math.abs(rt)>lim) rt=Math.sign(rt)*lim;
      const dn=rr-S.rn[k], dt_=rt-S.rt[k]; S.rn[k]=rr; S.rt[k]=rt; for(let c=0;c<4;c++){ S.v[c][0]+=w[c]*dt_; S.v[c][1]+=w[c]*dn; } s1[k]=pvel(k)[0]; rows.push({name:(k?'bottom-right':'bottom-left')+' particle', v0:[vx,vy], W, u:[ux,uy], rn:rr, lim, rt, v1:pvel(k)}); }
    S.mf={mode:'pic', it:S.it, rows};
    let t;
    if(Math.max(into[0],into[1])>5e-3) t=`The floor is checked at the particles, not at the corners. Each bottom particle\u2019s velocity is a blend of all four corners (56% from its nearest corner, 19% from each side neighbor, 6% from the far one). They are moving into the floor at <b>${f2(into[0])}</b> and <b>${f2(into[1])}</b>${into[1]<into[0]-.05?' (the second is already slower: the first particle\u2019s push also lifted the corners they share)':''}. Each push is spread back to all four corners with those same weights (rose), so even the top corners get lifted a little.`;
    else t='Neither bottom particle is moving into the floor, so the floor does nothing.';
    if(Math.abs(s1[0]-s0[0])>5e-3 || Math.abs(s1[1]-s0[1])>5e-3) t+=` Floor friction changes the particles\u2019 sliding speed from ${f2(s0[0])} and ${f2(s0[1])} to ${f2(s1[0])} and ${f2(s1[1])}.`;
    return {before, into, t};
  }
  function floorFix(){
    const res = S.cmode==='pic' ? floorFixPic() : floorFixGrid();
    const h=S.hist[S.hist.length-1]; h.bl=res.into[0]; h.br=res.into[1];
    S.last={kind:'floor', before:res.before, title:`Iteration ${S.it} \u00B7 floor fix`, text:res.t};
    S.phase='cell'; endIter();
  }
  function endIter(){ let ch=0; for(let c=0;c<4;c++) ch=Math.max(ch, Math.abs(S.v[c][0]-S.itStart[c][0]), Math.abs(S.v[c][1]-S.itStart[c][1]));
    if(ch<2e-3){ S.done=true; S.last.title=`Iteration ${S.it} \u00B7 converged`; S.last.text+=' Nothing changed during this whole iteration, so the solver stops. These are the velocities the particles will read back.'+(S.cmode==='pic'&&S.floor?' With particle contacts, corner arrows may still point into the floor. Only the particles\u2019 own velocities must not.':''); } }
  function nextFix(){ if(S.done){ S.v=cp(S.v0); restart(); } if(S.phase==='cell') cellFix(); else floorFix(); }
  function stopTimer(){ if(S.timer){ clearInterval(S.timer); S.timer=0; const b=$('#zSolve'); if(b) b.textContent='Solve'; } }
  function solve(){ if(S.timer){ stopTimer(); return; } if(S.done){ S.v=cp(S.v0); restart(); }
    if(motion.matches){ for(let i=0;i<120 && !S.done;i++) nextFix(); draw(); return; }
    $('#zSolve').textContent='Pause'; S.timer=setInterval(()=>{ nextFix(); draw(); if(S.done || S.it>=40) stopTimer(); }, 550); }

  let geo=null;
  function drawChart(){
    const cv=$('#zChart'); const {ctx,w,h}=fit(cv,.3); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const n=Math.max(8,S.hist.length), pad=6, base=h-16, top=8; let mx=.1; for(const e of S.hist) mx=Math.max(mx,e.sq,e.bl||0,e.br||0);
    const gw=(w-2*pad)/n, bw=Math.max(2,gw*.34);
    ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(pad,base+.5); ctx.lineTo(w-pad,base+.5); ctx.stroke();
    ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='top'; ctx.fillStyle=COL.muted;
    for(let i=0;i<n;i++){ const x=pad+gw*i+gw/2; ctx.fillText(String(i+1), x, base+3); const e=S.hist[i]; if(!e) continue;
      const h1=(base-top)*e.sq/mx; ctx.fillStyle=COL.strain; ctx.fillRect(x-bw-1, base-h1, bw, h1);
      if(e.bl!==null){ const v=Math.max(e.bl,e.br), h2=(base-top)*v/mx; ctx.fillStyle=COL.collider; ctx.fillRect(x+1, base-h2, bw, h2); }
      ctx.fillStyle=COL.muted; }
    if(!S.hist.length){ ctx.textBaseline='middle'; ctx.fillText('bars appear as you iterate', w/2, (base+top)/2); }
  }
  function draw(){
    const cv=$('#zCv'); const {ctx,w,h}=fit(cv,.62); const L=Math.min(h*.46, w*.34), cx=w/2, cy=h*.42, k=L*.24;
    const P=[[cx-L/2,cy+L/2],[cx+L/2,cy+L/2],[cx-L/2,cy-L/2],[cx+L/2,cy-L/2]]; geo={P,k};
    const PP=PART.map(([fx,fy])=>[cx-L/2+fx*L, cy+L/2-fy*L]);
    const pic=S.cmode==='pic';
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const lk=S.last?S.last.kind:null;
    if(S.floor){ const fy=cy+L/2; ctx.fillStyle=rgba('collider',.12); ctx.fillRect(0,fy,w,h-fy); ctx.strokeStyle=rgba('collider',.25); ctx.lineWidth=1; ctx.beginPath(); for(let x=-h;x<w;x+=12){ ctx.moveTo(x,h); ctx.lineTo(x+(h-fy),fy); } ctx.stroke(); ctx.strokeStyle=COL.collider; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(0,fy); ctx.lineTo(w,fy); ctx.stroke(); ctx.fillStyle=COL.collider; ctx.font=FONT_MONO; ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText('floor', 10, fy+8); }
    ctx.fillStyle=rgba('strain', .05+Math.min(.45,.25*S.lam[0])); ctx.fillRect(cx-L/2,cy-L/2,L,L);
    ctx.strokeStyle=lk==='cell'&&!S.done?COL.strain:COL.ink; ctx.lineWidth=lk==='cell'&&!S.done?3:1.5; ctx.strokeRect(cx-L/2,cy-L/2,L,L);
    ctx.fillStyle=COL.strain; ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(`pressure ${f2(S.lam[0])}`, cx, cy);
    const q=P.map((p,c)=>[p[0]+S.v[c][0]*k*.9, p[1]-S.v[c][1]*k*.9]); const r=rates(S.v);
    ctx.setLineDash([5,4]); ctx.strokeStyle=r.vol<-5e-3?COL.strain:rgba('ink',.4); ctx.lineWidth=1.3; ctx.beginPath(); ctx.moveTo(...q[0]); ctx.lineTo(...q[1]); ctx.lineTo(...q[3]); ctx.lineTo(...q[2]); ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    // particles
    for(let i=0;i<4;i++){ const [x,y]=PP[i];
      if(pic && S.floor){
        if(i<2){ ctx.strokeStyle=COL.collider; ctx.lineWidth=lk==='floor'&&!S.done?3:1.8; ctx.beginPath(); ctx.arc(x,y,lk==='floor'&&!S.done?12:9,0,7); ctx.stroke(); }
        else { ctx.setLineDash([3,3]); ctx.strokeStyle=rgba('muted',.8); ctx.lineWidth=1.2; ctx.beginPath(); ctx.arc(x,y,9,0,7); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle=COL.muted; ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.fillText('shielded', x, y-12); } }
      if(pic){ const [vx,vy]=pvel(i); arrow(ctx,x,y,x+vx*k*.8,y-vy*k*.8,rgba('particle',.9),1.6); }
      ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(x,y,4.5,0,7); ctx.fill(); }
    if(lk==='floor' && !S.done && !pic) for(const c of [0,1]){ ctx.strokeStyle=COL.collider; ctx.lineWidth=2.5; ctx.beginPath(); ctx.arc(P[c][0],P[c][1],11,0,7); ctx.stroke(); }
    for(let c=0;c<4;c++){ const [x,y]=P[c]; const tx=x+S.v[c][0]*k, ty=y-S.v[c][1]*k;
      if(S.last){ const b=S.last.before[c], bx=x+b[0]*k, by=y-b[1]*k;
        if(Math.hypot(tx-bx,ty-by)>1.5){ ctx.setLineDash([4,4]); arrow(ctx,x,y,bx,by,rgba('muted',.8),1.5); ctx.setLineDash([]); arrow(ctx,bx,by,tx,ty,S.last.kind==='cell'?COL.strain:COL.collider,2.5); } }
      arrow(ctx,x,y,tx,ty,COL.node,3);
      ctx.fillStyle=COL.surface; ctx.strokeStyle=COL.node; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(tx,ty,5,0,7); ctx.fill(); ctx.stroke();
      ctx.fillStyle=COL.ink; ctx.beginPath(); ctx.arc(x,y,4.5,0,7); ctx.fill();
      ctx.fillStyle=COL.muted; ctx.font=FONT_MONO; ctx.textAlign=c%2?'left':'right'; ctx.textBaseline=c<2?'top':'bottom'; ctx.fillText(SHORT[c], x+(c%2?12:-12), y+(c<2?8:-8)); }
    const fx=$('#zFix'); fx.className='zfix '+(S.done?'ok':(lk||''));
    fx.innerHTML = S.last ? `<b class="t">${S.last.title}</b><span>${S.last.text}</span>` : `<b class="t">Start \u00B7 right after particle \u2192 grid</b><span>These corner velocities are just free motion. Press <b>Next fix</b>.</span>`;
    const sq=Math.max(0,-r.vol), sr=Math.hypot(r.dev,r.shr), tau=Math.hypot(S.lam[1],S.lam[2]), cap=S.mu*S.lam[0];
    const frs = sr<5e-3 ? 'no shape change' : (S.lam[0]<5e-3 ? 'no pressure, no grip' : (tau>=cap-1e-6 ? 'at its limit: sliding' : 'holding'));
    const down=S.floor?Math.max(0,-Math.min(pic?pvel(0)[1]:S.v[0][1], pic?pvel(1)[1]:S.v[1][1])):0;
    $('#zRead').innerHTML=`<div><span>Squeeze now</span><b>${f2(sq)}</b><em>${sq>=5e-3?'area shrinking':'area not shrinking'}</em></div>`+
      `<div><span>Pressure</span><b>${f2(S.lam[0])}</b><em>pushes corners apart</em></div>`+
      `<div><span>Friction used</span><b>${f2(tau)} / ${f2(cap)}</b><em>${frs}</em></div>`+
      `<div><span>${S.floor?(pic?'Particles into floor':'Corners into floor'):'Iterations'}</span><b>${S.floor?f2(down):S.it}</b><em>${S.floor?(down>=5e-3?'still entering':'none entering'):(S.done?'converged':'')}</em></div>`;
    $('#zThFloor').textContent = pic ? 'into floor, bottom particles' : 'into floor BL / BR';
    $('#zTab').innerHTML = S.hist.length ? S.hist.map(e=>`<tr><td>${e.it}</td><td>${f2(e.sq)}</td><td>${f2(e.p)}</td><td>${e.bl===null?(S.floor?'\u2026':'\u2013'):f2(e.bl)+' / '+f2(e.br)}</td><td>${e.fr}</td></tr>`).join('') : `<tr><td colspan="5" style="color:var(--muted)">One row per iteration.</td></tr>`;
    const tb=document.querySelector('.ztab'); if(tb) tb.scrollTop=tb.scrollHeight;
    $('#zMath').innerHTML=mathHTML();
    drawChart();
  }
  function init(){
    seg('#zPre', v=>{ load(v); draw(); });
    seg('#zCol', v=>{ S.cmode=v; S.v=cp(S.v0); restart(PRE[S.pre].note); draw(); });
    $('#zIt').onclick=()=>{ stopTimer(); nextFix(); draw(); };
    registerPlayback('zSolve', stopTimer, { keep: true });
    motion.addEventListener('change', stopTimer);
    $('#zSolve').onclick=()=>{ solve(); draw(); };
    $('#zReset').onclick=()=>{ load(S.pre); draw(); };
    $('#zFloor').onchange=e=>{ S.floor=e.target.checked; S.v=cp(S.v0); restart(PRE[S.pre].note); draw(); };
    $('#zMu').oninput=e=>{ S.mu=+e.target.value; $('#oZMu').textContent=S.mu.toFixed(2); S.v=cp(S.v0); restart(); draw(); };
    const cv=$('#zCv'); const pos=e=>{ const r=cv.getBoundingClientRect(); return [e.clientX-r.left, e.clientY-r.top]; };
    cv.addEventListener('pointerdown', e=>{ if(!geo) return; const [x,y]=pos(e); let best=-1, bd=22; for(let c=0;c<4;c++){ const tx=geo.P[c][0]+S.v[c][0]*geo.k, ty=geo.P[c][1]-S.v[c][1]*geo.k; const d=Math.hypot(x-tx,y-ty); if(d<bd){bd=d;best=c;} } if(best>=0){ S.drag=best; cv.setPointerCapture(e.pointerId); } });
    cv.addEventListener('pointermove', e=>{ if(S.drag<0) return; const [x,y]=pos(e); const c=S.drag; const cl=v=>Math.max(-3,Math.min(3,v)); S.v[c]=[cl((x-geo.P[c][0])/geo.k), cl(-(y-geo.P[c][1])/geo.k)]; restart('Start: your own corner velocities. Press Next fix or Solve.'); draw(); });
    cv.addEventListener('pointerup', ()=>{ S.drag=-1; });
    load('land');
  }
  return {init, draw};
})();

// =====================================================================
// Shared: categorical colors for graph colorings (theme-aware)
// =====================================================================
function catColor(i, n, alpha){ const dark = (COL.bgRGB[0]+COL.bgRGB[1]+COL.bgRGB[2]) < 300; const h=(i*360/n + 200) % 360; return `hsla(${h.toFixed(0)}, ${dark?62:58}%, ${dark?62:50}%, ${alpha==null?1:alpha})`; }

// =====================================================================
// E2a. Drucker-Prager projection widget
// =====================================================================
const DP = (() => {
  const S={p:2.4, t:2.3, mu:.6, c:.25, drag:false}; const P0=-1.6, P1=4.2, T1=3.4; let geo=null;
  function project(){ const apex=-S.c/S.mu; if(S.t <= S.mu*S.p + S.c + 1e-12 && S.p >= apex) return {p:S.p,t:S.t,kind:'hold'};
    const L=Math.hypot(1,S.mu), dx=1/L, dy=S.mu/L; const tt=(S.p-apex)*dx + S.t*dy; if(tt<=0) return {p:apex,t:0,kind:'pull'};
    return {p:apex+tt*dx, t:tt*dy, kind:'slide'}; }
  function draw(){
    const cv=$('#dpCv'); const {ctx,w,h}=fit(cv,.66); const padL=44, padB=34, padT=12, padR=12;
    const X=p=>padL+(p-P0)/(P1-P0)*(w-padL-padR), Y=t=>h-padB-(t/T1)*(h-padB-padT); geo={X,Y,w,h,padL,padB,padR,padT};
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const apex=-S.c/S.mu;
    ctx.fillStyle=rgba('collider',.08); ctx.fillRect(X(P0),Y(T1),X(apex)-X(P0),Y(0)-Y(T1));
    ctx.fillStyle=rgba('strain',.13); ctx.beginPath(); ctx.moveTo(X(apex),Y(0)); const pe=Math.min(P1,(T1-S.c)/S.mu); ctx.lineTo(X(pe),Y(S.mu*pe+S.c)); if(pe<P1) ctx.lineTo(X(P1),Y(T1)); ctx.lineTo(X(P1),Y(0)); ctx.closePath(); ctx.fill();
    ctx.strokeStyle=COL.grid; ctx.lineWidth=1; ctx.beginPath(); for(let p=Math.ceil(P0);p<=P1;p++){ ctx.moveTo(X(p)+.5,Y(0)); ctx.lineTo(X(p)+.5,Y(T1)); } for(let t=1;t<=T1;t++){ ctx.moveTo(X(P0),Y(t)+.5); ctx.lineTo(X(P1),Y(t)+.5); } ctx.stroke();
    ctx.strokeStyle=COL.ink; ctx.lineWidth=1.2; ctx.beginPath(); ctx.moveTo(X(P0),Y(0)); ctx.lineTo(X(P1),Y(0)); ctx.moveTo(X(0),Y(0)); ctx.lineTo(X(0),Y(T1)); ctx.stroke();
    ctx.strokeStyle=COL.strain; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(X(apex),Y(0)); ctx.lineTo(X(pe),Y(S.mu*pe+S.c)); ctx.stroke();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top'; for(let p=Math.ceil(P0);p<=P1;p++) ctx.fillText(String(p),X(p),Y(0)+4);
    ctx.fillText('pressure p  (squeeze \u2192)', (X(0)+X(P1))/2, h-15);
    ctx.save(); ctx.translate(13,(Y(0)+Y(T1))/2); ctx.rotate(-Math.PI/2); ctx.fillText('shear |\u03C4|', 0, 0); ctx.restore();
    ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillStyle=COL.collider; ctx.fillText('pulling apart', X(P0)+6, Y(T1)+4);
    ctx.fillStyle=COL.strain; ctx.fillText('allowed: |\u03C4| \u2264 \u03BC p + c', X(1.3), Y(T1)+4);
    if(S.c>0){ ctx.fillStyle=COL.muted; ctx.textAlign='right'; ctx.textBaseline='middle'; ctx.fillText('c', X(0)-5, Y(S.c)); ctx.fillStyle=COL.strain; ctx.beginPath(); ctx.arc(X(0),Y(S.c),3,0,7); ctx.fill(); }
    const q=project();
    if(q.kind!=='hold'){ ctx.setLineDash([5,4]); arrow(ctx,X(S.p),Y(S.t),X(q.p),Y(q.t),COL.collider,2); ctx.setLineDash([]); }
    ctx.fillStyle=COL.strain; ctx.beginPath(); ctx.arc(X(q.p),Y(q.t),7,0,7); ctx.fill();
    ctx.strokeStyle=COL.ink; ctx.lineWidth=2.5; ctx.beginPath(); ctx.arc(X(S.p),Y(S.t),10,0,7); ctx.stroke();
    ctx.fillStyle=COL.ink; ctx.font=FONT_MONO; ctx.textAlign='left'; ctx.textBaseline='bottom'; ctx.fillText('trial', X(S.p)+12, Y(S.t)-4);
    const cut=Math.hypot(S.p-q.p,S.t-q.t);
    const V={hold:['ok','Holds (sticks)',`The trial stress is inside the cone. The material can supply it, so it is accepted as is. The node's strain rate is cancelled: no plastic flow, the sand holds.`],
      slide:['cell','Slides (yields)',`Too much shear for this pressure. The stress is cut back to the cone edge: shear ${q.t.toFixed(2)} = \u03BC\u00B7${q.p.toFixed(2)} + c. The cut-away part (${cut.toFixed(2)}) becomes plastic flow, so the layers slide.`],
      pull:['floor','Pulls apart (separates)',`The trial stress asks sand to pull (pressure below the apex). Sand can't pull more than its cohesion, so the stress drops to the apex. The node is allowed to expand: grains separate, as at a free surface.`]}[q.kind];
    const vd=$('#dpVerdict'); vd.className='zfix '+V[0]; vd.innerHTML=`<b class="t">${V[1]}</b><span>${V[2]}</span>`;
    $('#dpCap').textContent=`trial (${S.p.toFixed(2)}, ${S.t.toFixed(2)}) \u2192 projected (${q.p.toFixed(2)}, ${q.t.toFixed(2)}). Drag the ring, or use arrow keys.`;
  }
  function init(){
    const cv=$('#dpCv'); const to=e=>{ const r=cv.getBoundingClientRect(); const x=e.clientX-r.left, y=e.clientY-r.top; const g=geo; S.p=Math.max(P0,Math.min(P1,P0+(x-g.padL)/(g.w-g.padL-g.padR)*(P1-P0))); S.t=Math.max(0,Math.min(T1,(g.h-g.padB-y)/(g.h-g.padB-g.padT)*T1)); draw(); };
    cv.addEventListener('pointerdown', e=>{ S.drag=true; cv.setPointerCapture(e.pointerId); to(e); });
    cv.addEventListener('pointermove', e=>{ if(S.drag) to(e); }); cv.addEventListener('pointerup', ()=>{ S.drag=false; });
    cv.addEventListener('keydown', e=>{ const d=.05, m={ArrowLeft:[-d,0],ArrowRight:[d,0],ArrowUp:[0,d],ArrowDown:[0,-d]}[e.key]; if(m){ e.preventDefault(); S.p=Math.max(P0,Math.min(P1,S.p+m[0])); S.t=Math.max(0,Math.min(T1,S.t+m[1])); draw(); } });
    $('#dpMu').oninput=e=>{ S.mu=+e.target.value; $('#oDpMu').textContent=S.mu.toFixed(2); draw(); };
    $('#dpC').oninput=e=>{ S.c=+e.target.value; $('#oDpC').textContent=S.c.toFixed(2); draw(); };
    document.querySelectorAll('[data-dp]').forEach(b=>b.addEventListener('click',()=>{ const k=b.dataset.dp; if(k==='hold'){S.p=2.4;S.t=.9;} else if(k==='slide'){S.p=2.4;S.t=2.6;} else {S.p=-1.3;S.t=.5;} draw(); }));
  }
  return {init, draw};
})();

// =====================================================================
// E2b. Graph coloring of strain cells
// =====================================================================
const GSC = (() => {
  const NX=12, NY=7; const S={n:1, mode:'gs', cur:0, hover:-1, timer:0};
  const per=()=>S.n+1;
  const colorOf=(i,j)=>{ const k=per(); return (i%k) + k*(j%k); };
  function stencil(i,j){ const n=S.n; if(n===1) return {lo:[i,j],cnt:2,off:0}; if(n===2) return {lo:[i-1,j-1],cnt:3,off:.5}; return {lo:[i-1,j-1],cnt:4,off:0}; }
  function draw(){
    const cv=$('#gsCv'); const {ctx,w,h}=fit(cv,NY/NX); const s=w/NX, X=x=>x*s, Y=y=>(NY-y)*s; const k=per(), nc=k*k;
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const hi=S.hover>=0?[S.hover%NX,(S.hover/NX)|0]:null;
    for(let j=0;j<NY;j++) for(let i=0;i<NX;i++){ const c=colorOf(i,j); const active = S.mode==='jac' ? true : c===S.cur;
      let a = active ? .9 : .22; if(hi){ const conf=Math.abs(i-hi[0])<k && Math.abs(j-hi[1])<k; a = (i===hi[0]&&j===hi[1]) ? 1 : conf ? .55 : .12; }
      ctx.fillStyle=catColor(c,nc,a); ctx.fillRect(X(i)+1,Y(j+1)+1,s-2,s-2);
      if(hi && Math.abs(i-hi[0])<k && Math.abs(j-hi[1])<k && !(i===hi[0]&&j===hi[1])){ ctx.strokeStyle=COL.collider; ctx.lineWidth=2; ctx.strokeRect(X(i)+2,Y(j+1)+2,s-4,s-4); } }
    ctx.strokeStyle=rgba('ink',.25); ctx.lineWidth=1; ctx.beginPath(); for(let i=0;i<=NX;i++){ ctx.moveTo(X(i)+.5,0); ctx.lineTo(X(i)+.5,h);} for(let j=0;j<=NY;j++){ ctx.moveTo(0,Y(j)+.5); ctx.lineTo(w,Y(j)+.5);} ctx.stroke();
    if(hi){ const st=stencil(hi[0],hi[1]); ctx.fillStyle=COL.node; for(let a=0;a<st.cnt;a++) for(let b=0;b<st.cnt;b++){ const x=st.lo[0]+a+st.off, y=st.lo[1]+b+st.off; ctx.beginPath(); ctx.arc(X(x),Y(y),Math.max(3,s*.09),0,7); ctx.fill(); }
      ctx.strokeStyle=COL.ink; ctx.lineWidth=2.5; ctx.strokeRect(X(hi[0])+1,Y(hi[1]+1)+1,s-2,s-2); }
    $('#gsCap').textContent = hi ? `This cell's stress touches the ${per()*per()} teal velocity nodes. Cells outlined in rose share at least one of them, so they must not update at the same time. All of them have other colors.` : (S.mode==='jac' ? 'Jacobi: every cell updates at once from the same old velocities. Neighbors push shared nodes together, so each push must be damped, and convergence is slower.' : `Color ${S.cur+1} of ${nc} updating: all bright cells run in parallel, because none of them share a velocity node.`);
    $('#gsRead').innerHTML=`<div><span>Colors in 2D</span><b>${nc}</b><em>${k} per axis</em></div><div><span>Colors in 3D</span><b>${k*k*k}</b><em>one sweep = ${k*k*k} parallel passes</em></div>`;
  }
  function tick(){ S.cur=(S.cur+1)%(per()*per()); draw(); }
  function stop(){
    if (!S.timer) return;
    clearInterval(S.timer);
    S.timer = 0;
    $('#gsPlay').textContent = 'Play sweep';
  }
  function init(){
    registerPlayback('gsPlay', stop);
    seg('#gsBasis', v=>{ S.n=+v; S.cur=0; draw(); }); seg('#gsMode', v=>{ S.mode=v; draw(); });
    $('#gsStep').onclick=()=>{ tick(); };
    $('#gsPlay').onclick=()=>{ if(S.timer){ stop(); } else { S.timer=setInterval(tick,650); $('#gsPlay').textContent='Pause'; } };
    const cv=$('#gsCv'); cv.addEventListener('pointermove', e=>{ const r=cv.getBoundingClientRect(); const s=r.width/NX; const i=Math.floor((e.clientX-r.left)/s), j=NY-1-Math.floor((e.clientY-r.top)/s); const id=(i>=0&&i<NX&&j>=0&&j<NY)?j*NX+i:-1; if(id!==S.hover){ S.hover=id; draw(); } });
    cv.addEventListener('pointerleave', ()=>{ S.hover=-1; draw(); });
  }
  return {init, draw};
})();

// =====================================================================
// E2c. Convergence chart (data from the Part 1 toy)
// =====================================================================
const CVG = (() => {
  let T=null;
  function draw(){
    if(!T) T=SIM.traces(60);
    const cv=$('#cvCv'); const {ctx,w,h}=fit(cv,.36); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const padL=46, padB=26, padT=10, padR=12; const N=T.gs.length-1;
    const lg=v=>Math.log10(Math.max(v,1e-6)); let ymax=-9, ymin=9; for(const k of ['gs','jac','nes']) for(const v of T[k]){ ymax=Math.max(ymax,lg(v)); ymin=Math.min(ymin,lg(v)); }
    ymax=Math.ceil(ymax); ymin=Math.max(-6,Math.floor(ymin));
    const X=i=>padL+i/N*(w-padL-padR), Y=v=>padT+(ymax-lg(v))/(ymax-ymin)*(h-padT-padB);
    ctx.strokeStyle=COL.grid; ctx.lineWidth=1; ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='right'; ctx.textBaseline='middle';
    for(let e=ymin;e<=ymax;e++){ const y=Y(10**e); ctx.beginPath(); ctx.moveTo(padL,y+.5); ctx.lineTo(w-padR,y+.5); ctx.stroke(); ctx.fillText('1e'+e, padL-6, y); }
    ctx.textAlign='center'; ctx.textBaseline='top'; for(let i=0;i<=N;i+=10){ ctx.fillText(String(i), X(i), h-padB+6); }
    ctx.fillText('iteration', (padL+w-padR)/2, h-12);
    const line=(arr,col,lw)=>{ ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.beginPath(); arr.forEach((v,i)=>{ i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v)); }); ctx.stroke(); };
    line(T.jac, COL.muted, 2); line(T.gs, COL.node, 2.4); line(T.nes, COL.strain, 2.4);
    const at=(a,i)=>a[Math.min(i,a.length-1)];
    $('#cvCap').textContent=`${T.cells} cells and ${T.contacts} wall nodes. Error after 20 iterations: Gauss\u2013Seidel ${at(T.gs,20).toExponential(1)}, Jacobi ${at(T.jac,20).toExponential(1)}, with Nesterov ${at(T.nes,20).toExponential(1)} (cells per second). Toy data: Newton's real problem is 3D with 6-number stresses, but the pattern is the same.`;
  }
  return {draw};
})();

// =====================================================================
// VBD chapter widgets
// =====================================================================
const sym2inv=(a,b,d)=>{ const det=a*d-b*b; return Math.abs(det)<1e-12?null:[d/det,-b/det,a/det]; };
function springFH(px,py,qx,qy,k,l0){ const dx=px-qx, dy=py-qy, l=Math.hypot(dx,dy)||1e-9, nx=dx/l, ny=dy/l; const fx=-k*(l-l0)*nx, fy=-k*(l-l0)*ny; const s=Math.max(0,1-l0/l);
  return {fx,fy,hxx:k*(nx*nx+s*(1-nx*nx)), hxy:k*(nx*ny-s*nx*ny), hyy:k*(ny*ny+s*(1-ny*ny))}; }

// ---------- V1: one vertex's Newton step ----------
const VL = (() => {
  const A=[-1.2,0], B=[1.25,.35], l0=1, k=1; const D0={x:[-.5,.75], y:[.25,-.95]};
  const S={x:D0.x.slice(), y:D0.y.slice(), w:1, drag:null, anim:0}; const X0=-2.3, X1=2.3, Y0=-1.7, Y1=1.7; let geo=null;
  const G=(x,y)=>{ let g=S.w/2*((x-S.y[0])**2+(y-S.y[1])**2); for(const n of [A,B]){ const l=Math.hypot(x-n[0],y-n[1]); g+=.5*k*(l-l0)**2; } return g; };
  function FH(x,y){ let fx=S.w*(S.y[0]-x), fy=S.w*(S.y[1]-y), a=S.w, b=0, d=S.w; for(const n of [A,B]){ const s=springFH(x,y,n[0],n[1],k,l0); fx+=s.fx; fy+=s.fy; a+=s.hxx; b+=s.hxy; d+=s.hyy; } return {fx,fy,a,b,d}; }
  function step(x,y){ const q=FH(x,y), inv=sym2inv(q.a,q.b,q.d); if(!inv) return [x,y]; return [x+inv[0]*q.fx+inv[1]*q.fy, y+inv[1]*q.fx+inv[2]*q.fy]; }
  function trueMin(){ let p=S.y.slice(); for(let i=0;i<80;i++){ const n=step(p[0],p[1]); const g0=G(p[0],p[1]); let t=1, c=n; while(G(p[0]+(c[0]-p[0])*t,p[1]+(c[1]-p[1])*t)>g0 && t>1e-3) t*=.5; p=[p[0]+(c[0]-p[0])*t, p[1]+(c[1]-p[1])*t]; } return p; }
  function draw(){
    const cv=$('#vlCv'); const {ctx,w,h}=fit(cv,.74); const sx=w/(X1-X0), sy=h/(Y1-Y0); const X=x=>(x-X0)*sx, Y=y=>(Y1-y)*sy; geo={X,Y,w,h,sx,sy};
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const mn=trueMin(), gmin=G(mn[0],mn[1]); const Bk=4;
    for(let py=0;py<h;py+=Bk) for(let px=0;px<w;px+=Bk){ const x=X0+(px+Bk/2)/sx, y=Y1-(py+Bk/2)/sy; const v=Math.log(1+(G(x,y)-gmin)*3); const band=Math.floor(v*3.2); const t=Math.max(0,1-v/3.2);
      ctx.fillStyle=rgba('strain', .04+.30*t*(band%2?1:.72)); ctx.fillRect(px,py,Bk,Bk); }
    ctx.strokeStyle=rgba('ink',.5); ctx.lineWidth=1.2; for(const n of [A,B]){ ctx.beginPath(); ctx.moveTo(X(n[0]),Y(n[1])); ctx.lineTo(X(S.x[0]),Y(S.x[1])); ctx.stroke(); }
    for(const n of [A,B]){ ctx.fillStyle=COL.ink; ctx.beginPath(); ctx.arc(X(n[0]),Y(n[1]),6,0,7); ctx.fill(); }
    ctx.setLineDash([4,3]); ctx.strokeStyle=COL.ink; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(X(S.y[0]),Y(S.y[1]),9,0,7); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=COL.ink; ctx.font=FONT_MONO; ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.fillText('y', X(S.y[0])+12, Y(S.y[1]));
    ctx.fillStyle=COL.strain; ctx.save(); ctx.translate(X(mn[0]),Y(mn[1])); ctx.rotate(Math.PI/4); ctx.fillRect(-6,-6,12,12); ctx.restore();
    const q=FH(S.x[0],S.x[1]), inv=sym2inv(q.a,q.b,q.d); const nx=step(S.x[0],S.x[1]);
    if(inv){ // ellipse of the quadratic model through x, centered at x+dx
      const tr=(q.a+q.d)/2, dd=Math.sqrt(((q.a-q.d)/2)**2+q.b*q.b), l1=tr+dd, l2=tr-dd; const ang=.5*Math.atan2(2*q.b,q.a-q.d);
      const ex=S.x[0]-nx[0], ey=S.x[1]-nx[1]; const lev=q.a*ex*ex+2*q.b*ex*ey+q.d*ey*ey; const r1=Math.sqrt(lev/Math.max(l1,1e-9)), r2=Math.sqrt(lev/Math.max(l2,1e-9));
      ctx.save(); ctx.translate(X(nx[0]),Y(nx[1])); ctx.scale(sx,-sy); ctx.rotate(ang); ctx.beginPath(); ctx.ellipse(0,0,r1,r2,0,0,7); ctx.restore(); ctx.strokeStyle=rgba('node',.8); ctx.lineWidth=1.5; ctx.setLineDash([6,4]); ctx.stroke(); ctx.setLineDash([]); }
    const fs=.25; arrow(ctx,X(S.x[0]),Y(S.x[1]),X(S.x[0]+q.fx*fs),Y(S.x[1]+q.fy*fs),COL.particle,2);
    arrow(ctx,X(S.x[0]),Y(S.x[1]),X(nx[0]),Y(nx[1]),COL.node,3);
    ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(X(S.x[0]),Y(S.x[1]),8,0,7); ctx.fill(); ctx.strokeStyle=COL.surface; ctx.lineWidth=2; ctx.stroke();
    const d0=Math.hypot(S.x[0]-mn[0],S.x[1]-mn[1]), d1=Math.hypot(nx[0]-mn[0],nx[1]-mn[1]);
    $('#vlRead').innerHTML=`<div><span>Distance to minimum</span><b>${d0.toFixed(3)}</b><em>before the step</em></div><div><span>After one step</span><b>${d1.toFixed(3)}</b><em>${d0>1e-6?((1-d1/d0)*100).toFixed(0)+'% closer':'at the minimum'}</em></div>`;
    $('#vlCap').textContent=`Orange arrow = force f (downhill, scaled). Teal = Newton step H\u207B\u00B9f. Dashed ellipse = the quadratic bowl H assumes. Press the button repeatedly: each step lands much closer.`;
  }
  function init(){
    const cv=$('#vlCv'); const to=e=>{ const r=cv.getBoundingClientRect(); return [X0+(e.clientX-r.left)/geo.sx, Y1-(e.clientY-r.top)/geo.sy]; };
    cv.addEventListener('pointerdown', e=>{ const p=to(e); const dx=Math.hypot(p[0]-S.x[0],p[1]-S.x[1]), dy=Math.hypot(p[0]-S.y[0],p[1]-S.y[1]); S.drag = dy<dx && dy<.35 ? 'y' : 'x'; cv.setPointerCapture(e.pointerId); S[S.drag]=p; draw(); });
    cv.addEventListener('pointermove', e=>{ if(!S.drag) return; S[S.drag]=to(e); draw(); }); cv.addEventListener('pointerup', ()=>{ S.drag=null; });
    motion.addEventListener('change', () => cancelAnimationFrame(S.anim));
    $('#vlStep').onclick=()=>{
      const next = step(S.x[0], S.x[1]);
      const start = S.x.slice();
      cancelAnimationFrame(S.anim);
      if (motion.matches) {
        S.x = next;
        draw();
        return;
      }
      let startTime;
      function move(time) {
        startTime ??= time;
        const progress = Math.min(1, (time - startTime) / 180);
        const blend = 1 - (1 - progress) ** 2;
        S.x = start.map((value, axis) => value + (next[axis] - value) * blend);
        draw();
        if (progress < 1) S.anim = requestAnimationFrame(move);
      }
      S.anim = requestAnimationFrame(move);
    };
    $('#vlReset').onclick=()=>{ cancelAnimationFrame(S.anim); S.x=D0.x.slice(); S.y=D0.y.slice(); draw(); };
    $('#vlW').oninput=e=>{ S.w=10**(+e.target.value); $('#oVlW').textContent=S.w.toFixed(S.w<1?2:1); draw(); };
  }
  return {init, draw};
})();

// ---------- V2: 2D VBD toy ----------
const VT = (() => {
  const h=1/60, g=-9.8, floorY=.35, kc=2e5;
  const S={scene:'rope', mode:'gs', iters:10, k:10**4.7, x:[], v:[], y:[], xp:[], pin:[], adj:[], col:[], ncol:3, springs:[], stage:0, it:0, c:0, playing:false, last:null, blown:false, steps:0};
  function build(){
    const x=[], pin=[], springs=[]; let ncol, col=[];
    if(S.scene==='rope'){ const n=26; for(let i=0;i<n;i++){ x.push([1.2+i*.3, 5.2]); pin.push(i===0); col.push(i%3); } ncol=3;
      for(let i=0;i<n-1;i++) springs.push([i,i+1,.3,1,true]); for(let i=0;i<n-2;i++) springs.push([i,i+2,.6,.04,false]); }
    else { const nx=13, ny=9, s=.42, x0=2.5, y0=5.4; const id=(i,j)=>j*nx+i;
      for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){ x.push([x0+i*s, y0-j*s]); pin.push(j===0 && (i===0||i===nx-1||i===6)); col.push((i%2)+2*(j%2)); } ncol=4;
      for(let j=0;j<ny;j++) for(let i=0;i<nx;i++){ if(i<nx-1) springs.push([id(i,j),id(i+1,j),s,1,true]); if(j<ny-1) springs.push([id(i,j),id(i,j+1),s,1,true]);
        if(i<nx-1&&j<ny-1){ springs.push([id(i,j),id(i+1,j+1),s*Math.SQRT2,.3,false]); springs.push([id(i+1,j),id(i,j+1),s*Math.SQRT2,.3,false]); } } }
    S.x=x; S.v=x.map(()=>[0,0]); S.pin=pin; S.springs=springs; S.col=col; S.ncol=ncol;
    S.adj=x.map(()=>[]); springs.forEach(([a,b,l,f])=>{ S.adj[a].push([b,l,f]); S.adj[b].push([a,l,f]); });
    S.stage=0; S.it=0; S.c=0; S.last=null; S.blown=false; S.steps=0; S.y=x.map(p=>p.slice()); S.xp=x.map(p=>p.slice());
  }
  function vertexStep(i, X){ const m=1, w=m/(h*h); const p=X[i]; let fx=w*(S.y[i][0]-p[0]), fy=w*(S.y[i][1]-p[1]), a=w, b=0, d=w;
    for(const [j,l0,f] of S.adj[i]){ const s=springFH(p[0],p[1],X[j][0],X[j][1],S.k*f,l0); fx+=s.fx; fy+=s.fy; a+=s.hxx; b+=s.hxy; d+=s.hyy; }
    if(p[1]<floorY){ fy+=kc*(floorY-p[1]); d+=kc; }
    const inv=sym2inv(a,b,d); if(!inv) return [0,0,fx,fy]; return [inv[0]*fx+inv[1]*fy, inv[1]*fx+inv[2]*fy, fx, fy]; }
  function predict(){ S.xp=S.x.map(p=>p.slice()); S.y=S.x.map((p,i)=>S.pin[i]?p.slice():[p[0]+h*S.v[i][0], p[1]+h*S.v[i][1]+h*h*g]); S.x=S.y.map(p=>p.slice()); S.it=0; S.c=0; }
  function sweepColor(c){ const moves=[]; const X = S.mode==='jac' ? S.x.map(p=>p.slice()) : S.x;
    const list = S.mode==='jac' ? S.x.map((_,i)=>i) : S.x.map((_,i)=>i).filter(i=>S.col[i]===c);
    const res=[]; for(const i of list){ if(S.pin[i]) continue; const r=vertexStep(i,X); res.push([i,r]); if(S.mode!=='jac'){ S.x[i][0]+=r[0]; S.x[i][1]+=r[1]; } }
    if(S.mode==='jac') for(const [i,r] of res){ S.x[i][0]+=r[0]; S.x[i][1]+=r[1]; }
    S.last={c, moves:res}; }
  function finalize(){ S.v=S.x.map((p,i)=>[(p[0]-S.xp[i][0])/h,(p[1]-S.xp[i][1])/h]); S.steps++;
    if(S.x.some(p=>!isFinite(p[0])||!isFinite(p[1])||Math.abs(p[0])>200||Math.abs(p[1])>200)){ S.blown=true; build(); S.blown=true; } }
  const passes=()=>S.mode==='jac'?1:S.ncol;
  function full(){ if(S.stage===0||S.stage==='done'){ predict(); } else if(S.stage==='pred'){ } for(;S.it<S.iters;S.it++){ for(;S.c<passes();S.c++) sweepColor(S.c); S.c=0; } finalize(); S.stage='done'; S.last=null; }
  function next(){ S.blown=false;
    if(S.stage===0||S.stage==='done'){ predict(); S.stage='pred'; S.last=null; return; }
    if(S.stage==='pred'||S.stage==='sweep'){ if(S.it>=S.iters){ finalize(); S.stage='done'; S.last=null; return; } sweepColor(S.c); S.stage='sweep'; S.c++; if(S.c>=passes()){ S.c=0; S.it++; } } }
  function maxStretch(){ let m=0; for(const [a,b,l0,,str] of S.springs){ if(!str) continue; const l=Math.hypot(S.x[a][0]-S.x[b][0],S.x[a][1]-S.x[b][1]); m=Math.max(m,l/l0-1); } return m; }
  function text(){ const el=$('#vtText'); let t, cls='';
    if(S.blown){ cls='floor'; t=`<b class="t">Blew up</b><span>Updating every vertex at once with stiff springs made neighbors overshoot each other until positions exploded. That is why VBD colors the vertices. The scene was reset.</span>`; }
    else if(S.stage===0) t=`<b class="t">Start</b><span>Positions and velocities from the last step. Press <b>Next</b>.</span>`;
    else if(S.stage==='pred'){ cls='cell'; t=`<b class="t">1 \u00B7 Predict</b><span>Each vertex gets its inertia target y = x + h v + h\u00B2 g (hollow rings): where it would go with no forces. Every vertex starts there. Pinned vertices stay put.</span>`; }
    else if(S.stage==='sweep'){ const it=S.c===0?S.it:S.it+1, cc=S.c===0?passes():S.c; cls=''; t=`<b class="t">2 \u00B7 Iteration ${it} of ${S.iters}, ${S.mode==='jac'?'all vertices at once':'color '+cc+' of '+S.ncol}</b><span>${S.mode==='jac'?'Every vertex computed its step from the same old positions, then all moved together.':'Only the highlighted vertices move. Each takes one Newton step H\u207B\u00B9f with its neighbors frozen. None of them share a spring, so they can all run in parallel.'} Teal arrows show the moves, scaled up.</span>`; }
    else { cls='ok'; t=`<b class="t">3 \u00B7 Finalize</b><span>All ${S.iters} iterations done. New velocity = (x \u2212 x_prev)/h. Max stretch is ${(maxStretch()*100).toFixed(2)}%: that is how far the result is from rest length under gravity, plus whatever VBD did not converge.</span>`; }
    el.className='zfix '+cls; el.innerHTML=t; }
  function draw(){
    const cv=$('#vtCv'); const {ctx,w,h:H}=fit(cv,.6); const W=10, Hh=6, s=w/W; const X=x=>x*s, Y=y=>H-y*s;
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,H);
    ctx.fillStyle=rgba('collider',.12); ctx.fillRect(0,Y(floorY),w,H-Y(floorY)); ctx.strokeStyle=COL.collider; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(0,Y(floorY)); ctx.lineTo(w,Y(floorY)); ctx.stroke();
    for(const [a,b,l0,f,str] of S.springs){ if(!str && S.scene==='net') continue; const l=Math.hypot(S.x[a][0]-S.x[b][0],S.x[a][1]-S.x[b][1]); const e=Math.min(1,Math.max(0,(l/l0-1)*12));
      ctx.strokeStyle=str? `rgba(${mix('line','strain',e).map(v=>v|0).join(',')},1)` : rgba('muted',.25); ctx.lineWidth=str?2:1; ctx.beginPath(); ctx.moveTo(X(S.x[a][0]),Y(S.x[a][1])); ctx.lineTo(X(S.x[b][0]),Y(S.x[b][1])); ctx.stroke(); }
    const stepView = !S.playing && (S.stage==='pred'||S.stage==='sweep');
    if(stepView){ ctx.strokeStyle=rgba('ink',.45); ctx.lineWidth=1; for(let i=0;i<S.y.length;i++){ ctx.beginPath(); ctx.arc(X(S.y[i][0]),Y(S.y[i][1]),4,0,7); ctx.stroke(); } }
    const curC = S.stage==='sweep' && S.last ? S.last.c : -1;
    S.x.forEach((p,i)=>{ const showCol = stepView || S.stage==='done' && !S.playing; let fill = showCol ? catColor(S.col[i],S.ncol, (S.mode==='jac'||curC<0||S.col[i]===curC)?1:.3) : COL.particle;
      if(S.pin[i]) fill=COL.ink; ctx.fillStyle=fill; ctx.beginPath(); ctx.arc(X(p[0]),Y(p[1]),S.scene==='rope'?5:4,0,7); ctx.fill(); });
    if(stepView && S.last){ let mx=1e-9; for(const [,r] of S.last.moves) mx=Math.max(mx,Math.hypot(r[0],r[1])); const sc=Math.min(400, 26/(mx*s)); for(const [i,r] of S.last.moves){ const p=S.x[i]; arrow(ctx,X(p[0]-r[0]),Y(p[1]-r[1]),X(p[0]-r[0]+r[0]*sc),Y(p[1]-r[1]+r[1]*sc),COL.node,1.8); } }
    $('#vtStats').innerHTML=`<span>step <b>${S.steps}</b></span><span>colors <b>${S.mode==='jac'?'none':S.ncol}</b></span><span>iterations <b>${S.iters}</b></span><span>max stretch <b>${(maxStretch()*100).toFixed(2)}%</b></span><span>vertex solves per step <b>${S.iters*S.x.length}</b></span>`;
    if(!S.playing) text(); else { $('#vtText').className='zfix'; $('#vtText').innerHTML=`<b class="t">Playing</b><span>${S.iters} iterations per step, ${S.mode==='jac'?'all vertices at once':S.ncol+' colors'}. Max stretch ${(maxStretch()*100).toFixed(2)}%.</span>`; }
  }
  let raf=0, lastT=0;
  function loop(t){ if(!S.playing) return; if(t-lastT>=1000/60){ lastT=t; S.stage='done'; full(); if(S.blown){ stop(); } draw(); } raf=requestAnimationFrame(loop); }
  function stop(){ S.playing=false; cancelAnimationFrame(raf); $('#vtPlay').textContent='Play'; draw(); }
  function init(){
    registerPlayback('vtPlay', () => { if (S.playing) stop(); });
    build();
    seg('#vtScene', v=>{ stop(); S.scene=v; build(); draw(); }); seg('#vtMode', v=>{ S.mode=v; if(S.stage==='sweep'||S.stage==='pred'){ S.c=0; } draw(); });
    $('#vtNext').onclick=()=>{ if(S.playing) stop(); next(); draw(); };
    $('#vtStep').onclick=()=>{ if(S.playing) stop(); S.blown=false; full(); draw(); };
    $('#vtReset').onclick=()=>{ stop(); build(); draw(); };
    $('#vtPlay').onclick=()=>{ if(S.playing) stop(); else { S.blown=false; S.playing=true; $('#vtPlay').textContent='Pause'; raf=requestAnimationFrame(loop); } };
    $('#vtIt').oninput=e=>{ S.iters=+e.target.value; $('#oVtIt').textContent=S.iters; if(!S.playing) draw(); };
    const kSet=v=>{ S.k=10**v; $('#oVtK').textContent=S.k>=1e4?(S.k/1e3).toFixed(0)+'k':S.k.toFixed(0); }; kSet(4.7);
    $('#vtK').oninput=e=>{ kSet(+e.target.value); if(!S.playing) draw(); };
  }
  return {init, draw};
})();

// ---------- V3a: Neo-Hookean membrane triangle ----------
const VM = (() => {
  const R=[[0,0],[1.3,0],[.35,1.05]]; const S={x:R.map(p=>p.slice()), mu:1, la:3, drag:-1};
  const PRESET={rest:R, stretch:[[0,0],[1.3,0],[.4,1.5]], shear:[[0,0],[1.3,0],[.95,1.05]], squash:[[0,0],[1.05,0],[.3,.75]]};
  const Dm=[[R[1][0]-R[0][0], R[2][0]-R[0][0]],[R[1][1]-R[0][1], R[2][1]-R[0][1]]]; const dD=Dm[0][0]*Dm[1][1]-Dm[0][1]*Dm[1][0];
  const Di=[[Dm[1][1]/dD,-Dm[0][1]/dD],[-Dm[1][0]/dD,Dm[0][0]/dD]]; const area=Math.abs(dD)/2;
  const X0=-.5, X1=2.3, Y0=-.45, Y1=1.75; let geo=null;
  function calc(){ const x=S.x; const Ds=[[x[1][0]-x[0][0], x[2][0]-x[0][0]],[x[1][1]-x[0][1], x[2][1]-x[0][1]]];
    const F=[[Ds[0][0]*Di[0][0]+Ds[0][1]*Di[1][0], Ds[0][0]*Di[0][1]+Ds[0][1]*Di[1][1]],[Ds[1][0]*Di[0][0]+Ds[1][1]*Di[1][0], Ds[1][0]*Di[0][1]+Ds[1][1]*Di[1][1]]];
    const Ic=F[0][0]**2+F[0][1]**2+F[1][0]**2+F[1][1]**2, Jd=F[0][0]*F[1][1]-F[0][1]*F[1][0], J=Math.abs(Jd), sg=Jd>=0?1:-1;
    const lp=S.la+S.mu, al=lp>0?1+S.mu/lp:1; const cof=[[F[1][1],-F[1][0]],[-F[0][1],F[0][0]]];
    const c=lp*(J-al)*sg; const P=[[S.mu*F[0][0]+c*cof[0][0], S.mu*F[0][1]+c*cof[0][1]],[S.mu*F[1][0]+c*cof[1][0], S.mu*F[1][1]+c*cof[1][1]]];
    // forces = -area * P * Di^T
    const Hm=[[-(P[0][0]*Di[0][0]+P[0][1]*Di[0][1])*area, -(P[0][0]*Di[1][0]+P[0][1]*Di[1][1])*area],[-(P[1][0]*Di[0][0]+P[1][1]*Di[0][1])*area, -(P[1][0]*Di[1][0]+P[1][1]*Di[1][1])*area]];
    const f1=[Hm[0][0],Hm[1][0]], f2=[Hm[0][1],Hm[1][1]], f0=[-f1[0]-f2[0],-f1[1]-f2[1]];
    const eS=S.mu/2*(Ic-2), eA=lp/2*((J-al)**2-(1-al)**2);
    return {F,Ic,J,Jd,eS,eA,f:[f0,f1,f2]}; }
  function draw(){
    const cv=$('#vmCv'); const {ctx,w,h}=fit(cv,.72); const sx=w/(X1-X0), sy=h/(Y1-Y0), s=Math.min(sx,sy); const X=x=>(x-X0)*s+(w-(X1-X0)*s)/2, Y=y=>h-(y-Y0)*s-(h-(Y1-Y0)*s)/2; geo={X,Y,s,w,h};
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const q=calc();
    ctx.setLineDash([5,4]); ctx.strokeStyle=rgba('ink',.5); ctx.lineWidth=1.5; ctx.beginPath(); R.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1]))); ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
    const t=Math.max(-1,Math.min(1,(q.J-1)*2)); const c=mix('surface', t>=0?'strain':'node', .25+.5*Math.abs(t));
    ctx.fillStyle=`rgba(${c.map(v=>v|0).join(',')},.9)`; ctx.beginPath(); S.x.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1]))); ctx.closePath(); ctx.fill(); ctx.strokeStyle=COL.ink; ctx.lineWidth=2; ctx.stroke();
    let mf=1e-9; q.f.forEach(f=>mf=Math.max(mf,Math.hypot(f[0],f[1]))); const fsc=Math.min(.6/mf, .5);
    S.x.forEach((p,i)=>{ const f=q.f[i]; arrow(ctx,X(p[0]),Y(p[1]),X(p[0]+f[0]*fsc),Y(p[1]+f[1]*fsc),COL.particle,2.4); ctx.fillStyle=COL.ink; ctx.beginPath(); ctx.arc(X(p[0]),Y(p[1]),7,0,7); ctx.fill(); });
    const e=q.eS+q.eA;
    $('#vmRead').innerHTML=`<div><span>Stretch I_c</span><b>${q.Ic.toFixed(3)}</b><em>2 at rest</em></div><div><span>Area ratio J</span><b>${q.J.toFixed(3)}</b><em>${q.J>1.005?'bigger':q.J<.995?'smaller':'same area'}</em></div><div><span>Shape energy</span><b>${q.eS.toFixed(3)}</b><em>&#x3BC;/2 (I_c \u2212 2)</em></div><div><span>Area energy</span><b>${q.eA.toFixed(3)}</b><em>total ${e.toFixed(3)} (above rest)</em></div>`;
    $('#vmCap').textContent=`F = [[${q.F[0][0].toFixed(2)}, ${q.F[0][1].toFixed(2)}], [${q.F[1][0].toFixed(2)}, ${q.F[1][1].toFixed(2)}]]. Orange arrows: forces on the corners (scaled), always pulling back toward a rest-like shape. They sum to zero.`;
  }
  function init(){
    const cv=$('#vmCv'); const to=e=>{ const r=cv.getBoundingClientRect(); const g=geo; const px=e.clientX-r.left, py=e.clientY-r.top; return [(px-(g.w-(X1-X0)*g.s)/2)/g.s+X0, (g.h-py-(g.h-(Y1-Y0)*g.s)/2)/g.s+Y0]; };
    cv.addEventListener('pointerdown', e=>{ const p=to(e); let b=-1, bd=.25; S.x.forEach((q,i)=>{ const d=Math.hypot(q[0]-p[0],q[1]-p[1]); if(d<bd){bd=d;b=i;} }); S.drag=b; if(b>=0) cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', e=>{ if(S.drag<0) return; const p=to(e); S.x[S.drag]=[Math.max(X0,Math.min(X1,p[0])),Math.max(Y0,Math.min(Y1,p[1]))]; draw(); });
    cv.addEventListener('pointerup', ()=>{ S.drag=-1; });
    document.querySelectorAll('[data-vm]').forEach(b=>b.addEventListener('click',()=>{ S.x=PRESET[b.dataset.vm].map(p=>p.slice()); draw(); }));
    $('#vmMu').oninput=e=>{ S.mu=+e.target.value; $('#oVmMu').textContent=S.mu.toFixed(2); draw(); };
    $('#vmLa').oninput=e=>{ S.la=+e.target.value; $('#oVmLa').textContent=S.la.toFixed(1); draw(); };
  }
  return {init, draw};
})();

// ---------- V3b: bending hinge ----------
const VBE = (() => {
  const S={th:50, tr:0, ke:1};
  function draw(){
    const cv=$('#vbCv2'); const {ctx,w,h}=fit(cv,.62); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const cx=w*.3, cy=h*.42, L=Math.min(w*.24,h*.36); const th=S.th*Math.PI/180, tr=S.tr*Math.PI/180;
    // left wing fixed to the left; right wing rotated by theta from flat
    const wing=(a,col,lw,dash)=>{ ctx.setLineDash(dash||[]); ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+L*Math.cos(a),cy-L*Math.sin(a)); ctx.stroke(); ctx.setLineDash([]); };
    wing(Math.PI, COL.ink, 6); wing(tr, rgba('ink',.4), 3, [6,5]); wing(th, COL.strain, 6);
    ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(cx,cy,7,0,7); ctx.fill();
    ctx.strokeStyle=COL.node; ctx.lineWidth=2; ctx.beginPath(); const r=L*.35; if(th>=0) ctx.arc(cx,cy,r,-th,0); else ctx.arc(cx,cy,r,0,-th); ctx.stroke();
    const k=S.ke*1.0, E=.5*k*(th-tr)**2, tq=-k*(th-tr);
    const tip=[cx+L*Math.cos(th), cy-L*Math.sin(th)]; const dir=Math.sign(tq); const ta=th+dir*.35;
    if(Math.abs(tq)>1e-3){ ctx.strokeStyle=COL.particle; ctx.lineWidth=2.5; ctx.beginPath(); const rr=L*1.05; if(dir>0) ctx.arc(cx,cy,rr,-ta,-th); else ctx.arc(cx,cy,rr,-th,-ta); ctx.stroke();
      const ex=cx+L*1.05*Math.cos(ta), ey=cy-L*1.05*Math.sin(ta); arrow(ctx, cx+L*1.05*Math.cos(ta-dir*.05), cy-L*1.05*Math.sin(ta-dir*.05), ex, ey, COL.particle, 2.5); }
    ctx.font=FONT_MONO; ctx.fillStyle=COL.node; ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.fillText('\u03B8 = '+S.th+'\u00B0', cx+r+6, cy-r*.35*Math.sign(th||1));
    ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.fillText('shared edge (seen end-on)', cx, cy+30);
    // energy plot on the right
    const px0=w*.62, px1=w-12, py0=14, py1=h-28; const R2=Math.PI*160/180; const Emax=.5*k*(R2+Math.abs(tr))**2+1e-6;
    const PX=t=>px0+(t+R2)/(2*R2)*(px1-px0), PY=e=>py1-(e/Emax)*(py1-py0);
    ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(px0,py1); ctx.lineTo(px1,py1); ctx.moveTo(PX(0),py0); ctx.lineTo(PX(0),py1); ctx.stroke();
    ctx.strokeStyle=COL.strain; ctx.lineWidth=2; ctx.beginPath(); for(let i=0;i<=80;i++){ const t=-R2+2*R2*i/80; const e=.5*k*(t-tr)**2; i?ctx.lineTo(PX(t),PY(e)):ctx.moveTo(PX(t),PY(e)); } ctx.stroke();
    ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(PX(th),PY(E),5,0,7); ctx.fill();
    ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top'; ctx.fillText('energy vs \u03B8', (px0+px1)/2, py1+6);
    $('#vbCap2').textContent=`E = ${E.toFixed(3)}, restoring torque ${tq.toFixed(3)} (orange arrow). Violet = current wing, dashed = rest angle.`;
  }
  function init(){
    $('#vbTh').oninput=e=>{ S.th=+e.target.value; $('#oVbTh').textContent=S.th+'\u00B0'; draw(); };
    $('#vbTr').oninput=e=>{ S.tr=+e.target.value; $('#oVbTr').textContent=S.tr+'\u00B0'; draw(); };
    $('#vbKe').oninput=e=>{ S.ke=+e.target.value; $('#oVbKe').textContent=S.ke.toFixed(2); draw(); };
  }
  return {init, draw};
})();

// ---------- V3c: tet Neo-Hookean energy curve ----------
const VN = (() => {
  const S={mu:1, la:4, cmp:true};
  function draw(){
    const cv=$('#vnCv'); const {ctx,w,h}=fit(cv,.62); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const s0=-.7, s1=1.7, padL=36, padB=28, padT=10, padR=10; const lp=S.la+S.mu, al=1+S.mu/lp;
    const psi=s=>{ const J=s*s*s; return S.mu/2*(3*s*s-3) + lp/2*((J-al)**2-(1-al)**2); };
    const classic=s=>{ if(s<=0) return null; const J=s*s*s, lj=Math.log(J); return S.mu/2*(3*s*s-3) - S.mu*lj + S.la/2*lj*lj; };
    let ymax=0; for(let i=0;i<=100;i++){ const s=s0+(s1-s0)*i/100; ymax=Math.max(ymax,psi(s)); } ymax=Math.max(ymax,1);
    const X=s=>padL+(s-s0)/(s1-s0)*(w-padL-padR), Y=e=>padT+(1-Math.min(e,ymax*1.05)/(ymax*1.05))*(h-padT-padB);
    ctx.fillStyle=rgba('collider',.09); ctx.fillRect(X(s0),padT,X(0)-X(s0),h-padT-padB);
    ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(padL,Y(0)); ctx.lineTo(w-padR,Y(0)); ctx.moveTo(X(0),padT); ctx.lineTo(X(0),h-padB); ctx.moveTo(X(1),padT); ctx.lineTo(X(1),h-padB); ctx.stroke();
    if(S.cmp){ ctx.strokeStyle=COL.muted; ctx.lineWidth=2; ctx.setLineDash([5,4]); ctx.beginPath(); let pen=false; for(let i=0;i<=300;i++){ const s=s0+(s1-s0)*i/300; const e=classic(s); if(e===null){ pen=false; continue; } const y=Y(e); if(!pen){ ctx.moveTo(X(s),y); pen=true; } else ctx.lineTo(X(s),y); } ctx.stroke(); ctx.setLineDash([]); }
    ctx.strokeStyle=COL.strain; ctx.lineWidth=2.6; ctx.beginPath(); for(let i=0;i<=200;i++){ const s=s0+(s1-s0)*i/200; const y=Y(psi(s)); i?ctx.lineTo(X(s),y):ctx.moveTo(X(s),y); } ctx.stroke();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top';
    for(const s of [-.5,0,.5,1,1.5]) ctx.fillText(String(s), X(s), h-padB+4); ctx.fillText('scale s   (J = s\u00B3)', (padL+w-padR)/2, h-13);
    ctx.fillStyle=COL.collider; ctx.textAlign='left'; ctx.fillText('inverted', X(s0)+4, padT+2); ctx.fillStyle=COL.muted; ctx.fillText('rest', X(1)+4, padT+2);
    $('#vnCap').textContent=`Violet: stable Neo-Hookean, finite everywhere and smallest at s = 1. ${S.cmp?'Dashed: classic Neo-Hookean, which shoots to infinity as s \u2192 0 and does not exist for s < 0.':''} Energy at s = 0 (fully collapsed): ${psi(0).toFixed(2)}.`;
  }
  function init(){ $('#vnMu').oninput=e=>{ S.mu=+e.target.value; $('#oVnMu').textContent=S.mu.toFixed(2); draw(); }; $('#vnLa').oninput=e=>{ S.la=+e.target.value; $('#oVnLa').textContent=S.la.toFixed(1); draw(); }; $('#vnCmp').onchange=e=>{ S.cmp=e.target.checked; draw(); }; }
  return {init, draw};
})();

// ---------- V4: contact force + friction ----------
const VC = (() => {
  const S={r:.2, eps:.01}; const h=1/60;
  function chart(cv, xs, series, opts){ const {ctx,w,h:H}=fit(cv,.62); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,H); const padL=40,padB=26,padT=10,padR=10;
    const X=x=>padL+(x-opts.x0)/(opts.x1-opts.x0)*(w-padL-padR); const Y=opts.log ? (v=>padT+(Math.log10(opts.y1)-Math.log10(Math.max(v,opts.y0)))/(Math.log10(opts.y1)-Math.log10(opts.y0))*(H-padT-padB)) : (v=>padT+(opts.y1-v)/(opts.y1-opts.y0)*(H-padT-padB));
    (opts.bands||[]).forEach(([a,b,col,label])=>{ ctx.fillStyle=col; ctx.fillRect(X(a),padT,X(b)-X(a),H-padT-padB); ctx.fillStyle=COL.muted; ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='top'; if(label) ctx.fillText(label,(X(a)+X(b))/2,padT+2); });
    ctx.strokeStyle=COL.line; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(padL,H-padB); ctx.lineTo(w-padR,H-padB); ctx.moveTo(padL,padT); ctx.lineTo(padL,H-padB); ctx.stroke();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top'; (opts.xt||[]).forEach(([v,l])=>ctx.fillText(l,X(v),H-padB+4)); ctx.fillText(opts.xl,(padL+w-padR)/2,H-13);
    ctx.textAlign='right'; ctx.textBaseline='middle'; (opts.yt||[]).forEach(([v,l])=>ctx.fillText(l,padL-4,Y(v)));
    series.forEach(([f,col,lw,dash])=>{ ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.setLineDash(dash||[]); ctx.beginPath(); let pen=false; xs.forEach(x=>{ const v=f(x); if(v===null){pen=false;return;} const y=Math.max(padT,Math.min(H-padB,Y(v))); pen?ctx.lineTo(X(x),y):ctx.moveTo(X(x),y); pen=true; }); ctx.stroke(); ctx.setLineDash([]); }); }
  function draw(){
    const r=S.r, tau=r/2, k=1; const xs=[]; for(let i=1;i<=400;i++) xs.push(1.25*r*i/400);
    const barrier=d=> d>=r?null : d>=tau ? k*(r-d) : k*tau*tau/d; const quad=d=> d>=r?null:k*(r-d);
    chart($('#vcCv'), xs, [[quad,COL.muted,2,[5,4]],[barrier,COL.collider,2.6]], {x0:0,x1:1.25*r,y0:.01*r,y1:20*r,log:true,xl:'distance d (m)', xt:[[0,'0'],[tau,'r/2'],[r,'r']], yt:[[.1*r,'0.1'],[r,'1'],[10*r,'10']],
      bands:[[0,tau,rgba('collider',.10),'log barrier'],[tau,r,rgba('strain',.10),'soft spring'],[r,1.25*r,rgba('muted',.06),'free']]});
    $('#vcCap').textContent=`Push per unit stiffness, log scale. Dashed: a plain penalty spring, which stays finite even at d = 0, so a fast vertex can push through. Rose: Newton's barrier, which grows without limit as d \u2192 0 (r = ${r.toFixed(2)} m).`;
    const eu=S.eps*h, xs2=[]; for(let i=0;i<=300;i++) xs2.push(4*eu*i/300);
    const fr=u=> u>eu?1:(-u/eu+2)*u/eu; const coul=u=> u>0?1:0;
    chart($('#vfCv'), xs2, [[coul,COL.muted,2,[5,4]],[fr,COL.node,2.6]], {x0:0,x1:4*eu,y0:0,y1:1.15,xl:'slip |u| this step', xt:[[0,'0'],[eu,'\u03B5_u'],[2*eu,'2\u03B5_u'],[4*eu,'4\u03B5_u']], yt:[[0,'0'],[1,'\u03BC f_n']], bands:[[0,eu,rgba('node',.08),'sticks'],[eu,4*eu,rgba('muted',.05),'slides']]});
    $('#vfCap').textContent=`\u03B5_u = friction_epsilon \u00B7 h = ${(eu*1000).toFixed(3)} mm at h = 1/60 s. Dashed: ideal Coulomb, a jump at zero slip, which Newton\u2019s method cannot handle. Teal: smoothed ramp, which reaches full friction at \u03B5_u.`;
  }
  function init(){ $('#vcR').oninput=e=>{ S.r=+e.target.value; $('#oVcR').textContent=S.r.toFixed(2); draw(); }; $('#vfE').oninput=e=>{ S.eps=10**(+e.target.value); $('#oVfE').textContent=S.eps.toFixed(3); draw(); }; }
  return {init, draw};
})();

// ---------- V4b: Divide and Truncate ----------
const VD = (() => {
  const S={p:[.2,1.1], d:[.55,-1.6], drag:false}; const segA=[-1.4,0], segB=[1.6,0], Q=1.0; const X0=-1.8, X1=2, Y0=-.9, Y1=1.9; let geo=null;
  function draw(){
    const cv=$('#vdCv'); const {ctx,w,h}=fit(cv,.62); const s=Math.min(w/(X1-X0), h/(Y1-Y0)); const ox=(w-(X1-X0)*s)/2, oy=(h-(Y1-Y0)*s)/2; const X=x=>ox+(x-X0)*s, Y=y=>h-oy-(y-Y0)*s; geo={s,ox,oy,h};
    ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    ctx.fillStyle=rgba('collider',.10); ctx.fillRect(X(segA[0]),Y(0),X(segB[0])-X(segA[0]),Y(Y0)-Y(0));
    ctx.strokeStyle=COL.collider; ctx.lineWidth=4; ctx.beginPath(); ctx.moveTo(X(segA[0]),Y(0)); ctx.lineTo(X(segB[0]),Y(0)); ctx.stroke();
    const py=S.p[1]/2; ctx.setLineDash([7,5]); ctx.strokeStyle=COL.ink; ctx.lineWidth=1.5; ctx.beginPath(); ctx.moveTo(X(X0),Y(py)); ctx.lineTo(X(X1),Y(py)); ctx.stroke(); ctx.setLineDash([]);
    const maxd=.5*.85*Q; ctx.strokeStyle=rgba('node',.5); ctx.setLineDash([3,4]); ctx.beginPath(); ctx.arc(X(S.p[0]),Y(S.p[1]),maxd*s,0,7); ctx.stroke(); ctx.setLineDash([]);
    let t=1; if(S.d[1]<0){ t=Math.min(1,(S.p[1]-py)/(-S.d[1])); } let dx=S.d[0]*t, dy=S.d[1]*t; const L=Math.hypot(dx,dy); if(L>maxd){ dx*=maxd/L; dy*=maxd/L; }
    ctx.setLineDash([5,4]); arrow(ctx,X(S.p[0]),Y(S.p[1]),X(S.p[0]+S.d[0]),Y(S.p[1]+S.d[1]),rgba('muted',.9),2); ctx.setLineDash([]);
    arrow(ctx,X(S.p[0]),Y(S.p[1]),X(S.p[0]+dx),Y(S.p[1]+dy),COL.node,3.2);
    ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(X(S.p[0]),Y(S.p[1]),8,0,7); ctx.fill();
    ctx.strokeStyle=COL.ink; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(X(S.p[0]+S.d[0]),Y(S.p[1]+S.d[1]),7,0,7); ctx.stroke();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='left'; ctx.textBaseline='bottom'; ctx.fillText('dividing plane', X(X0)+6, Y(py)-3); ctx.fillStyle=COL.collider; ctx.textBaseline='top'; ctx.fillText('triangle (side view)', X(segA[0])+4, Y(0)+6);
    const crossed=S.p[1]+S.d[1]<0;
    $('#vdRead').innerHTML=`<div><span>Proposed move</span><b>${Math.hypot(S.d[0],S.d[1]).toFixed(2)}</b><em>${crossed?'would pass through!':'stays above'}</em></div><div><span>Allowed move</span><b>${Math.hypot(dx,dy).toFixed(2)}</b><em>t = ${t.toFixed(2)}, cap ${maxd.toFixed(3)}</em></div>`;
    $('#vdCap').textContent='Grey dashed: the Newton step the vertex wanted. Teal: what it may do. It stops at the dividing plane and inside the dotted max-move circle. The triangle gets the same rule from its side, so the two can never meet.';
  }
  function init(){ const cv=$('#vdCv'); const to=e=>{ const r=cv.getBoundingClientRect(); const g=geo; return [(e.clientX-r.left-g.ox)/g.s+X0, (g.h-(e.clientY-r.top)-g.oy)/g.s+Y0]; };
    cv.addEventListener('pointerdown', e=>{ S.drag=true; cv.setPointerCapture(e.pointerId); const q=to(e); S.d=[q[0]-S.p[0],q[1]-S.p[1]]; draw(); });
    cv.addEventListener('pointermove', e=>{ if(!S.drag) return; const q=to(e); S.d=[q[0]-S.p[0],q[1]-S.p[1]]; draw(); }); cv.addEventListener('pointerup', ()=>{ S.drag=false; }); }
  return {init, draw};
})();

// ---------- V5: compliant ALM series spring ----------
const VA = (() => {
  const S={K:1e5, rho:10**5.5, bil:true};
  const fmt=v=>v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(0)+'k':v.toFixed(0);
  function draw(){
    const rho = S.bil ? Math.max(S.rho, 9*S.K) : S.rho; const s=S.K/(S.K+rho), keff=S.K*rho/(S.K+rho);
    const cv=$('#vaCv'); const {ctx,w,h}=fit(cv,.62); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const y=h*.36, x0=24, x1=w-24; const lK=Math.log10(S.K), lR=Math.log10(rho); const wK=(x1-x0)*.5*(1-s)+(x1-x0)*.15, wR=(x1-x0)-wK-30;
    const spring=(a,b,col,label)=>{ ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(a,y); const n=10, seg=(b-a)/n; for(let i=0;i<=n;i++){ ctx.lineTo(a+i*seg, y+(i%2?-10:10)*(i>0&&i<n?1:0)); } ctx.stroke(); ctx.fillStyle=col; ctx.font=FONT_MONO; ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.fillText(label,(a+b)/2,y-16); };
    ctx.fillStyle=COL.ink; ctx.fillRect(x0-8,y-24,8,48);
    spring(x0, x0+wK, COL.strain, 'material K = '+fmt(S.K)); ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(x0+wK+15,y,8,0,7); ctx.fill();
    spring(x0+wK+30, x1, COL.node, 'penalty \u03C1 = '+fmt(rho)); ctx.fillStyle=COL.ink; ctx.fillRect(x1,y-24,8,48);
    const bar=(yy,frac,col,label)=>{ ctx.fillStyle=rgba('muted',.15); ctx.fillRect(x0,yy,x1-x0,14); ctx.fillStyle=col; ctx.fillRect(x0,yy,(x1-x0)*Math.max(0,Math.min(1,frac)),14); ctx.fillStyle=COL.ink; ctx.font=FONT_MONO; ctx.textAlign='left'; ctx.textBaseline='bottom'; ctx.fillText(label,x0,yy-3); };
    bar(h*.66, keff/S.K, COL.strain, `k_eff / K = ${(keff/S.K*100).toFixed(1)}%`); bar(h*.88, s, COL.node, `s = K/(K+\u03C1) = ${s.toFixed(3)} (share of \u03BB kept)`);
    $('#vaRead').innerHTML=`<div><span>k_eff</span><b>${fmt(keff)}</b><em>${(keff/S.K*100).toFixed(1)}% of K</em></div><div><span>\u03C1 used</span><b>${fmt(rho)}</b><em>${S.bil&&rho>S.rho?'raised to 9K':'as set'}</em></div>`;
    $('#vaCap').textContent='Two springs in series: the material spring K and the solver\u2019s penalty \u03C1. The softer one dominates. The multiplier \u03BB then adds back the force the series spring alone would miss.';
  }
  function init(){ $('#vaK').oninput=e=>{ S.K=10**(+e.target.value); $('#oVaK').textContent=fmt(S.K); draw(); }; $('#vaR').oninput=e=>{ S.rho=10**(+e.target.value); $('#oVaR').textContent=fmt(S.rho); draw(); }; $('#vaBil').onchange=e=>{ S.bil=e.target.checked; draw(); };
    $('#oVaK').textContent=fmt(S.K); $('#oVaR').textContent=fmt(S.rho); }
  return {init, draw};
})();

// =====================================================================
// Playback shared by the Isaac Lab demos. Manual steps use the same updates.
function bindPlayback(id, update) {
  const button = document.getElementById(id);
  let frame = 0;
  let last = 0;

  function tick(time) {
    const elapsed = last ? Math.min(.05, (time - last) / 1000) : 0;
    last = time;
    update(elapsed);
    frame = requestAnimationFrame(tick);
  }

  function stop() {
    cancelAnimationFrame(frame);
    frame = 0;
    last = 0;
    button.textContent = 'Play';
  }

  registerPlayback(id, stop);
  button.onclick = () => {
    if (frame) stop();
    else if (!motion.matches) {
      button.textContent = 'Pause';
      frame = requestAnimationFrame(tick);
    }
  };
  return stop;
}

// ---------- L1: three clocks ----------
const LT = (() => {
  const S={dt:.005, dec:4, sub:2, ep:20, ph:0};
  function draw(){
    const cv=$('#ltCv'); const {ctx,w,h}=fit(cv,.3); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const padL=78, padR=10, rows=[['policy',COL.strain],['physics',COL.node],['solver',COL.collider]], rh=(h-30)/3;
    const span=2*S.dec*S.dt; const X=t=>padL+t/span*(w-padL-padR); const now=S.ph*span;
    ctx.font=FONT_MONO; ctx.textBaseline='middle'; ctx.textAlign='right';
    rows.forEach(([name,col],r)=>{ const y=8+r*rh; ctx.fillStyle=COL.muted; ctx.fillText(name, padL-8, y+rh/2-4);
      const n = r===0?2 : r===1?2*S.dec : 2*S.dec*S.sub; const seg=span/n;
      for(let i=0;i<n;i++){ const x0=X(i*seg)+1, x1=X((i+1)*seg)-1; const act= now>=i*seg && now<(i+1)*seg; ctx.fillStyle = act ? col : rgba(r===0?'strain':r===1?'node':'collider', .18); ctx.fillRect(x0, y+4, Math.max(1,x1-x0), rh-16); } });
    ctx.strokeStyle=COL.ink; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(X(now),4); ctx.lineTo(X(now),h-22); ctx.stroke();
    ctx.fillStyle=COL.muted; ctx.textAlign='center'; ctx.textBaseline='top'; for(let k=0;k<=2*S.dec;k++){ ctx.fillText((k*S.dt*1000).toFixed(0)+'ms', X(k*S.dt), h-18); if(2*S.dec>10 && k%2) continue; }
    const pol=1/(S.dt*S.dec), steps=Math.ceil(S.ep/(S.dt*S.dec));
    $('#ltStats').innerHTML=`<span>policy <b>${pol.toFixed(1)} Hz</b></span><span>physics <b>${(1/S.dt).toFixed(0)} Hz</b></span><span>solver <b>${(S.sub/S.dt).toFixed(0)} Hz</b> (dt ${(S.dt/S.sub*1000).toFixed(2)} ms)</span><span>solver steps per action <b>${S.dec*S.sub}</b></span><span>policy steps per episode <b>${steps}</b></span>`;
  }
  function init(){
    const bind=(id,o,k,f,fmt)=>{ $('#'+id).oninput=e=>{ S[k]=f(+e.target.value); $('#'+o).textContent=fmt(+e.target.value); draw(); }; };
    bind('ltDt','oLtDt','dt',v=>v/1000,v=>v.toFixed(1)); bind('ltDec','oLtDec','dec',v=>v,v=>v); bind('ltSub','oLtSub','sub',v=>v,v=>v); bind('ltEp','oLtEp','ep',v=>v,v=>v);
    const stop = bindPlayback('ltPlay', dt => { S.ph=(S.ph+dt/3)%1; draw(); });
    $('#ltStep').onclick = () => {
      stop();
      const count = 2 * S.dec * S.sub;
      S.ph = ((Math.floor(S.ph * count + 1e-9) + 1) % count) / count;
      draw();
    };
  }
  return {init, draw};
})();

// ---------- L2: env.step flow + 4096 envs ----------
const LF = (() => {
  const ST=[
    ['process_action','Action manager: raw action \u00D7 0.5 + default joint positions = joint targets for all 4,096 robots.',0],
    ['apply_action','Each action term pushes its targets to the robot (set_joint_position_target).',1],
    ['write_data_to_sim','Actuators turn targets into drive settings or torques and write them into Newton\u2019s Control.',1],
    ['sim.step','Newton: collide, actuators, num_substeps solver steps. One CUDA graph.',1],
    ['scene.update','Sensors and asset data refresh from Newton\u2019s new State (views, no copies).',1],
    ['terminations','Termination manager: time_out (20 s) or base_contact (body touched the ground) \u2192 done flags.',0],
    ['rewards','Reward manager: weighted sum of terms \u00D7 step_dt (velocity tracking, torques, air time\u2026).',0],
    ['reset done envs','Only the robots that finished: curriculum, scene reset, reset events (pose, joints), managers reset.',0],
    ['commands','Command manager: update target velocities, resample every 10 s.',0],
    ['interval events','Event manager: e.g. push_robot every 10\u201315 s.',0],
    ['observations','Observation manager: compute terms, add noise, return obs for the policy.',0],
  ];
  const S={cur:0, loop:0, t:0, dec:4};
  const N=4096, GW=64; const ep=new Float32Array(N), flash=new Float32Array(N); const r=rng(9); for(let i=0;i<N;i++) ep[i]=Math.floor(r()*1000);
  let resets=0, steps=0;
  function envStep(){ resets=0; for(let i=0;i<N;i++){ ep[i]+=1; const fall=r()<1/350; if(fall||ep[i]>=1000){ ep[i]=0; flash[i]=1; resets++; } } steps++; }
  function advance(){ const [,,inLoop]=ST[S.cur]; if(inLoop && S.cur===4 && S.loop<S.dec-1){ S.loop++; S.cur=1; } else { if(S.cur===4) S.loop=0; S.cur=(S.cur+1)%ST.length; if(S.cur===7) envStep(); } }
  function draw(){
    const cv=$('#lfCv'); const {ctx,w,h}=fit(cv,.78); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const bx=16, bw=w-32, n=ST.length, gap=6, bh=(h-16-gap*(n-1))/n;
    ctx.fillStyle=rgba('node',.07); const ly=8+1*(bh+gap)-3, lh=4*(bh+gap)+0; ctx.fillRect(bx-8,ly,bw+16,lh); ctx.strokeStyle=rgba('node',.5); ctx.setLineDash([5,4]); ctx.strokeRect(bx-8,ly,bw+16,lh); ctx.setLineDash([]);
    ctx.font=FONT_MONO; ctx.fillStyle=COL.node; ctx.textAlign='right'; ctx.textBaseline='top'; ctx.fillText(`decimation loop  ${S.cur>=1&&S.cur<=4?S.loop+1:'-'} / ${S.dec}`, bx+bw+4, ly+3);
    ST.forEach(([name],i)=>{ const y=8+i*(bh+gap); const act=i===S.cur; const col = i===0||i>=5&&i!==10 ? (i===10?'strain':'node') : (i===3?'collider':'particle');
      ctx.fillStyle = act ? rgba(col,.9) : rgba(col,.14); ctx.fillRect(bx+(i>=1&&i<=4?14:0), y, bw-(i>=1&&i<=4?28:0), bh);
      ctx.fillStyle = act ? COL.surface : COL.ink; ctx.font=`600 ${Math.max(11,Math.min(14,bh*.45))}px 'JetBrains Mono', ui-monospace, monospace`; ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.fillText(`${i+1}. ${name}`, bx+(i>=1&&i<=4?26:12), y+bh/2); });
    $('#lfCap').textContent=ST[S.cur][1];
    drawGrid();
  }
  function drawGrid(){
    const cv=$('#lgCv'); const {ctx,w,h}=fit(cv,1); const c=w/GW; ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    for(let i=0;i<N;i++){ const x=(i%GW)*c, y=((i/GW)|0)*c; if(flash[i]>0){ ctx.fillStyle=rgba('collider',.35+.65*flash[i]); } else { const p=ep[i]/1000; const m=mix('surface','node',.15+.7*p); ctx.fillStyle=`rgb(${m[0]|0},${m[1]|0},${m[2]|0})`; } ctx.fillRect(x+.5,y+.5,Math.max(1,c-1),Math.max(1,c-1)); }
    let mean=0; for(let i=0;i<N;i++) mean+=ep[i]; mean/=N;
    $('#lgStats').innerHTML=`<span>env steps <b>${steps}</b></span><span>reset this step <b>${resets}</b></span><span>mean episode progress <b>${(mean/50).toFixed(1)} s</b></span><span>darker = longer into the episode</span>`;
  }
  function init(){
    const stop = bindPlayback('lfPlay', dt => {
      S.t += dt;
      if (S.t < .55) return;
      S.t = 0;
      flash.fill(0);
      advance();
      draw();
    });
    $('#lfStep').onclick = () => { stop(); flash.fill(0); advance(); draw(); };
  }
  return {init, draw};
})();

// ---------- L2b: PPO advantages (GAE) ----------
const LP = (() => {
  const T=24; const S={g:.99, l:.95, r:[], vn:[], done:new Array(T).fill(false), seed:3};
  function gen(){ const q=rng(S.seed); S.r=[]; S.vn=[]; for(let t=0;t<=T;t++){ S.r.push(1+.35*(q()-.5)+.25*Math.sin(t*.7)); S.vn.push(1+.25*(q()-.5)); } S.done=new Array(T).fill(false); S.done[15]=true; }
  function calc(){ const base=.98/(1-S.g); const V=S.vn.map(x=>base*x*.0+base+ (x-1)*base*.05); const d=[], A=new Array(T).fill(0);
    for(let t=0;t<T;t++){ const nd=S.done[t]?0:1; d.push(S.r[t] + S.g*nd*V[t+1] - V[t]); }
    let a=0; for(let t=T-1;t>=0;t--){ const nd=S.done[t]?0:1; a=d[t]+S.g*S.l*nd*a; A[t]=a; } return {V,d,A}; }
  let geo=null;
  function draw(){
    const cv=$('#lpCv'); const {ctx,w,h}=fit(cv,.62); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const {V,d,A}=calc(); const padL=38, padR=8, top=10, mid=h*.62, bot=h-24; const cw=(w-padL-padR)/T; geo={padL,cw};
    const rmax=1.6; ctx.font=FONT_MONO;
    for(let t=0;t<T;t++){ const x=padL+t*cw; if(S.done[t]){ ctx.fillStyle=rgba('collider',.12); ctx.fillRect(x,top,cw,bot-top); }
      const rh=(S.r[t]/rmax)*(mid*.42); ctx.fillStyle=rgba('muted',.5); ctx.fillRect(x+cw*.2, top+mid*.45-rh, cw*.6, rh); }
    ctx.fillStyle=COL.muted; ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText('reward r_t', 4, top);
    let am=1e-6; A.forEach(a=>am=Math.max(am,Math.abs(a))); const zy=(mid+bot)/2+8, sc=(bot-mid)/2/am;
    ctx.strokeStyle=COL.line; ctx.beginPath(); ctx.moveTo(padL,zy); ctx.lineTo(w-padR,zy); ctx.stroke();
    for(let t=0;t<T;t++){ const x=padL+t*cw; const a=A[t]; ctx.fillStyle = a>=0?COL.node:COL.particle; const hh=a*sc; ctx.fillRect(x+cw*.15, a>=0?zy-hh:zy, cw*.7, Math.max(1,Math.abs(hh))); }
    ctx.fillStyle=COL.muted; ctx.fillText('advantage A_t', 4, mid-4);
    for(let t=0;t<T;t++){ if(S.done[t]){ ctx.fillStyle=COL.collider; ctx.textAlign='center'; ctx.fillText('fall', padL+(t+.5)*cw, top); } }
    ctx.textAlign='center'; ctx.fillStyle=COL.muted; for(let t=0;t<T;t+=4) ctx.fillText(String(t), padL+(t+.5)*cw, bot+6);
    const fi=S.done.indexOf(true);
    $('#lpCap').textContent=`Teal = better than expected, orange = worse. V \u2248 ${(0.98/(1-S.g)).toFixed(0)} (value of a typical state at \u03B3 = ${S.g}). ${fi>=0?`A fall at step ${fi}: its \u03B4 = r \u2212 V \u2248 \u2212${(0.98/(1-S.g)).toFixed(0)}, so the steps leading up to it get negative advantages that fade by \u03B3\u03BB = ${(S.g*S.l).toFixed(3)} per step back.`:'No fall in this rollout: advantages are just noise around zero.'} Click a column to toggle a fall.`;
  }
  function init(){ gen();
    $('#lpG').oninput=e=>{ S.g=+e.target.value; $('#oLpG').textContent=S.g.toFixed(3); draw(); };
    $('#lpL').oninput=e=>{ S.l=+e.target.value; $('#oLpL').textContent=S.l.toFixed(2); draw(); };
    $('#lpNew').onclick=()=>{ S.seed++; gen(); S.done[15]=false; S.done[6+Math.floor(rng(S.seed*7)()*16)]=true; draw(); };
    $('#lpCv').addEventListener('click', e=>{ const r=e.currentTarget.getBoundingClientRect(); const t=Math.floor((e.clientX-r.left-geo.padL)/geo.cw); if(t>=0&&t<T){ S.done[t]=!S.done[t]; draw(); } });
  }
  return {init, draw};
})();

// ---------- L3: one actuated joint ----------
const LA = (() => {
  const m=2, L=.5, I=m*L*L/3, gq=m*9.81*L/2, sat=120, vmax=7.5;
  const S={type:'imp', kp:10**1.6, kd:10**.7, lim:80, dt:.005, q:.3, v:0, tgt:1.1, hist:[], acc:0, tAuto:0, auto:false, tau:0, clip:false, unstable:false, time:0, fr:0};
  function physStep(){ const dt=S.dt, e=S.tgt-S.q, g=gq*Math.sin(S.q); let tau;
    if(S.type==='imp'){ const vn=(I*S.v + dt*(S.kp*e - g))/(I + dt*dt*S.kp + dt*S.kd); tau=S.kp*(e-dt*vn)+S.kd*(0-vn); S.clip=Math.abs(tau)>S.lim;
      if(S.clip){ tau=Math.sign(tau)*S.lim; S.v+=dt*(tau-g)/I; } else S.v=vn; }
    else { const pd=S.kp*e + S.kd*(0-S.v); if(S.type==='pd'){ tau=Math.max(-S.lim,Math.min(S.lim,pd)); }
      else { const v=Math.max(-vmax*(1+S.lim/sat), Math.min(vmax*(1+S.lim/sat), S.v)); const tmax=Math.min(S.lim, sat*(1-v/vmax)), tmin=Math.max(-S.lim, sat*(-1-v/vmax)); tau=Math.max(tmin,Math.min(tmax,pd)); }
      S.clip=Math.abs(tau-pd)>1e-6; S.v+=dt*(tau-g)/I; }
    if(Math.sign(tau)!==Math.sign(S.tau)&&Math.abs(tau)>.5*S.lim) S.fr+=(1-S.fr)*.05; else S.fr*=.97;
    S.q+=dt*S.v; S.tau=tau; S.time+=dt;
    if(!isFinite(S.q)||Math.abs(S.v)>400){ S.unstable=true; S.q=.3; S.v=0; S.hist=[]; }
    S.hist.push([S.time,S.q,S.tgt,tau]); while(S.hist.length && S.hist[0][0]<S.time-3) S.hist.shift(); }
  function tick(dt){ S.acc+=dt; let n=0; while(S.acc>=S.dt && n<4000){ physStep(); S.acc-=S.dt; n++; } if(S.auto){ S.tAuto+=dt; if(S.tAuto>1.2){ S.tAuto=0; S.tgt=(Math.random()*2-1)*1.4; } } draw(); }
  function draw(){
    const cv=$('#laCv'); const {ctx,w,h}=fit(cv,.5); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const px=w/2, py=h*.42, R=Math.min(w,h)*.42; const tip=a=>[px+R*Math.sin(a), py+R*Math.cos(a)];
    ctx.setLineDash([6,5]); ctx.strokeStyle=rgba('ink',.45); ctx.lineWidth=3; const tg=tip(S.tgt); ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(tg[0],tg[1]); ctx.stroke(); ctx.setLineDash([]);
    const tp=tip(S.q); ctx.strokeStyle=COL.particle; ctx.lineWidth=10; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(tp[0],tp[1]); ctx.stroke(); ctx.lineCap='butt';
    ctx.fillStyle=COL.particle; ctx.beginPath(); ctx.arc(tp[0],tp[1],14,0,7); ctx.fill();
    const f=Math.max(-1,Math.min(1,S.tau/S.lim)); ctx.strokeStyle=S.clip?COL.collider:COL.node; ctx.lineWidth=5; ctx.beginPath(); const a0=-Math.PI/2; ctx.arc(px,py,30, f>=0?a0:a0+f*Math.PI, f>=0?a0+f*Math.PI:a0); ctx.stroke();
    ctx.fillStyle=COL.ink; ctx.beginPath(); ctx.arc(px,py,8,0,7); ctx.fill();
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText(`q ${S.q.toFixed(2)} rad   target ${S.tgt.toFixed(2)}   torque ${S.tau.toFixed(1)} N\u00B7m`, 8, 6);
    const pc=$('#laPlot'); const P=fit(pc,.32), c2=P.ctx; c2.fillStyle=COL.surface; c2.fillRect(0,0,P.w,P.h);
    if(S.hist.length>1){ const t0=S.time-3, X=t=>(t-t0)/3*P.w, Yq=q=>P.h/2-q/1.8*(P.h/2-6), Yt=t=>P.h/2-t/Math.max(S.lim,1)*(P.h/2-6);
      c2.strokeStyle=COL.line; c2.beginPath(); c2.moveTo(0,P.h/2); c2.lineTo(P.w,P.h/2); c2.stroke();
      const line=(k,Y,col,lw,dash)=>{ c2.strokeStyle=col; c2.lineWidth=lw; c2.setLineDash(dash||[]); c2.beginPath(); S.hist.forEach((p,i)=>{ i?c2.lineTo(X(p[0]),Y(p[k])):c2.moveTo(X(p[0]),Y(p[k])); }); c2.stroke(); c2.setLineDash([]); };
      line(3,Yt,rgba('node',.7),1.5); line(2,Yq,COL.ink,1.5,[5,4]); line(1,Yq,COL.particle,2.4);
      c2.font=FONT_MONO; c2.fillStyle=COL.muted; c2.textAlign='left'; c2.textBaseline='top'; c2.fillText('last 3 s: angle (orange) vs target (dashed), torque (teal, \u00B1 effort limit)', 6, 4); }
    const crit = 2/Math.sqrt(S.kp/I);
    const msg = S.unstable ? ['floor','Unstable',`The explicit torque overshot every step and the joint blew up (reset). With dt = ${(S.dt*1000).toFixed(1)} ms and these gains, an explicit PD needs dt below about ${(Math.min(crit, 2*I/Math.max(S.kd,1e-9))*1000).toFixed(1)} ms. Implicit drives are stable with any gains.`]
      : (S.type!=='imp' && S.fr>.3) ? ['floor','Chattering',`The explicit PD overshoots within one ${(S.dt*1000).toFixed(1)} ms step, so the torque flips between +limit and \u2212limit every step. Only the effort clamp keeps it bounded. Lower the gains or the physics dt (explicit PD needs dt below about ${(Math.min(crit, 2*I/Math.max(S.kd,1e-9))*1000).toFixed(1)} ms here), or use an implicit drive.`]
      : S.clip ? ['cell','Saturated',S.type==='dc'?'The DC-motor envelope limits torque: at speed q\u0307 it allows at most 120\u00B7(1 \u2212 q\u0307/7.5), capped at the effort limit.':'The PD law asks for more torque than the effort limit, so the torque is clipped and the joint moves at its torque limit.']
      : ['ok','Tracking',S.type==='imp'?'The solver applies the PD drive implicitly, using the end-of-step position and velocity, so even stiff gains stay stable.':'Explicit PD: torque computed from the current error, held for the whole physics step.'];
    const el=$('#laTxt'); el.className='zfix '+msg[0]; el.innerHTML=`<b class="t">${msg[1]}</b><span>${msg[2]}</span>`;
    $('#laCap').textContent=`Link: 2 kg, 0.5 m, inertia ${I.toFixed(3)} kg\u00B7m\u00B2, gravity torque up to ${gq.toFixed(1)} N\u00B7m. Physics at ${(1/S.dt).toFixed(0)} Hz.`;
  }
  function init(){
    seg('#laType', v=>{ S.type=v; S.unstable=false; draw(); });
    const fmt=v=>v>=100?v.toFixed(0):v>=10?v.toFixed(1):v.toFixed(2);
    const bind=(id,o,set,show)=>{ const el=$('#'+id); const f=()=>{ set(+el.value); $('#'+o).textContent=show(); S.unstable=false; S.fr=0; }; el.oninput=()=>{ f(); draw(); }; f(); };
    bind('laKp','oLaKp',v=>S.kp=10**v,()=>fmt(S.kp)); bind('laKd','oLaKd',v=>S.kd=10**v,()=>fmt(S.kd)); bind('laTau','oLaTau',v=>S.lim=v,()=>S.lim.toFixed(0)); bind('laDt','oLaDt',v=>S.dt=v/1000,()=>(S.dt*1000).toFixed(1));
    const stop = bindPlayback('laPlay', tick);
    $('#laJump').onclick=()=>{ S.tgt=(Math.random()*2-1)*1.4; S.unstable=false; draw(); };
    $('#laStep').onclick=()=>{ stop(); tick(.2); };
    $('#laAuto').onchange=e=>{ S.auto=e.target.checked; };
  }
  return {init, draw};
})();

// ---------- L4: height scan ----------
const LS = (() => {
  const S={x:0, noise:true, box:false, bx:9, t:0}; const terr=x=>.18*Math.sin(x*1.3)+.1*Math.sin(x*3.1+1)+(Math.floor(x/2.2)%3===0?.12:0);
  function hitZ(x){ let z=terr(x); if(S.box && Math.abs(x-S.bx)<.3) z=Math.max(z, terr(S.bx)+.55); return z; }
  function draw(){
    const cv=$('#lsCv'); const {ctx,w,h}=fit(cv,.5); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const view=4, x0=S.x-view/2, sc=w/view, Y=z=>h*.78-z*sc*.9, X=x=>(x-x0)*sc;
    ctx.fillStyle=rgba('muted',.18); ctx.beginPath(); ctx.moveTo(0,h); for(let px=0;px<=w;px+=3){ ctx.lineTo(px,Y(terr(x0+px/sc))); } ctx.lineTo(w,h); ctx.closePath(); ctx.fill();
    ctx.strokeStyle=COL.ink; ctx.lineWidth=1.5; ctx.beginPath(); for(let px=0;px<=w;px+=3){ const y=Y(terr(x0+px/sc)); px?ctx.lineTo(px,y):ctx.moveTo(px,y); } ctx.stroke();
    if(S.box){ ctx.fillStyle=rgba('collider',.5); const bz=terr(S.bx); ctx.fillRect(X(S.bx-.3),Y(bz+.55),.6*sc,.55*sc*.9); }
    const bz=terr(S.x)+.55; ctx.fillStyle=COL.particle; ctx.fillRect(X(S.x-.45),Y(bz+.12),.9*sc,.24*sc*.9);
    for(const lx of [-.3,.3]){ ctx.strokeStyle=COL.particle; ctx.lineWidth=4; const ph=Math.sin(S.t*6+(lx>0?Math.PI:0)); ctx.beginPath(); ctx.moveTo(X(S.x+lx),Y(bz)); ctx.lineTo(X(S.x+lx+.1*ph),Y(terr(S.x+lx+.1*ph))); ctx.stroke(); }
    const vals=[]; const q=rng(Math.floor(S.t*20));
    for(let i=-8;i<=8;i++){ const rx=S.x+i*.1, hz=hitZ(rx); ctx.strokeStyle=rgba('node',.35); ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(X(rx),0); ctx.lineTo(X(rx),Y(hz)); ctx.stroke();
      ctx.fillStyle=COL.node; ctx.beginPath(); ctx.arc(X(rx),Y(hz),3,0,7); ctx.fill(); let v=bz-hz-.5; if(S.noise) v+=(q()*2-1)*.1; vals.push(Math.max(-1,Math.min(1,v))); }
    const sx=w*.05, sw=w*.9, sy=6, sh=26, cwid=sw/vals.length; vals.forEach((v,i)=>{ const c=mix('surface', v>=0?'node':'particle', Math.min(1,Math.abs(v))*.9); ctx.fillStyle=`rgb(${c.map(z=>z|0).join(',')})`; ctx.fillRect(sx+i*cwid,sy,cwid-1,sh); });
    ctx.font=FONT_MONO; ctx.fillStyle=COL.muted; ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText('observation: base height \u2212 hit height \u2212 0.5, clipped to \u00B11', sx, sy+sh+4);
    $('#lsCap').textContent=`Side slice of the scan: 17 of the rays, 10 cm apart. ${S.box?'The box is a moving body: Newton\u2019s ray caster hits it; a static-mesh ray caster would see through it.':''}`;
  }
  function init(){ $('#lsNoise').onchange=e=>{ S.noise=e.target.checked; draw(); }; $('#lsBox').onchange=e=>{ S.box=e.target.checked; S.bx=S.x+2.2; draw(); };
    const advance = dt => {
      S.t += dt; S.x += dt * .6;
      if (S.box) {
        S.bx -= dt * .5;
        if (S.bx < S.x - 2.5) S.bx = S.x + 2.4;
      }
      draw();
    };
    const stop = bindPlayback('lsPlay', advance);
    $('#lsStep').onclick = () => { stop(); advance(.5); };
  }
  return {init, draw};
})();

// ---------- L4b: domain randomization histograms ----------
const LD = (() => {
  const N=4096; let seed=1, H=null;
  function sample(){ const q=rng(seed++); const mass=[], com=[], push=[], joint=[];
    for(let i=0;i<N;i++){ mass.push(Math.exp(Math.log(.8)+q()*(Math.log(1.25)-Math.log(.8)))); com.push((q()*2-1)*.05); push.push(10+q()*5); joint.push(.5+q()); }
    const hist=(a,lo,hi,b=24)=>{ const c=new Array(b).fill(0); a.forEach(v=>{ const k=Math.min(b-1,Math.max(0,Math.floor((v-lo)/(hi-lo)*b))); c[k]++; }); return {c,lo,hi}; };
    H=[['base mass scale (log-uniform)',hist(mass,.75,1.3),'\u00D7'],['base COM x offset (m)',hist(com,-.06,.06),''],['time between pushes (s)',hist(push,9.5,15.5),''],['reset joint scale',hist(joint,.45,1.55),'\u00D7']]; }
  function draw(){ if(!H) sample(); const cv=$('#ldCv'); const {ctx,w,h}=fit(cv,.62); ctx.fillStyle=COL.surface; ctx.fillRect(0,0,w,h);
    const cols=2, rows=2, pw=w/cols, ph=h/rows;
    H.forEach(([name,hs],idx)=>{ const ox=(idx%cols)*pw, oy=Math.floor(idx/cols)*ph; const mx=Math.max(...hs.c); const bw=(pw-24)/hs.c.length;
      ctx.font=FONT_MONO; ctx.fillStyle=COL.ink; ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText(name, ox+10, oy+6);
      hs.c.forEach((c,i)=>{ const bh=(ph-46)*c/mx; ctx.fillStyle=[COL.strain,COL.node,COL.collider,COL.particle][idx]; ctx.fillRect(ox+12+i*bw, oy+ph-18-bh, bw-1, bh); });
      ctx.fillStyle=COL.muted; ctx.textBaseline='top'; ctx.fillText(hs.lo.toFixed(2), ox+12, oy+ph-15); ctx.textAlign='right'; ctx.fillText(hs.hi.toFixed(2), ox+pw-12, oy+ph-15); });
    $('#ldCap').textContent='4,096 draws per term. The mass term is log-uniform, so it is as likely to make the robot 10% lighter as 10% heavier. Every robot in the batch is a slightly different machine.';
  }
  function init(){ $('#ldNew').onclick=()=>{ sample(); draw(); }; }
  return {init, draw};
})();

// ---------- boot ----------
const demos = [
  ['#simCv', SIM], ['#zCv', ZOOM], ['#vbCap', VB], ['#v3Cv', V3], ['#vbSurf', VS],
  ['#sbTrue', SB], ['#cbCv', CB], ['#dpCv', DP], ['#gsCv', GSC], ['#cvCv', CVG],
  ['#vlCv', VL], ['#vtCv', VT], ['#vmCv', VM], ['#vbCv2', VBE], ['#vnCv', VN],
  ['#vcCv', VC], ['#vdCv', VD], ['#vaCv', VA], ['#ltCv', LT], ['#lfCv', LF],
  ['#lpCv', LP], ['#laCv', LA], ['#lsCv', LS], ['#ldCv', LD],
];

function drawAll(root = document) {
  readCols();
  for (const [selector, demo] of demos) {
    const element = root.querySelector(selector);
    if (element?.getClientRects().length) (demo.draw ?? demo.resize)();
  }
}

readCols();
for (const [, demo] of demos) demo.init?.();
document.addEventListener('lessonchange', () => { syncPlayback(); drawAll(); });
document.addEventListener('visibilitychange', syncPlayback);
document.addEventListener('toggle', event => {
  if (!(event.target instanceof HTMLDetailsElement)) return;
  syncPlayback();
  if (event.target.open) drawAll(event.target);
}, true);
motion.addEventListener('change', syncPlayback);

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(drawAll, 80);
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => drawAll());
document.fonts?.ready.then(() => drawAll());
syncPlayback();
})();
