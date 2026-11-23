(() => {
const N = s => '<span class="nw"><i>In Newton</i>' + s + '</span>';
const G = [
 ['method','The method',[
  ['MPM','Material Point Method','A simulation method where <b>particles</b> carry the material and its history, and a <b>background grid</b> is used each step to compute motion and forces, then thrown away. Particles handle huge deformation and topology change; the grid makes forces and contacts cheap. Popular for sand, snow, soil, mud and debris.', N('<code>SolverImplicitMPM</code>.')],
  ['Material point','particle','A small chunk of material with mass, volume, velocity and state: elastic strain, hardening, last stress. Particles are never connected to each other; they only interact through the grid.', N('<code>particle_q</code>, <code>particle_qd</code> and the <code>mpm:*</code> custom attributes.')],
  ['Grid, cell, voxel, node','background grid','A regular lattice laid over space. A cell (or voxel in 3D) is one box; nodes are points on it where unknowns live. The grid holds no permanent state.', N('<code>voxel_size</code> (default 0.1) is the cell edge length.')],
  ['P2G / G2P','particle-to-grid, grid-to-particle, transfer','P2G spreads each particle\'s mass and momentum to nearby nodes using the basis weights. G2P reads the solved grid velocity back at each particle with the same weights.', N('<code>integrate_mass</code>, <code>integrate_velocity</code> (P2G); <code>advect_particles</code> (G2P).')],
  ['PIC','Particle-In-Cell','Transfer scheme where particles simply take the interpolated grid velocity. Very stable, but averaging loses small swirls and adds numerical damping.', N('<code>transfer_scheme="pic"</code>.')],
  ['FLIP','Fluid-Implicit-Particle','Transfer scheme where particles add the grid\'s velocity <i>change</i> instead of replacing their velocity. Keeps energy but can be noisy. Often blended with PIC.', N('Not offered; APIC fills the same role.')],
  ['APIC','Affine Particle-In-Cell','Each particle also carries a velocity gradient <code>C</code>, so the transfer keeps local rotation and shear instead of averaging them away. Low dissipation without FLIP\'s noise.', N('Default <code>transfer_scheme="apic"</code>; <code>integrate_velocity_apic</code> adds <code>C_p (x_i &minus; x_p)</code>.')],
  ['GIMP','Generalized Interpolation Material Point','Treats a particle as a small box instead of a point, and integrates the weights over the box. Weights then change more smoothly as particles cross cells.', N('<code>integration_scheme="gimp"</code> (default <code>"pic"</code> = point particles).')],
  ['Explicit vs implicit','time integration','Explicit: compute forces from the current state and step forward. Simple, but stiff materials need tiny time steps. Implicit: solve for the end-of-step velocities and stresses together. Stable for large steps, costs an iterative solve each step.', N('Implicit. The docs call it unconditionally stable with respect to the time step.')],
  ['Semi-implicit','','Implicit in the stresses and contacts, explicit in moving particles and updating geometry. This is the approach of the Daviet &amp; Bertails-Descoubes 2016 paper Newton follows.', N('The whole step structure in Part 5.')],
  ['CFL condition','Courant number','Stability rule of thumb: material should not move more than about one cell per step, or transfers break down.', N('<code>advect_particles</code> clamps particle speed to <code>max_vel</code>.')],
 ]],
 ['basis','Grid and basis functions',[
  ['Shape / basis function','N<sub>i</sub>, kernel','One bump per node. A particle\'s weight for node i is the bump\'s height at the particle. Its slope gives the velocity gradient.', N('Built by <code>_make_grid_basis_space</code> via warp.fem.')],
  ['Partition of unity','','The weights at any point add up to exactly 1, so a uniform velocity is reproduced exactly and the weight gradients add up to zero (no fake strain).', N('True for all of Q1, B2, B3.')],
  ['Stencil / support','','The set of nodes one particle touches: 8 for Q1, 27 for B2, 64 for B3 in 3D. Wider stencils are smoother but couple more unknowns.', N('B2/B3 reserve 64 velocity nodes per active cell.')],
  ['Lagrange element (Q1)','trilinear','Piecewise-linear in each direction, nodes at cell corners. Interpolating: the field equals the nodal value at a node. Gradient jumps at cell faces.', N('<code>velocity_basis="Q1"</code> (default), also a strain and collider option.')],
  ['B-spline (B2, B3)','quadratic, cubic','Smooth piecewise polynomials spanning 3 or 4 cells. Gradient continuous (B2) or even smoother (B3). Not interpolating.', N('<code>velocity_basis="B2"</code> / <code>"B3"</code>.')],
  ['Serendipity element (S2)','','A quadratic element that uses only corner and edge-midpoint nodes (20 per hex in 3D, no face or center nodes). Quadratic accuracy along edges at lower cost than full quadratic.', N('<code>collider_basis="S2"</code> (default).')],
  ['P0, P1d, Q1d','discontinuous polynomials','Polynomials defined per cell with no continuity between cells. P0 = constant, P1d = linear, Q1d = trilinear. The "d" means discontinuous.', N('<code>strain_basis</code> options; P0 is the default.')],
  ['Continuous vs discontinuous','conforming','Continuous bases share nodes between neighboring cells, so the field has no jumps. Discontinuous bases give each cell its own copy, allowing jumps and keeping solves local.', N('Q1 strain is continuous; P0/P1d/Q1d are not.')],
  ['Quadrature','integration points','Computing an integral as a weighted sum of samples. In MPM the particles are the quadrature points, each weighted by its volume.', N('<code>fem.PicQuadrature</code>; every <code>@fem.integrand</code> runs over particles.')],
  ['Gauss&ndash;Legendre points','','The standard optimal sample points for integrating polynomials on an interval.', N('Q1d strain nodes sit at these points.')],
  ['pic / picN basis','particle basis','Instead of grid nodes, put the unknowns at the particles themselves (at most N per cell).', N('<code>strain_basis="pic8"</code>, <code>collider_basis="pic8"</code>. A fixed N is required for graph capture.')],
  ['Cell-crossing error','grid-crossing noise','With Q1, a particle\'s weight gradients flip abruptly when it crosses a cell face, which makes forces and stress noisy. Smoother bases (B2, B3) or GIMP reduce it.', N('Main reason to try <code>velocity_basis="B2"</code>.')],
  ['Locking','inf&ndash;sup / LBB condition','If there are too many stress constraints for the available velocity freedom, the material can\'t move as it should and acts artificially stiff. The inf&ndash;sup condition is the math test for a stable velocity/stress pairing.', N('Why the default pairs Q1 velocity with P0 strain.')],
  ['Function space','','The set of all fields a basis can represent. Newton builds three: velocity, strain and collider.', N('<code>rebuild_function_spaces</code> in the solver scratchpad.')],
  ['Mass matrix, lumped','M','Relates grid velocity to momentum. "Lumped" means it is made diagonal: each node just has a mass. That makes M<sup>&minus;1</sup> trivial.', N('<code>inv_mass_matrix</code> from <code>free_velocity</code>.')],
 ]],
 ['material','Material: strain, stress, plasticity',[
  ['Rheology','','The study of how materials flow and deform under stress. A "rheology solver" finds the stresses that a given flow law allows.', N('<code>solve_rheology</code>, <code>rheology_solver_kernels.py</code>.')],
  ['Stress &sigma;','Cauchy stress','Internal force per unit area. A symmetric 3&times;3 tensor: 6 numbers in 3D. Unit: pascal.', N('One per strain node; stored per particle for warm start.')],
  ['Pressure and deviatoric stress','p, shear stress','Pressure is the average squeezing part of stress. The deviatoric part is what is left: the part that tries to change shape (shear).', N('The solver rotates each node\'s stress into a normal (pressure-like) and tangential (shear) part.')],
  ['Strain and strain rate','&epsilon;, D','Strain measures deformation; strain rate is how fast it changes. Strain rate = symmetric part of the velocity gradient.', N('<code>B</code> maps grid velocity to strain increments at strain nodes.')],
  ['Velocity gradient &nabla;v','','How velocity changes across space. Its symmetric part is the strain rate; its skew part is pure rotation (spin), which creates no stress.', N('Computed in G2P; skew part rotates elastic strain in <code>update_particle_strains</code>.')],
  ['Deformation gradient F','elastic strain F<sub>e</sub>','Maps small directions in the rest shape to the current shape. The elastic part F<sub>e</sub> is what stores recoverable deformation.', N('<code>mpm:particle_elastic_strain</code>.')],
  ['Hencky strain','logarithmic strain','Strain measured as the log of stretch. Behaves well for large deformation.', N('<code>USE_HENCKY_STRAIN_MEASURE</code> switch in the kernels.')],
  ['Young\'s modulus, Poisson ratio','E, &nu;','E is stiffness: stress per unit stretch. &nu; is how much a material narrows when stretched (0.5 = incompressible).', N('<code>mpm:young_modulus</code>, Poisson ratio attribute.')],
  ['Compliance','','Inverse of stiffness: how much strain per unit stress. Zero compliance = perfectly rigid.', N('<code>compliance_form</code>. Rigid (inelastic) particles have zero compliance.')],
  ['Damping (elastic)','','Viscous resistance on the elastic response, so vibrations die out.', N('Elastic parameter; enters as <code>compliance / (1 + damping/dt)</code>.')],
  ['Plasticity','','Permanent deformation that happens once stress reaches a limit. Sand flowing, clay staying dented.', N('Plastic strain increment from the solve; <code>plastic_strain_delta</code>.')],
  ['Rigid-plastic','fully inelastic','No elastic deformation at all: the material does not deform until stress reaches yield, then it flows. A good model for dry sand.', N('The docs call the solver well suited to this limit.')],
  ['Yield surface','','A boundary in stress space. Inside it the material holds; on it, the material flows. Stress is never allowed outside.', N('<code>YieldParamVec</code>; enforced by projection in the solver.')],
  ['Mohr&ndash;Coulomb','','The classic soil law: shear strength = cohesion + pressure &times; tan(friction angle). In 3D it is a pyramid with six sharp edges.', N('Not used directly; Drucker&ndash;Prager is its smooth cousin.')],
  ['Drucker&ndash;Prager','DP','A smooth cone version of Mohr&ndash;Coulomb: allowed shear stress grows linearly with pressure, |shear| &le; &mu; p + cohesion. Smooth means projecting onto it is easy and stable. The standard yield law for granular MPM.', N('The pressure-dependent yield in the solver docstring.')],
  ['Friction coefficient &mu;','friction angle','The slope of the yield cone: how much extra shear strength each unit of pressure buys. Sets the angle at which a pile stops (angle of repose).', N('Per-particle friction attribute (try the slider in Part 1).')],
  ['Cohesion','','Shear strength at zero pressure: what lets wet sand or snow hold a vertical wall.', N('Yield parameter. Disabled when packing drops below <code>critical_fraction</code>.')],
  ['Tensile strength, pressure cap','p_max','Limits on the yield surface: how much pulling it takes to break the material apart, and how much squeezing before it crushes.', N('<code>p_max</code> is the first entry of <code>YieldParamVec</code>.')],
  ['Flow rule','associated, non-associated','Once yielding, the flow rule says which direction plastic strain goes. Associated: perpendicular to the yield surface, which for Drucker&ndash;Prager makes the material expand as it shears (often too much). Non-associated: decoupled, usually with less volume change.', N('A piecewise-linear anisotropic flow rule, controlled by dilatancy.')],
  ['Dilatancy','','Volume change while shearing. Dense sand has to expand to shear. 0 means shear without volume change.', N('Per-particle dilatancy parameter.')],
  ['Hardening / softening','J<sub>p</sub>','The yield surface grows (hardening) or shrinks (softening) with plastic history. J<sub>p</sub> tracks plastic volume change: compaction hardens, expansion softens. Snow is the classic example.', N('<code>hardening_law</code>, <code>hardening_rate</code>, <code>softening_rate</code>.')],
  ['Viscosity','','Stress proportional to strain rate, as in honey or mud. Faster flow, more resistance.', N('Yield parameter. With discontinuous strain bases it is combined using squared basis weights.')],
  ['Cam-Clay','Modified Cam-Clay','A critical-state soil model with an elliptical yield surface that closes off at high pressure.', N('Internal development toggle <code>_USE_CAM_CLAY</code> (off).')],
  ['Packing fraction','volume fraction, critical fraction','How much of space the grains fill. Below the critical fraction, material is loose: no cohesion, and it may compress again.', N('<code>critical_fraction</code>, <code>volume_recovery_*</code> options.')],
  ['Unilateral incompressibility','','Grains can separate freely but cannot be squeezed denser than packed. A one-sided constraint on volume change.', N('<code>unilateral_strain_offset</code> in <code>RheologyData</code>.')],
 ]],
 ['solver','Solving the implicit step',[
  ['Strain operator B','','The matrix that turns grid velocities into strain increments at strain nodes. Its transpose turns stresses back into nodal forces.', N('<code>strain_delta_form</code>; <code>strain_mat</code>.')],
  ['Delassus operator','W = B M<sup>&minus;1</sup> B<sup>T</sup>','From contact mechanics: how much a unit stress (or impulse) at one node changes the strain rate at that node and its neighbors. Its diagonal blocks drive the cheap local updates.', N('<code>_DelassusOperator</code>, diagonal blocks eigen-decomposed into normal and tangential parts.')],
  ['Complementarity / projection','','Either stress is strictly inside the yield surface and nothing flows, or it is on the surface and flow happens in the allowed direction, never both. Solved by repeatedly projecting stress back onto the allowed set.', N('The local solve per strain node.')],
  ['Gauss&ndash;Seidel (GS)','projected GS','Sweep over unknowns one at a time, each time using the newest values of the others. Converges quickly but is naturally sequential. "Projected" GS clips each update onto the allowed set (yield surface, contact cone).', N('<code>solver="gs"</code>, picked by <code>"auto"</code> for Q1.')],
  ['Graph coloring','','Group strain nodes so that no two in a group share a velocity node. A whole group (color) can then be updated in parallel without conflicts, which turns GS into a GPU algorithm.', N('<code>color_offsets</code>, <code>color_blocks</code>.')],
  ['Jacobi','','Update all unknowns at once using only last iteration\'s values. Fully parallel but converges slower and needs damping. Good as a smoother before GS.', N('<code>solver="jacobi"</code>, or a chain like <code>("jacobi","gs")</code>.')],
  ['Batched GS','gs-batched','Merges several colors into one batch and splits node mass between the unknowns that share it, Jacobi-style. Fewer, bigger parallel passes; suits wide B2/B3 stencils with many colors.', N('<code>solver="gs-batched"</code>, picked by <code>"auto"</code> for B2/B3.')],
  ['GS-SoA','gs-soa','The same Gauss&ndash;Seidel with a struct-of-arrays memory layout for faster GPU memory access.', N('<code>solver="gs-soa"</code>.')],
  ['CG, CR, GMRES','Krylov solvers','Fast iterative solvers for linear systems (conjugate gradient, conjugate residual, generalized minimal residual). They cannot handle the clipping that yield or contact needs.', N('Only alone without contacts, or as a warm start, e.g. <code>("cr","gs")</code>.')],
  ['Nesterov acceleration','momentum, FISTA restart','Extrapolate each iterate a bit along its recent direction to converge faster. Restart the momentum when the residual grows.', N('<code>acceleration="nesterov"</code> in <code>solve_rheology</code>.')],
  ['Residual, tolerance','','The residual measures how much the current answer still violates the equations. The solve stops when it drops below the tolerance or hits the iteration cap.', N('<code>tolerance=1e-4</code>, <code>max_iterations=250</code>.')],
  ['Warm start','','Start the iteration from last step\'s stresses and contact impulses instead of zero. Usually cuts iterations a lot.', N('<code>warmstart_mode</code>; stored on particles or on the grid.')],
  ['Proximal regularization','','Add a small damping term to each local block so nearly empty or badly conditioned nodes don\'t produce huge updates.', N('<code>_DELASSUS_SCALED_REG</code>.')],
  ['Unconverged residual strain','','What to do with strain the solver had not finished resolving when it hit the iteration cap.', N('<code>carry_residual_strain</code>: "none", "unconverged", "sampled".')],
 ]],
 ['contact','Contact and colliders',[
  ['SDF','signed distance field','Distance to a surface, negative inside the object. Gives both penetration depth and the normal direction (its gradient).', N('Colliders rasterized onto collider nodes; <code>collider_normal_from_sdf_gradient</code>.')],
  ['Non-penetration','unilateral contact','Velocity into a wall is not allowed; velocity away is. Like unilateral incompressibility, it is one-sided.', N('Enforced at collider basis points.')],
  ['Coulomb friction','friction cone','Tangential contact force can be at most &mu; &times; normal force. Below that the material sticks; at the limit it slides.', N('Per-collider friction, <code>collider_friction</code>.')],
  ['Contact impulse','r','Force &times; time step applied at a contact. Implicit solvers work with impulses rather than forces.', N('<code>collider_impulse</code>, warm-started each step.')],
  ['Adhesion','','A limited pull that keeps material stuck to a wall.', N('<code>collider_adhesion</code>.')],
  ['Two-way coupling','rigidity operator','Collider bodies also receive the impulses back, so sand can push a dynamic box.', N('<code>rigidity_operator</code>, <code>apply_rigidity_operator</code>.')],
  ['Penetration recovery, shielding','','Recovery gently pushes particles out of walls over a time constant. Shielding skips contacts hidden behind the first particle layer.', N('<code>collider_recovery_time</code>, <code>collider_contact_shielding</code> (pic collider bases).')],
 ]],
 ['vbd','VBD: cloth, soft bodies, cables',[
  ['VBD','Vertex Block Descent','Solve implicit Euler as an energy minimization by visiting one vertex at a time: freeze its neighbors, take one Newton step for that vertex, move on. Repeat for a fixed number of iterations.', N('<code>SolverVBD</code>; Chen et al. 2024.')],
  ['Incremental potential','G(x)','The quantity implicit Euler minimizes each step: inertia term &Sigma; m/(2h&sup2;)|x &minus; y|&sup2; plus all elastic and contact energies. Its minimum is the end-of-step position.', N('Assembled per vertex as force f and Hessian H in <code>solve_elasticity</code>.')],
  ['Inertia target y','predicted position','Where a vertex would be after the step with no forces: y = x + h v + h&sup2; g. Also VBD\'s first guess.', N('<code>forward_step</code> kernel.')],
  ['Block coordinate descent','Gauss&ndash;Seidel on positions','Minimize over one small group of unknowns (a block) at a time while holding the rest fixed. In VBD a block is one vertex (3 unknowns) or one rigid body (6).', N('Per-color sweep in <code>_solve_particle_iteration</code>.')],
  ['Vertex coloring','graph coloring','Give vertices colors so no two vertices of the same color share an element. All vertices of one color can then update in parallel.', N('Required: <code>ModelBuilder.color()</code>.')],
  ['Deformation gradient F','','A small matrix mapping rest edges to current edges of a triangle (3&times;2) or tet (3&times;3). All stretch, shear and volume measures come from it.', N('Built from rest-pose inverse <code>tri_poses</code> / <code>tet_poses</code>.')],
  ['Stable Neo-Hookean','Smith et al. 2018','A hyperelastic energy &mu;/2(I_c &minus; d) + &lambda;\'/2(J &minus; &alpha;)&sup2; that stays finite when elements collapse or invert, unlike the classic log(J) form.', N('Cloth membranes (tri_ke = &mu;, tri_ka = &lambda;) and tets (k_mu, k_lambda).')],
  ['Dihedral angle bending','','Bending energy &frac12;k(&theta; &minus; &theta;_rest)&sup2; on the angle between two triangles that share an edge, with k = edge_ke &middot; rest edge length.', N('<code>edge_ke</code>, <code>edge_kd</code>.')],
  ['Rod / cable','discrete elastic rod','A chain of capsule bodies joined by ROD joints that resist stretch, shear, bending and twist.', N('<code>ModelBuilder.add_rod</code>; solved on VBD\'s rigid side.')],
  ['Log barrier','','A contact energy that grows like &minus;log(d), so the push becomes unbounded as two surfaces touch. Prevents passing through.', N('Self-contact below half the margin.')],
  ['IPC friction','smoothed Coulomb','Coulomb friction with a smooth ramp over a tiny slip &epsilon;_u, so its Hessian exists and Newton steps work.', N('<code>friction_epsilon</code> (default 1e-2) &times; h.')],
  ['Divide and Truncate','DAT','After each color, cut every proposed move so it stops at a dividing plane between nearby primitives and below a maximum length. Guarantees no tunneling.', N('<code>_penetration_free_truncation</code>; <code>dat_conservative_bound_relaxation</code>.')],
  ['Augmented Lagrangian (ALM)','AVBD, compliant ALM','Treat a constraint as a penalty spring plus a multiplier &lambda; that accumulates the missing force across iterations, so stiff joints converge without huge penalties.', N('Rigid joints and contacts: k_eff = K&rho;/(K+&rho;), &lambda; &larr; s(&lambda; + &rho;C).')],
 ]],
 ['lab','Isaac Lab and robot learning',[
  ['Isaac Lab','','NVIDIA\'s open-source robot-learning framework. Builds vectorized environments, manages actions, rewards and resets, and trains policies. Since 3.0, Newton is its default physics engine.', N('<code>NewtonCfg</code> in <code>SimulationCfg.physics</code>.')],
  ['Decimation','','How many physics steps run per policy step. Policy rate = 1 / (sim.dt &times; decimation).', N('Physics substeps inside each step: <code>NewtonCfg.num_substeps</code>.')],
  ['Manager-based env','ManagerBasedRLEnv','An environment assembled from configured terms grouped into managers: actions, observations, rewards, terminations, events, commands, curriculum.', N('Same loop on any physics backend.')],
  ['Direct env','DirectRLEnv','An environment where you write the step pieces (actions, dones, rewards, observations) as methods instead of manager terms.', N('')],
  ['Event / domain randomization','EventManager','Terms that change the simulation at startup, on reset or at intervals: masses, friction, pushes, initial poses. Makes policies robust to the sim-to-real gap.', N('Newton material randomization uses a single friction coefficient.')],
  ['PPO','Proximal Policy Optimization','The on-policy RL algorithm used by rsl_rl: collect rollouts, estimate advantages, take several clipped gradient steps.', N('ANYmal: 24 steps/env, 5 epochs, 4 mini-batches.')],
  ['GAE','Generalized Advantage Estimation','A_t = &delta;_t + &gamma;&lambda;A_t+1 with &delta;_t = r_t + &gamma;V(s_t+1) &minus; V(s_t). &lambda; trades bias against variance.', N('&gamma; 0.99, &lambda; 0.95.')],
  ['Time-out bootstrap','','When an episode ends only because time ran out, the value of the last state is added back to the reward, because the robot did not fail.', N('rsl_rl uses <code>extras["time_outs"]</code>.')],
  ['Implicit actuator','ImplicitActuatorCfg','A PD joint drive that the physics solver integrates implicitly, so high gains stay stable.', N('stiffness/damping &rarr; <code>joint_target_ke/kd</code>.')],
  ['Explicit actuator','IdealPD, DCMotor, actuator nets','A torque computed once per physics step from the current state and applied as a constant. Lets you model motor limits, delays or learned dynamics.', N('Runs as a native Newton actuator inside the CUDA graph.')],
  ['Height scan','RayCaster','A grid of downward rays around the robot base giving terrain heights as an observation.', N('Newton\'s ray caster also hits moving bodies.')],
  ['Sim-to-sim','','Testing a policy in a different physics engine than the one it was trained in, as a cheap first check before real hardware.', N('PhysX &harr; Newton transfer guide: PP, PN, NN, NP.')],
 ]],
 ['impl','Implementation and GPU',[
  ['Warp, warp.fem','','NVIDIA\'s Python framework for GPU kernels, and its finite-element module that builds the bases, function spaces and integrals.', N('Everything above runs on <code>warp</code> / <code>warp.fem</code>.')],
  ['Sparse grid, NanoVDB','','Store only the active cells near particles, in a hierarchical tree (NanoVDB) instead of a full dense box.', N('<code>grid_type="sparse"</code> (default), <code>max_active_cell_count</code>.')],
  ['CUDA graph capture','','Record a whole step\'s GPU launches once and replay them with almost no CPU overhead. Requires fixed sizes.', N('Why <code>picN</code> needs a fixed N and the sparse grid needs a capacity.')],
  ['Kinematic particles','','Particles with prescribed motion that push material but are not pushed back.', N('Density 0: treated as infinite mass in <code>integrate_mass</code>.')],
  ['Air drag','','A small extra mass in every node\'s denominator so nearly empty nodes don\'t get huge velocities.', N('<code>air_drag</code> in <code>free_velocity</code>.')],
  ['Separate worlds','','Many independent simulations batched together on one GPU, with or without a shared grid.', N('<code>separate_worlds</code>.')],
 ]],
];
const body=document.getElementById('glBody'), jump=document.getElementById('glJump');
let html='', nav='';
for(const [id,title,items] of G){
  nav += `<a href="#gl-${id}">${title}</a>`;
  html += `<div class="gl-group" id="gl-${id}"><h3>${title}</h3><div class="gl">` + items.map(([t,a,d,n])=>`<div class="term"><div class="tt"><b>${t}</b>${a?`<span class="aka">${a}</span>`:''}</div><p>${d}</p>${n}</div>`).join('') + `</div></div>`;
}
body.innerHTML=html; jump.innerHTML=nav;
const total=body.querySelectorAll('.term').length, cnt=document.getElementById('glCount'), empty=document.getElementById('glEmpty');
cnt.textContent = total + ' terms';
document.getElementById('glq').addEventListener('input', e=>{
  const q=e.target.value.trim().toLowerCase(); let shown=0;
  body.querySelectorAll('.gl-group').forEach(g=>{ let any=0; g.querySelectorAll('.term').forEach(t=>{ const ok=!q || t.textContent.toLowerCase().includes(q); t.hidden=!ok; if(ok){any++;shown++;} }); g.hidden=!any; });
  cnt.textContent = q ? `${shown} of ${total} terms` : `${total} terms`; empty.hidden = shown>0;
});
})();
