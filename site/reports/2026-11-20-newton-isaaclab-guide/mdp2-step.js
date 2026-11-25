(() => {
  const lab = document.querySelector('#mdp2-lab');
  if (!lab) return;

  const episodeLimit = 4;
  const initialWorlds = [
    {name: 'A', position: -0.34, angle: -14, age: 0, push: 35},
    {name: 'B', position: 0.84, angle: 16, age: 1, push: 65},
    {name: 'C', position: 0.09, angle: -8, age: 3, push: -25},
  ];
  const labels = {
    ready: 'Step physics',
    physics: 'Evaluate outcomes',
    evaluate: 'Apply masked reset',
    reset: 'Step physics',
  };
  const outcomes = {
    continuing: 'Continues',
    terminated: 'Travel limit',
    truncated: 'Time limit',
  };
  const phases = [...lab.querySelectorAll('[data-mdp2-phase]')];
  const template = lab.querySelector('[data-mdp2-world-template]');
  const container = lab.querySelector('[data-mdp2-worlds]');
  const mask = lab.querySelector('[data-mdp2-mask]');
  const status = lab.querySelector('[data-mdp2-status]');
  const advance = lab.querySelector('[data-mdp2-advance]');

  const worlds = initialWorlds.map(initial => {
    const element = template.content.firstElementChild.cloneNode(true);
    const input = element.querySelector('input');
    element.querySelector('[data-mdp2-name]').textContent = 'World ' + initial.name;
    input.setAttribute('aria-label', 'Push in world ' + initial.name);
    container.append(element);
    return {
      initial,
      element,
      input,
      cart: element.querySelector('.mdp2-cart'),
      pole: element.querySelector('.mdp2-pole'),
      age: element.querySelector('[data-mdp2-age]'),
      push: element.querySelector('[data-mdp2-push]'),
      outcome: element.querySelector('[data-mdp2-outcome]'),
    };
  });

  let phase;
  let result;
  let state;

  function updatePush(world) {
    const push = Number(world.input.value);
    world.push.textContent = (push > 0 ? '+' : '') + push + '%';
    const spoken = push ? Math.abs(push) + '% ' + (push < 0 ? 'left' : 'right') : 'No push';
    world.input.setAttribute('aria-valuetext', spoken);
  }

  function render(message) {
    const currentPhase = phases.findIndex(element => element.dataset.mdp2Phase === phase);
    phases.forEach((element, index) => {
      element.toggleAttribute('data-complete', index < currentPhase);
      if (index === currentPhase) element.setAttribute('aria-current', 'step');
      else element.removeAttribute('aria-current');
    });

    worlds.forEach((world, index) => {
      const {position, angle, age} = state[index];
      world.cart.style.setProperty('--cart-position', (position * 33) + '%');
      world.pole.style.setProperty('--pole-angle', angle + 'deg');
      world.age.textContent = age + ' / ' + episodeLimit + ' steps';
      world.input.disabled = phase === 'physics' || phase === 'evaluate';

      const outcome = result?.[index];
      const reset = phase === 'reset' && outcome !== 'continuing';
      world.element.dataset.outcome = reset ? 'reset' : outcome || 'pending';
      const pending = phase === 'physics' ? 'Awaiting check' : 'Active';
      world.outcome.textContent = reset ? 'New episode' : outcomes[outcome] || pending;
    });

    mask.textContent = result ? '[' + result.map(outcome => Number(outcome !== 'continuing')).join(', ') + ']' : '—';
    advance.textContent = labels[phase];
    status.textContent = message;
  }

  function classify({position, age}) {
    if (Math.abs(position) > 1) return 'terminated';
    return age >= episodeLimit ? 'truncated' : 'continuing';
  }

  function restart() {
    lab.classList.remove('mdp2-moving');
    state = initialWorlds.map(({position, angle, age}) => ({position, angle, age}));
    phase = 'ready';
    result = null;
    worlds.forEach(world => {
      world.input.value = world.initial.push;
      updatePush(world);
    });
    render('Adjust a push and step the physics.');
  }

  advance.addEventListener('click', event => {
    if (phase === 'physics') {
      result = state.map(classify);
      phase = 'evaluate';
      const descriptions = result.map((outcome, index) => {
        const name = 'World ' + worlds[index].initial.name;
        if (outcome === 'terminated') return name + ' terminated at the travel limit';
        if (outcome === 'truncated') return name + ' truncated at the time limit';
        return name + ' continues';
      });
      const bits = result.map(outcome => Number(outcome !== 'continuing'));
      render(descriptions.join('. ') + '. Reset mask [' + bits.join(', ') + '].');
    } else if (phase === 'evaluate') {
      lab.classList.remove('mdp2-moving');
      const resetWorlds = [];
      state = state.map((world, index) => {
        if (result[index] === 'continuing') return world;
        resetWorlds.push(worlds[index].initial.name);
        return {position: 0, angle: 0, age: 0};
      });
      phase = 'reset';
      const message = resetWorlds.length
        ? 'Reset applied to ' + (resetWorlds.length === 1 ? 'world ' : 'worlds ') + resetWorlds.join(', ')
          + '. Their next observations come from the new episodes.'
        : 'All worlds keep their current state. The reset mask is all zeros.';
      render(message);
    } else {
      // Keep the mask logic visible with simple kinematics instead of modeling Cartpole dynamics.
      lab.classList.toggle('mdp2-moving', event.detail > 0);
      state = state.map((world, index) => {
        const push = Number(worlds[index].input.value) / 100;
        return {
          position: world.position + 0.4 * push,
          angle: Math.max(-40, Math.min(40, world.angle - 16 * push)),
          age: world.age + 1,
        };
      });
      result = null;
      phase = 'physics';
      render('The carts moved. Evaluate the travel limits and episode timers.');
    }
  });
  lab.querySelector('[data-mdp2-restart]').addEventListener('click', restart);
  worlds.forEach(world => world.input.addEventListener('input', () => updatePush(world)));
  restart();
  lab.parentElement.querySelector('[data-mdp2-fallback]').hidden = true;
  lab.hidden = false;
})();
