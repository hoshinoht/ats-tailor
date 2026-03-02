// ===== Utility Components =====

function makeTagInput(tags, onChange) {
  const wrap = document.createElement('div');
  wrap.className = 'tag-input-wrap';

  function renderChips() {
    // Clear existing chips but keep the input
    Array.from(wrap.children).forEach(c => c.remove());

    tags.forEach((tag, i) => {
      const chip = document.createElement('span');
      chip.className = 'tag-chip';
      const txt = document.createTextNode(tag + ' ');
      chip.appendChild(txt);
      const rm = document.createElement('button');
      rm.className = 'tag-chip-remove';
      rm.textContent = '\u00d7';
      rm.onclick = () => { tags.splice(i, 1); renderChips(); onChange(); };
      chip.appendChild(rm);
      wrap.appendChild(chip);
    });
    const input = document.createElement('input');
    input.className = 'tag-text-input';
    input.placeholder = 'Add tag\u2026';
    input.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault();
        const val = input.value.trim().replace(/,+$/, '');
        if (val) { tags.push(val); renderChips(); onChange(); }
      } else if (e.key === 'Backspace' && !input.value && tags.length) {
        tags.pop(); renderChips(); onChange();
      }
    };
    input.onblur = () => {
      const val = input.value.trim().replace(/,+$/, '');
      if (val) { tags.push(val); renderChips(); onChange(); }
    };
    wrap.appendChild(input);
    wrap.onclick = () => input.focus();
  }

  renderChips();
  return wrap;
}

function makeBulletList(bullets, onChange) {
  const div = document.createElement('div');
  div.className = 'bullet-list';

  function render() {
    while (div.firstChild) div.removeChild(div.firstChild);
    if (bullets.length === 0) {
      const nm = document.createElement('div');
      nm.className = 'no-items';
      nm.textContent = 'No items';
      div.appendChild(nm);
    }
    bullets.forEach((b, i) => {
      const row = document.createElement('div');
      row.className = 'bullet-item';
      const ta = document.createElement('textarea');
      ta.value = b;
      ta.rows = 2;
      ta.oninput = () => { bullets[i] = ta.value; onChange(); };
      const rm = makeIconBtn('\u00d7', 'danger', () => { bullets.splice(i, 1); render(); onChange(); });
      row.appendChild(ta);
      row.appendChild(rm);
      div.appendChild(row);
    });
    const addBtn = document.createElement('button');
    addBtn.className = 'add-btn';
    addBtn.textContent = '+ Add bullet';
    addBtn.onclick = () => { bullets.push(''); render(); onChange(); };
    div.appendChild(addBtn);
  }

  render();
  return div;
}

function makeIconBtn(label, extraClass, onClick) {
  const btn = document.createElement('button');
  btn.className = 'icon-btn' + (extraClass ? ' ' + extraClass : '');
  btn.textContent = label;
  btn.onclick = onClick;
  return btn;
}

function fieldRow(label, inputEl) {
  const row = document.createElement('div');
  row.className = 'field-row';
  const lbl = document.createElement('span');
  lbl.className = 'field-label';
  lbl.textContent = label;
  row.appendChild(lbl);
  row.appendChild(inputEl);
  return row;
}

function makeTextInput(obj, key, onChange, placeholder) {
  const inp = document.createElement('input');
  inp.className = 'field-input';
  inp.type = 'text';
  inp.value = obj[key] || '';
  if (placeholder) inp.placeholder = placeholder;
  inp.oninput = () => { obj[key] = inp.value; onChange(); };
  return inp;
}

function makeSelect(obj, key, options, onChange) {
  const sel = document.createElement('select');
  sel.className = 'field-select';
  options.forEach(opt => {
    const o = document.createElement('option');
    o.value = opt;
    o.textContent = opt;
    if (obj[key] === opt) o.selected = true;
    sel.appendChild(o);
  });
  sel.onchange = () => { obj[key] = sel.value; onChange(); };
  if (!obj[key] && options.length) obj[key] = options[0];
  return sel;
}

function reorderBtns(arr, i, render, onChange) {
  const frag = document.createDocumentFragment();
  const up = makeIconBtn('\u2191', '', () => {
    if (i > 0) { [arr[i-1], arr[i]] = [arr[i], arr[i-1]]; render(); onChange(); }
  });
  const dn = makeIconBtn('\u2193', '', () => {
    if (i < arr.length - 1) { [arr[i], arr[i+1]] = [arr[i+1], arr[i]]; render(); onChange(); }
  });
  frag.appendChild(up);
  frag.appendChild(dn);
  return frag;
}

function makeAccordion(titleContent, bodyEl, collapsed) {
  const acc = document.createElement('div');
  acc.className = 'section-accordion' + (collapsed ? ' collapsed' : '');
  const hdr = document.createElement('div');
  hdr.className = 'section-header';
  const tog = document.createElement('span');
  tog.className = 'section-toggle';
  tog.textContent = '\u25be';
  hdr.appendChild(tog);
  if (typeof titleContent === 'string') {
    const t = document.createElement('span');
    t.className = 'section-title';
    t.textContent = titleContent;
    hdr.appendChild(t);
  } else {
    hdr.appendChild(titleContent);
  }
  hdr.onclick = (e) => {
    if (e.target.tagName === 'INPUT') return;
    acc.classList.toggle('collapsed');
  };
  const body = document.createElement('div');
  body.className = 'section-body';
  body.appendChild(bodyEl);
  acc.appendChild(hdr);
  acc.appendChild(body);
  return acc;
}

// ===== Certifications =====

function makeTopAddBtn(label, onClick) {
  const btn = document.createElement('button');
  btn.className = 'add-btn add-btn-top';
  btn.textContent = label;
  btn.onclick = onClick;
  return btn;
}

function renderCertifications(container, data, onChange) {
  if (!Array.isArray(data)) data = [];

  function addNew() {
    data.push({ id: '', name: '', issuer: '', ats_keywords: [], relevance: [] });
    render();
    onChange();
    // Scroll the new card into view
    container.querySelector('.item-card:last-of-type')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function render() {
    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(makeTopAddBtn('+ Add Certification', addNew));
    if (data.length === 0) {
      const nm = document.createElement('div');
      nm.className = 'no-items';
      nm.textContent = 'No certifications';
      container.appendChild(nm);
    }
    data.forEach((cert, i) => {
      const card = document.createElement('div');
      card.className = 'item-card';
      const hdr = document.createElement('div');
      hdr.className = 'item-card-header';
      const ttl = document.createElement('span');
      ttl.className = 'item-card-title';
      ttl.textContent = cert.name || ('Cert #' + (i+1));
      hdr.appendChild(ttl);
      hdr.appendChild(reorderBtns(data, i, render, onChange));
      hdr.appendChild(makeIconBtn('\u00d7', 'danger', () => { data.splice(i, 1); render(); onChange(); }));
      card.appendChild(hdr);
      ['id', 'name', 'issuer'].forEach(f => card.appendChild(fieldRow(f, makeTextInput(cert, f, onChange))));
      cert.ats_keywords = cert.ats_keywords || [];
      cert.relevance = cert.relevance || [];
      card.appendChild(fieldRow('ats_keywords', makeTagInput(cert.ats_keywords, onChange)));
      card.appendChild(fieldRow('relevance', makeTagInput(cert.relevance, onChange)));
      container.appendChild(card);
    });
    container.appendChild(makeTopAddBtn('+ Add Certification', addNew));
  }

  render();
  return data;
}

// ===== Experience =====

function renderExperience(container, data, onChange) {
  if (!data || typeof data !== 'object') data = { roles: [], education: [] };
  data.roles = data.roles || [];
  data.education = data.education || [];
  while (container.firstChild) container.removeChild(container.firstChild);

  function addRole() {
    data.roles.push({ id: '', company: '', title: '', period: '', type: 'internship', skills_used: [], bullets: [], ats_tags: [] });
  }
  function addEdu() {
    data.education.push({ institution: '', degree: '', field: '', period: '' });
  }

  function renderRoles(rolesContainer) {
    while (rolesContainer.firstChild) rolesContainer.removeChild(rolesContainer.firstChild);
    rolesContainer.appendChild(makeTopAddBtn('+ Add Role', () => { addRole(); renderRoles(rolesContainer); onChange(); }));
    if (data.roles.length === 0) {
      const nm = document.createElement('div'); nm.className = 'no-items'; nm.textContent = 'No roles';
      rolesContainer.appendChild(nm);
    }
    data.roles.forEach((role, i) => {
      const card = document.createElement('div');
      card.className = 'item-card';
      const hdr = document.createElement('div');
      hdr.className = 'item-card-header';
      const ttl = document.createElement('span');
      ttl.className = 'item-card-title';
      ttl.textContent = role.company || ('Role #' + (i+1));
      hdr.appendChild(ttl);
      hdr.appendChild(reorderBtns(data.roles, i, () => renderRoles(rolesContainer), onChange));
      hdr.appendChild(makeIconBtn('\u00d7', 'danger', () => { data.roles.splice(i, 1); renderRoles(rolesContainer); onChange(); }));
      card.appendChild(hdr);
      ['id', 'company', 'title', 'period'].forEach(f => card.appendChild(fieldRow(f, makeTextInput(role, f, onChange))));
      card.appendChild(fieldRow('type', makeSelect(role, 'type', ['research', 'internship'], onChange)));
      role.skills_used = role.skills_used || [];
      role.ats_tags = role.ats_tags || [];
      role.bullets = role.bullets || [];
      card.appendChild(fieldRow('skills_used', makeTagInput(role.skills_used, onChange)));
      card.appendChild(fieldRow('ats_tags', makeTagInput(role.ats_tags, onChange)));
      const blbl = document.createElement('div'); blbl.className = 'sub-label'; blbl.textContent = 'Bullets';
      card.appendChild(blbl);
      card.appendChild(makeBulletList(role.bullets, onChange));
      rolesContainer.appendChild(card);
    });
    rolesContainer.appendChild(makeTopAddBtn('+ Add Role', () => { addRole(); renderRoles(rolesContainer); onChange(); }));
  }

  function renderEducation(eduContainer) {
    while (eduContainer.firstChild) eduContainer.removeChild(eduContainer.firstChild);
    eduContainer.appendChild(makeTopAddBtn('+ Add Education', () => { addEdu(); renderEducation(eduContainer); onChange(); }));
    if (data.education.length === 0) {
      const nm = document.createElement('div'); nm.className = 'no-items'; nm.textContent = 'No education entries';
      eduContainer.appendChild(nm);
    }
    data.education.forEach((edu, i) => {
      const card = document.createElement('div');
      card.className = 'item-card';
      const hdr = document.createElement('div');
      hdr.className = 'item-card-header';
      const ttl = document.createElement('span');
      ttl.className = 'item-card-title';
      ttl.textContent = edu.institution || ('Education #' + (i+1));
      hdr.appendChild(ttl);
      hdr.appendChild(reorderBtns(data.education, i, () => renderEducation(eduContainer), onChange));
      hdr.appendChild(makeIconBtn('\u00d7', 'danger', () => { data.education.splice(i, 1); renderEducation(eduContainer); onChange(); }));
      card.appendChild(hdr);
      ['institution', 'degree', 'field', 'period'].forEach(f => card.appendChild(fieldRow(f, makeTextInput(edu, f, onChange))));
      eduContainer.appendChild(card);
    });
    eduContainer.appendChild(makeTopAddBtn('+ Add Education', () => { addEdu(); renderEducation(eduContainer); onChange(); }));
  }

  const rolesDiv = document.createElement('div');
  const eduDiv = document.createElement('div');
  renderRoles(rolesDiv);
  renderEducation(eduDiv);
  container.appendChild(makeAccordion('Roles', rolesDiv, false));
  container.appendChild(makeAccordion('Education', eduDiv, false));
  return data;
}

// ===== Projects =====

function renderProjects(container, data, onChange) {
  if (!Array.isArray(data)) data = [];
  const stackKeys = ['languages', 'frameworks', 'infrastructure', 'patterns', 'tools'];

  function addNew() {
    data.push({ id: '', name: '', type: '', period: '', repo: '', summary: '', tech_stack: {}, responsibilities: [], impact: [], ats_tags: [] });
    render();
    onChange();
    container.querySelector('.item-card:last-of-type')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function render() {
    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(makeTopAddBtn('+ Add Project', addNew));
    if (data.length === 0) {
      const nm = document.createElement('div'); nm.className = 'no-items'; nm.textContent = 'No projects';
      container.appendChild(nm);
    }
    data.forEach((proj, i) => {
      const card = document.createElement('div');
      card.className = 'item-card';
      const hdr = document.createElement('div');
      hdr.className = 'item-card-header';
      const ttl = document.createElement('span');
      ttl.className = 'item-card-title';
      ttl.textContent = proj.name || ('Project #' + (i+1));
      hdr.appendChild(ttl);
      hdr.appendChild(reorderBtns(data, i, render, onChange));
      hdr.appendChild(makeIconBtn('\u00d7', 'danger', () => { data.splice(i, 1); render(); onChange(); }));
      card.appendChild(hdr);
      ['id', 'name', 'type', 'period', 'repo', 'summary'].forEach(f => card.appendChild(fieldRow(f, makeTextInput(proj, f, onChange))));
      const tsLbl = document.createElement('div'); tsLbl.className = 'sub-label'; tsLbl.textContent = 'Tech Stack';
      card.appendChild(tsLbl);
      proj.tech_stack = proj.tech_stack || {};
      stackKeys.forEach(k => {
        proj.tech_stack[k] = proj.tech_stack[k] || [];
        card.appendChild(fieldRow(k, makeTagInput(proj.tech_stack[k], onChange)));
      });
      const respLbl = document.createElement('div'); respLbl.className = 'sub-label'; respLbl.textContent = 'Responsibilities';
      card.appendChild(respLbl);
      proj.responsibilities = proj.responsibilities || [];
      card.appendChild(makeBulletList(proj.responsibilities, onChange));
      const impLbl = document.createElement('div'); impLbl.className = 'sub-label'; impLbl.textContent = 'Impact';
      card.appendChild(impLbl);
      proj.impact = proj.impact || [];
      card.appendChild(makeBulletList(proj.impact, onChange));
      proj.ats_tags = proj.ats_tags || [];
      card.appendChild(fieldRow('ats_tags', makeTagInput(proj.ats_tags, onChange)));
      container.appendChild(card);
    });
    container.appendChild(makeTopAddBtn('+ Add Project', addNew));
  }

  render();
  return data;
}

// ===== Skills =====

function renderSkills(container, data, onChange) {
  if (!Array.isArray(data)) data = [];
  const proficiencyOpts = ['advanced', 'intermediate', 'familiar'];
  const evidenceTypes = ['role', 'project', 'cert'];

  function makeEvidenceList(evidenceArr) {
    const div = document.createElement('div');
    div.className = 'evidence-list';

    function renderEvidence() {
      while (div.firstChild) div.removeChild(div.firstChild);
      evidenceArr.forEach((ev, ei) => {
        const row = document.createElement('div');
        row.className = 'evidence-item';
        row.appendChild(makeSelect(ev, 'type', evidenceTypes, onChange));
        const refInp = document.createElement('input');
        refInp.className = 'field-input';
        refInp.placeholder = 'ref';
        refInp.value = ev.ref || '';
        refInp.oninput = () => { ev.ref = refInp.value; onChange(); };
        row.appendChild(refInp);
        const detInp = document.createElement('input');
        detInp.className = 'field-input';
        detInp.placeholder = 'detail (optional)';
        detInp.value = ev.detail || '';
        detInp.oninput = () => { ev.detail = detInp.value; onChange(); };
        row.appendChild(detInp);
        row.appendChild(makeIconBtn('\u00d7', 'danger', () => { evidenceArr.splice(ei, 1); renderEvidence(); onChange(); }));
        div.appendChild(row);
      });
      const addEv = document.createElement('button');
      addEv.className = 'add-btn';
      addEv.style.fontSize = '12px';
      addEv.textContent = '+ Add evidence';
      addEv.onclick = () => { evidenceArr.push({ type: 'role', ref: '', detail: '' }); renderEvidence(); onChange(); };
      div.appendChild(addEv);
    }

    renderEvidence();
    return div;
  }

  function makeSkillCard(skill, skillArr, si, reRenderSkills) {
    const card = document.createElement('div');
    card.className = 'item-card';
    const hdr = document.createElement('div');
    hdr.className = 'item-card-header';
    const ttl = document.createElement('span');
    ttl.className = 'item-card-title';
    ttl.textContent = skill.name || ('Skill #' + (si+1));
    hdr.appendChild(ttl);
    hdr.appendChild(reorderBtns(skillArr, si, reRenderSkills, onChange));
    hdr.appendChild(makeIconBtn('\u00d7', 'danger', () => { skillArr.splice(si, 1); reRenderSkills(); onChange(); }));
    card.appendChild(hdr);
    card.appendChild(fieldRow('name', makeTextInput(skill, 'name', () => { ttl.textContent = skill.name || ('Skill #' + (si+1)); onChange(); })));
    card.appendChild(fieldRow('proficiency', makeSelect(skill, 'proficiency', proficiencyOpts, onChange)));
    skill.ats_keywords = skill.ats_keywords || [];
    card.appendChild(fieldRow('ats_keywords', makeTagInput(skill.ats_keywords, onChange)));
    const evLbl = document.createElement('div'); evLbl.className = 'sub-label'; evLbl.textContent = 'Evidence';
    card.appendChild(evLbl);
    skill.evidence = skill.evidence || [];
    card.appendChild(makeEvidenceList(skill.evidence));
    return card;
  }

  function renderCat(cat, catDiv) {
    while (catDiv.firstChild) catDiv.removeChild(catDiv.firstChild);
    const skillArr = cat.skills || [];
    skillArr.forEach((skill, si) => {
      catDiv.appendChild(makeSkillCard(skill, skillArr, si, () => renderCat(cat, catDiv)));
    });
    const addSkill = document.createElement('button');
    addSkill.className = 'add-btn';
    addSkill.textContent = '+ Add Skill';
    addSkill.onclick = () => {
      skillArr.push({ name: '', proficiency: 'intermediate', ats_keywords: [], evidence: [] });
      renderCat(cat, catDiv); onChange();
    };
    catDiv.appendChild(addSkill);
  }

  function addCategory() {
    data.push({ name: '', skills: [] });
    render();
    onChange();
    container.querySelector('.section-accordion:last-of-type')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function render() {
    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(makeTopAddBtn('+ Add Category', addCategory));
    if (data.length === 0) {
      const nm = document.createElement('div'); nm.className = 'no-items'; nm.textContent = 'No categories';
      container.appendChild(nm);
    }
    data.forEach((cat, ci) => {
      cat.skills = cat.skills || [];
      const titleNode = document.createElement('div');
      titleNode.style.display = 'flex';
      titleNode.style.alignItems = 'center';
      titleNode.style.gap = '6px';
      titleNode.style.flex = '1';
      const nameInp = document.createElement('input');
      nameInp.className = 'field-input';
      nameInp.style.flex = '1';
      nameInp.value = cat.name || '';
      nameInp.placeholder = 'Category name';
      nameInp.oninput = (e) => { e.stopPropagation(); cat.name = nameInp.value; onChange(); };
      titleNode.appendChild(nameInp);
      titleNode.appendChild(reorderBtns(data, ci, render, onChange));
      titleNode.appendChild(makeIconBtn('\u00d7', 'danger', () => { data.splice(ci, 1); render(); onChange(); }));
      const catBody = document.createElement('div');
      renderCat(cat, catBody);
      container.appendChild(makeAccordion(titleNode, catBody, false));
    });
    container.appendChild(makeTopAddBtn('+ Add Category', addCategory));
  }

  render();
  return data;
}

// ===== Main Export =====

export function renderTree(container, fileName, data, onChange) {
  while (container.firstChild) container.removeChild(container.firstChild);
  const wrap = document.createElement('div');
  wrap.style.display = 'flex';
  wrap.style.flexDirection = 'column';
  wrap.style.gap = '12px';
  const name = fileName.replace('.yaml', '');
  let liveData = data;

  function changed() { onChange(liveData); }

  switch (name) {
    case 'certifications':
      liveData = data && data.certifications ? data.certifications : (Array.isArray(data) ? data : []);
      renderCertifications(wrap, liveData, () => { onChange({ certifications: liveData }); });
      break;
    case 'experience':
      liveData = data && (data.roles || data.education) ? data : { roles: [], education: [] };
      renderExperience(wrap, liveData, () => { onChange(liveData); });
      break;
    case 'projects':
      liveData = data && data.projects ? data.projects : (Array.isArray(data) ? data : []);
      renderProjects(wrap, liveData, () => { onChange({ projects: liveData }); });
      break;
    case 'skills':
      liveData = data && data.categories ? data.categories : (Array.isArray(data) ? data : []);
      renderSkills(wrap, liveData, () => { onChange({ categories: liveData }); });
      break;
    default: {
      const msg = document.createElement('div');
      msg.className = 'no-items';
      msg.textContent = 'No tree editor available for "' + fileName + '". Use Raw mode.';
      wrap.appendChild(msg);
    }
  }
  container.appendChild(wrap);
}
