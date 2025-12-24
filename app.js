// app.js — Optimized for speed and client-side stability.
// Tweaks (v21.2 - Cleaned up Pythonisms that cause JS errors):
// - k reduced to 8 for speed
// - Removed Python-specific cleaning calls that do not belong in JS.

const $ = (id) => document.getElementById(id);
const api = () => (($('base')?.value || '').trim().replace(/\/+$/,''));
const status = (cls, msg) => {
    const s = $('status'); if (!s) return;
    s.className = 'status ' + cls;
    const el = $('statusText'); if (el) el.textContent = msg;
};

const TIMEOUTS = {
    healthAlive: 15000,
    healthOllama: 25000,
    reset: 60000,
    askUpload: 900000,
    ask: 900000,
    askStreamIdle: 300000 
};

async function fetchWithTimeout(resource, options = {}) {
    const { timeout = 0 } = options;
    if (!timeout || timeout <= 0) return fetch(resource, options);
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const res = await fetch(resource, { ...options, signal: controller.signal });
        clearTimeout(id);
        return res;
    } catch (e) {
        clearTimeout(id);
        throw e;
    }
}

(function initBase(){
    const el = $('base');
    const saved = localStorage.getItem('apiBase');
    if (saved) el.value = saved;
    el.value = (el.value || '').trim();
    el.addEventListener('change', ()=>{
        el.value = (el.value || '').trim();
        localStorage.setItem('apiBase', el.value);
        checkConn();
    });
})();

async function checkConn(){
    const base = api();
    if (!base) {
        status('err','API Base is empty');
        const askBtn = $('ask'); if (askBtn) askBtn.disabled = true;
        return;
    }
    try {
        const r1 = await fetchWithTimeout(base + "/health/alive", { timeout: TIMEOUTS.healthAlive });
        const d1 = r1.ok ? await r1.json() : { ok:false };
        if (!d1.ok) throw new Error("Alive failed");
    } catch(e) {
        status('err', `Cannot reach API (${e.name||'Error'}: ${e.message||e})`);
        const askBtn = $('ask'); if (askBtn) askBtn.disabled = true;
        return;
    }
    status('warn','Connected (checking LLM…)');
    const askBtn = $('ask'); if (askBtn) askBtn.disabled = false;
    try {
        const r2 = await fetchWithTimeout(base + "/health/ollama_only", { timeout: TIMEOUTS.healthOllama });
        const d2 = r2.ok ? await r2.json() : { ok:false };
        status(d2.ok ? 'ok':'warn', d2.ok ? 'Connected' : 'Connected (LLM warming)');
    } catch {
        status('warn','Connected (LLM warming)');
    }
}
document.addEventListener('DOMContentLoaded', checkConn);

const on = (id, ev, fn) => { const el=$(id); if(el) el.addEventListener(ev, fn); };
on('btnPing','click', checkConn);

// ---------- Chat UI helpers ----------
function addMsg(role, text){
    const c=document.createElement('div'); c.className='msg '+role;
    const av=document.createElement('div'); av.className='avatar '+(role==='bot'?'bot':'');
    const b=document.createElement('div'); b.className='bubble'; b.setAttribute('dir','auto');
    b.textContent = text;
    c.appendChild(av); c.appendChild(b);
    $('chat').appendChild(c);
    $('chat').scrollTop = $('chat').scrollHeight;
    return b;
}

// ---------- Attachments ----------
on('attach','click', ()=> $('file')?.click());
on('file','change', (e)=>{
    const f=e.target.files?.[0]; const ta=$('q');
    if(f){
        const p=ta.value.trim();
        ta.value=(p?p+"\n\n":"")+`📎 ${f.name}\n\n`;
        const fh=$('fileHint'); if(fh) fh.textContent=`Attached: ${f.name}`;
        ta.focus(); ta.selectionStart=ta.selectionEnd=ta.value.length;
    } else {
        const fh=$('fileHint'); if(fh) fh.textContent='';
    }
});

// ---------- Reset DB modal wiring ----------
const resetModal = $('resetModal');
const openResetModal  = () => resetModal?.classList.add('open');
const closeResetModal = () => resetModal?.classList.remove('open');

// open modal instead of window.confirm
on('btnReset','click', (e)=>{
    e.preventDefault();
    openResetModal();
});

// click outside closes
resetModal?.addEventListener('click', (e)=>{
    if (e.target === resetModal) closeResetModal();
});

// buttons inside modal
on('cancelReset','click', closeResetModal);
on('confirmReset','click', async ()=>{
    try{
        const r = await fetchWithTimeout(api()+"/reset",{method:'POST', timeout: TIMEOUTS.reset});
        addMsg('bot', r.ok ? 'Vector DB reset.' : 'Reset failed: '+r.status);
        status(r.ok ? 'ok' : 'err', r.ok ? 'Reset complete' : 'Reset failed');
    }catch(e){
        addMsg('bot','Reset failed: '+e);
        status('err','Reset failed');
    }finally{
        closeResetModal();
    }
});

// ESC closes modal
document.addEventListener('keydown', (e)=>{
    if (e.key === 'Escape' && resetModal?.classList.contains('open')) closeResetModal();
});

// ---------- Ask / streaming ----------
let currentDocId=null, sending=false;

on('q','keydown',(e)=>{
    if(e.key==='Enter' && !e.shiftKey){
        e.preventDefault();
        if (!sending) send(); // guard double-send
    }
});
on('ask','click', ()=>{ if(!sending) send(); });

async function streamAsk(url, body, bubble){
    const controller = new AbortController();
    let idleTimer = null;
    const bump = () => {
        if (idleTimer) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => controller.abort(), TIMEOUTS.askStreamIdle);
    };

    const res = await fetch(url, { method:"POST", body, signal: controller.signal });
    if (!res.ok || !res.body) { if (idleTimer) clearTimeout(idleTimer); return { ok:false, res }; }

    bubble.textContent='';
    const reader = res.body.getReader();
    const dec = new TextDecoder();

    bump();
    try{
        while (true) {
            const { value, done } = await reader.read();
            bump();
            if (done) break;
            bubble.textContent += dec.decode(value, { stream: true });
        }
        bubble.textContent += dec.decode();
        clearTimeout(idleTimer);
        return { ok:true };
    }catch(e){
        clearTimeout(idleTimer);
        throw e;
    }
}

async function send(){
    if(sending) return;

    const ta=$('q'), fileEl=$('file'), btn=$('ask');
    const qs=(ta?.value||'').replace(/^📎 .*\n\n/m,'').trim();
    const file=fileEl?.files?.[0]||null;
    if(!qs && !file){ ta.focus(); return; }

    ta.value=''; sending=true; if (btn) btn.disabled=true;
    if(qs) addMsg('user', qs);
    const typing = addMsg('bot',''); typing.innerHTML = '<span class="spinner"></span> Thinking…';

    try{
        if(file){
            const fd = new FormData();
            fd.append('question', qs || '');
            fd.append('pdf', file, file.name);
            fd.append('k','8'); // Reduced k for speed
            
            const r = await fetchWithTimeout(api()+"/ask_upload",{ method:"POST", body: fd, timeout: TIMEOUTS.askUpload });
            const raw = await r.text();
            if(!r.ok){
                let detail = raw;
                try { const j = JSON.parse(raw); detail = j.detail || raw; } catch {}
                typing.textContent = 'Error '+r.status+' — '+detail;
                return;
            }
            const data = JSON.parse(raw);
            currentDocId = data.doc_id || null;
            if(fileEl) fileEl.value=''; const fh=$('fileHint'); if(fh) fh.textContent='';
            typing.textContent = data.answer || '(no answer)';
        }else{
            const body = new URLSearchParams();
            body.set('question', qs);
            body.set('k','8'); // Reduced k for speed
            if(currentDocId) body.set('doc_id', currentDocId);

            try{
                // Try streaming first
                const result = await streamAsk(api()+"/ask_stream", body, typing);
                if (!result.ok) throw new Error('stream not available');
            }catch{
                // Fallback to non-streaming endpoint
                const r = await fetchWithTimeout(api()+"/ask",{
                    method:"POST",
                    headers:{ "Content-Type":"application/x-www-form-urlencoded" },
                    body,
                    timeout: TIMEOUTS.ask
                });
                const raw = await r.text();
                if(!r.ok){
                    let detail = raw;
                    try { const j = JSON.parse(raw); detail = j.detail || raw; } catch {}
                    typing.textContent = `Error ${r.status} — ${detail}`;
                    return;
                }
                const data = JSON.parse(raw);
                typing.textContent = data.answer || '(no answer)';
            }
        }
    }catch(e){
        typing.textContent = (e.name === 'AbortError')
            ? 'Request aborted due to inactivity. Try again.'
            : 'Network error: ' + (e.message || e);
    }finally{
        sending=false; if (btn) btn.disabled=false;
    }
}
