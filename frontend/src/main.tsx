import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {UploadCloud, FileSpreadsheet, Play, Sparkles, ShieldCheck, Cpu, Download, ScanLine, Activity} from 'lucide-react';
import './styles.css';

type Job = {job_id:string; status:string; progress:number; filename:string; rows_count:number; metrics:Record<string, unknown>; error?:string|null; csv_url?:string|null; xlsx_url?:string|null};
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App(){
  const [file,setFile]=useState<File|null>(null);
  const [job,setJob]=useState<Job|null>(null);
  const [busy,setBusy]=useState(false);
  const [drag,setDrag]=useState(false);
  const inputRef=useRef<HTMLInputElement|null>(null);
  const preview=useMemo(()=>file?URL.createObjectURL(file):'', [file]);
  useEffect(()=>()=>{ if(preview) URL.revokeObjectURL(preview)},[preview]);
  useEffect(()=>{
    if(!job || ['done','failed'].includes(job.status)) return;
    const t=setInterval(async()=>{
      const res=await fetch(`${API_URL}/api/jobs/${job.job_id}`); setJob(await res.json());
    },900);
    return()=>clearInterval(t);
  },[job]);
  const upload=useCallback(async()=>{
    if(!file) return; setBusy(true);
    const fd=new FormData(); fd.append('file',file);
    try{ const res=await fetch(`${API_URL}/api/jobs`,{method:'POST',body:fd}); if(!res.ok) throw new Error(await res.text()); setJob(await res.json()); }
    catch(e){ alert(`Ошибка загрузки: ${e instanceof Error?e.message:String(e)}`); }
    finally{ setBusy(false); }
  },[file]);
  const pick=(f?:File)=>{ if(f && f.type.startsWith('video/')) {setFile(f); setJob(null)} };
  const download=(url?:string|null)=>{ if(url) window.location.href = `${API_URL}${url}`; };
  return <main>
    <section className="hero">
      <div className="nav"><div className="brand"><span className="logo">LT</span><b>ShelfVision</b></div><span className="pill">robot video → price tags → Excel</span></div>
      <div className="heroGrid">
        <div className="copy">
          <span className="eyebrow"><Sparkles size={16}/> Lenta Tech Life Hack</span>
          <h1>Видеоаналитика полки в стиле профессионального video processing tool</h1>
          <p>Загрузите видео с робота, запустите анализ ценников и скачайте структурированный отчет. Сейчас бэк работает через mock/sample adapter, но контракт уже готов под подключение CV/OCR нейросети.</p>
          <div className="stats"><Card icon={<ScanLine/>} label="29 полей" text="полный контракт результата"/><Card icon={<Cpu/>} label="FastAPI" text="очередь анализа и отчеты"/><Card icon={<ShieldCheck/>} label="local-ready" text="без внешних онлайн API"/></div>
        </div>
        <div className="panel upload" onDragOver={(e)=>{e.preventDefault(); setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={(e)=>{e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0])}}>
          <input ref={inputRef} type="file" accept="video/*" onChange={e=>pick(e.target.files?.[0])} hidden/>
          {!file ? <div className={drag?'drop active':'drop'} onClick={()=>inputRef.current?.click()}><UploadCloud size={56}/><h2>Перетащите видео сюда</h2><p>MP4, MOV, AVI, MKV, WEBM. Можно выбрать файл вручную.</p><button>Выбрать видео</button></div> : <div className="preview"><video src={preview} controls/><div><h3>{file.name}</h3><p>{(file.size/1024/1024).toFixed(1)} MB</p><button onClick={upload} disabled={busy}>{busy?'Загрузка...':'Запустить анализ'}</button><button className="ghost" onClick={()=>inputRef.current?.click()}>Заменить</button></div></div>}
        </div>
      </div>
    </section>
    <section className="workspace">
      <div className="panel status">
        <div className="sectionTitle"><Activity/> <h2>Pipeline</h2></div>
        <div className="timeline"><Step done={!!job} title="Upload"/><Step done={!!job&&job.progress>=25} title="Frame sampling"/><Step done={!!job&&job.progress>=55} title="Price-tag detection"/><Step done={!!job&&job.progress>=85} title="OCR + QR parse"/><Step done={job?.status==='done'} title="Report export"/></div>
        <div className="progress"><span style={{width:`${job?.progress||0}%`}}/></div>
        <p className="muted">{job?`Статус: ${job.status}, прогресс ${job.progress}%`:'Задача еще не запущена'}</p>
        {job?.error && <p className="error">{job.error}</p>}
      </div>
      <div className="panel result">
        <div className="sectionTitle"><FileSpreadsheet/> <h2>Результат</h2></div>
        <div className="metrics"><Metric k="Ценников" v={job?.rows_count ?? '—'}/><Metric k="Штрихкодов" v={(job?.metrics?.unique_barcodes as number) ?? '—'}/><Metric k="Confidence" v={(job?.metrics?.avg_confidence as number) ? `${Math.round(Number(job.metrics.avg_confidence)*100)}%`:'—'}/></div>
        <div className="downloads"><button disabled={job?.status!=='done'} onClick={()=>download(job?.xlsx_url)}><Download size={18}/> Excel XLSX</button><button disabled={job?.status!=='done'} onClick={()=>download(job?.csv_url)}><Download size={18}/> CSV</button></div>
      </div>
    </section>
    <section className="panel contract"><h2>Выходной контракт</h2><p>filename, product_name, price_default, price_card, price_discount, barcode, discount_amount, id_sku, print_datetime, code, additional_info, color, special_symbols, frame_timestamp, координаты bbox и QR-поля.</p></section>
  </main>
}
function Card({icon,label,text}:{icon:React.ReactNode,label:string,text:string}){return <div className="mini">{icon}<b>{label}</b><span>{text}</span></div>}
function Step({done,title}:{done:boolean,title:string}){return <div className={done?'step done':'step'}><span/><p>{title}</p></div>}
function Metric({k,v}:{k:string,v:React.ReactNode}){return <div className="metric"><b>{v}</b><span>{k}</span></div>}
createRoot(document.getElementById('root')!).render(<App/>);
