// Frontend/components/ui/SectorDonut.tsx
"use client";
import { useState } from "react";

const SECTOR_COLORS: Record<string, string> = {
  Technology: "hsl(217,91%,60%)", Financials: "hsl(142,71%,45%)",
  Healthcare: "hsl(280,70%,65%)", Energy: "hsl(38,92%,50%)",
  Consumer: "hsl(15,90%,58%)", Industrials: "hsl(195,80%,50%)",
  Utilities: "hsl(60,70%,50%)", Unknown: "hsl(215,15%,45%)",
};
const sc = (s: string) => SECTOR_COLORS[s] ?? SECTOR_COLORS.Unknown;

const R_OUTER=52, R_INNER=30, CX=60, CY=60;

function polarToCartesian(cx:number,cy:number,r:number,angleDeg:number){
  const rad=((angleDeg-90)*Math.PI)/180;
  return {x:cx+r*Math.cos(rad),y:cy+r*Math.sin(rad)};
}
function arcPath(startDeg:number,endDeg:number):string{
  const e=Math.min(endDeg,startDeg+359.99);
  const s1=polarToCartesian(CX,CY,R_OUTER,startDeg),e1=polarToCartesian(CX,CY,R_OUTER,e);
  const s2=polarToCartesian(CX,CY,R_INNER,e),e2=polarToCartesian(CX,CY,R_INNER,startDeg);
  const large=e-startDeg>180?1:0;
  return [`M ${s1.x} ${s1.y}`,`A ${R_OUTER} ${R_OUTER} 0 ${large} 1 ${e1.x} ${e1.y}`,
    `L ${s2.x} ${s2.y}`,`A ${R_INNER} ${R_INNER} 0 ${large} 0 ${e2.x} ${e2.y}`,"Z"].join(" ");
}

interface DonutSlice { sector:string; count:number; pct:number; startDeg:number; endDeg:number; }

function buildSlices(counts:Record<string,number>):DonutSlice[]{
  const total=Object.values(counts).reduce((a,b)=>a+b,0);
  if(!total) return [];
  let deg=0;
  return Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([sector,count])=>{
    const pct=count/total, span=pct*360;
    const s:DonutSlice={sector,count,pct,startDeg:deg,endDeg:deg+span};
    deg+=span; return s;
  });
}

function Donut({slices,hoveredSector,onHover,centerLabel}:{
  slices:DonutSlice[]; hoveredSector:string|null; onHover:(s:string|null)=>void; centerLabel:string;
}){
  return (
    <svg viewBox="0 0 120 120" className="w-24 h-24 flex-shrink-0">
      {slices.length===0
        ? <circle cx={CX} cy={CY} r={R_OUTER} fill="hsl(215,20%,16%)" />
        : slices.map(s=>(
          <path key={s.sector} d={arcPath(s.startDeg,s.endDeg)} fill={sc(s.sector)}
            opacity={hoveredSector===null||hoveredSector===s.sector?1:0.3}
            className="cursor-pointer transition-opacity"
            onMouseEnter={()=>onHover(s.sector)} onMouseLeave={()=>onHover(null)} />
        ))}
      <circle cx={CX} cy={CY} r={R_INNER-1} fill="hsl(215,25%,11%)" />
      <text x={CX} y={CY-4} textAnchor="middle" fontSize="7" fill="hsl(215,15%,55%)" fontWeight="500">{centerLabel}</text>
      <text x={CX} y={CY+6} textAnchor="middle" fontSize="9" fill="hsl(210,40%,88%)" fontWeight="700">{slices.length}</text>
      <text x={CX} y={CY+15} textAnchor="middle" fontSize="6" fill="hsl(215,15%,45%)">sectors</text>
    </svg>
  );
}

interface SectorDonutProps {
  currentSectors: Record<string,number>;
  previewSector: string|null;
  previewTicker: string|null;
}

export default function SectorDonut({currentSectors,previewSector,previewTicker}:SectorDonutProps){
  const [hovered,setHovered]=useState<string|null>(null);
  const currentSlices=buildSlices(currentSectors);
  const previewCounts={...currentSectors};
  if(previewSector) previewCounts[previewSector]=(previewCounts[previewSector]??0)+1;
  const previewSlices=buildSlices(previewCounts);
  const newSectors=new Set<string>();
  if(previewSector&&!(previewSector in currentSectors)) newSectors.add(previewSector);
  const hasPreview=previewSector!==null;

  return (
    <div className="rounded-xl p-5" style={{background:"hsl(215,25%,11%)",border:"1px solid hsl(215,20%,18%)"}}>
      <p className="text-sm font-semibold mb-1" style={{color:"hsl(210,40%,92%)"}}>Sector Distribution</p>
      <p className="text-xs mb-4" style={{color:"hsl(215,15%,50%)"}}>
        {hasPreview ? `Preview: adding ${previewTicker}` : "Hover a recommendation to preview"}
      </p>
      <div className={`grid gap-6 ${hasPreview?"grid-cols-2":"grid-cols-1"}`}>
        <div>
          <p className="text-xs font-medium mb-3 text-center" style={{color:"hsl(215,15%,50%)"}}>Current</p>
          <div className="flex items-center gap-4">
            <Donut slices={currentSlices} hoveredSector={hovered} onHover={setHovered} centerLabel="Now" />
            <div className="flex flex-col gap-1 min-w-0">
              {currentSlices.map(s=>(
                <div key={s.sector} className="flex items-center gap-2 cursor-pointer"
                  style={{opacity:hovered===null||hovered===s.sector?1:0.4}}
                  onMouseEnter={()=>setHovered(s.sector)} onMouseLeave={()=>setHovered(null)}>
                  <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{background:sc(s.sector)}} />
                  <span className="text-xs truncate" style={{color:"hsl(210,40%,85%)"}}>{s.sector}</span>
                  <span className="ml-auto text-xs font-semibold flex-shrink-0" style={{color:"hsl(215,15%,55%)"}}>{Math.round(s.pct*100)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        {hasPreview && (
          <div>
            <p className="text-xs font-medium mb-3 text-center" style={{color:"hsl(142,71%,45%)"}}>+ {previewTicker}</p>
            <div className="flex items-center gap-4">
              <Donut slices={previewSlices} hoveredSector={hovered} onHover={setHovered} centerLabel="After" />
              <div className="flex flex-col gap-1 min-w-0">
                {previewSlices.map(s=>(
                  <div key={s.sector} className="flex items-center gap-2 cursor-pointer"
                    style={{opacity:hovered===null||hovered===s.sector?1:0.4}}
                    onMouseEnter={()=>setHovered(s.sector)} onMouseLeave={()=>setHovered(null)}>
                    <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{background:sc(s.sector)}} />
                    <span className="text-xs truncate" style={{color:"hsl(210,40%,85%)"}}>{s.sector}</span>
                    {newSectors.has(s.sector) && (
                      <span className="text-xs px-1 py-0.5 rounded" style={{background:"hsla(142,71%,45%,0.15)",color:"hsl(142,71%,45%)",fontSize:"9px"}}>new</span>
                    )}
                    <span className="ml-auto text-xs font-semibold flex-shrink-0" style={{color:"hsl(215,15%,55%)"}}>{Math.round(s.pct*100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      {hasPreview && newSectors.size>0 && (
        <div className="mt-4 px-3 py-2 rounded-lg flex items-center gap-2"
          style={{background:"hsla(142,71%,45%,0.1)",border:"1px solid hsla(142,71%,45%,0.25)"}}>
          <span className="text-xs" style={{color:"hsl(142,71%,45%)"}}>
            ✦ Adds <strong>{Array.from(newSectors).join(", ")}</strong> — a sector not currently in your portfolio
          </span>
        </div>
      )}
      {hasPreview && newSectors.size===0 && previewSector && (
        <div className="mt-4 px-3 py-2 rounded-lg" style={{background:"hsla(38,92%,50%,0.08)",border:"1px solid hsla(38,92%,50%,0.2)"}}>
          <span className="text-xs" style={{color:"hsl(38,92%,50%)"}}>
            {previewTicker} is in <strong>{previewSector}</strong> — a sector you already hold.
          </span>
        </div>
      )}
    </div>
  );
}
