// In local development, use Vite's /api proxy. Set VITE_API_URL only for a separately hosted API.
const API = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');
function networkError(error: unknown): Error {
  const detail = error instanceof Error ? error.message : String(error);
  if (detail === 'Failed to fetch' || detail.includes('NetworkError') || detail.includes('fetch')) {
    return new Error(`Cannot reach the Stock Sense backend. Start it in a separate terminal with: cd backend && uvicorn main:app --reload --port 8000. API address: ${API}`);
  }
  return error instanceof Error ? error : new Error(detail);
}
async function request<T>(path:string):Promise<T>{
  let response: Response;
  try { response=await fetch(`${API}${path}`); } catch (error) { throw networkError(error); }
  if(!response.ok){let message=`Request failed (${response.status})`;try{const body=await response.json();message=body.detail||message}catch{}throw new Error(message)}
  return response.json();
}
export const api={
  search:(q:string)=>request<{results:any[]}>(`/search?q=${encodeURIComponent(q)}`),
  markets:()=>request<any>('/markets'),
  quotes:(symbols:string[])=>request<any>(`/quotes?symbols=${encodeURIComponent(symbols.join(','))}`),
  stock:(symbol:string,period='1y')=>request<any>(`/stock/${encodeURIComponent(symbol)}?period=${period}`),
  fundamentals:(symbol:string)=>request<any>(`/stock/${encodeURIComponent(symbol)}/fundamentals`),
  news:(symbol:string)=>request<any>(`/stock/${encodeURIComponent(symbol)}/news`),
  ml:(symbol:string)=>request<any>(`/stock/${encodeURIComponent(symbol)}/ml`),
  forecast:(symbol:string,horizon:string)=>request<any>(`/stock/${encodeURIComponent(symbol)}/forecast?horizon=${encodeURIComponent(horizon)}`),
  risk:(symbol:string)=>request<any>(`/stock/${encodeURIComponent(symbol)}/risk`),
  analyzeDataset:async(file:File,mode:'single'|'portfolio')=>{
    const body=new FormData(); body.append('file',file); body.append('mode',mode);
    let response: Response;
    try { response=await fetch(`${API}/datasets/analyze`,{method:'POST',body}); } catch (error) { throw networkError(error); }
    if(!response.ok){let message=`Analysis failed (${response.status})`;try{message=(await response.json()).detail||message}catch{}throw new Error(message)}
    return response.json();
  },
  ask:async(symbol:string,question:string)=>{let response:Response;try{response=await fetch(`${API}/stock/${encodeURIComponent(symbol)}/ask`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})})}catch(error){throw networkError(error)};if(!response.ok){let message=`Request failed (${response.status})`;try{message=(await response.json()).detail||message}catch{}throw new Error(message)}return response.json()},
  watchlist:()=>request<any>('/watchlist'),
};
