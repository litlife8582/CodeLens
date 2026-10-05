import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, Search, Code2, Database, Network, Activity, Zap, CheckCircle, AlertTriangle, GitBranch, Send, Bot, User, Wand2 } from 'lucide-react';
import toast, { Toaster } from 'react-hot-toast';

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [githubUrl, setGithubUrl] = useState('');
  const [githubStatus, setGithubStatus] = useState('');
  
  // Chat State
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const chatEndRef = useRef(null);

  const [chatMode, setChatMode] = useState('showdown'); // 'showdown' or 'agent'
  const [activeMetrics, setActiveMetrics] = useState(null); // stores the metrics of the latest/selected query

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isQuerying]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus('uploading');
    const toastId = toast.loading('Uploading and ingesting ZIP codebase...');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      await axios.post('http://127.0.0.1:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadStatus('');
      setFile(null);
      toast.success('Codebase ingested successfully!', { id: toastId });
    } catch (err) {
      console.error(err);
      setUploadStatus('');
      toast.error('Failed to upload ZIP. Please ensure the backend is running.', { id: toastId });
    }
  };

  const handleGithubUpload = async () => {
    if (!githubUrl) return;
    setGithubStatus('uploading');
    const toastId = toast.loading('Cloning and ingesting GitHub repository...');
    
    try {
      await axios.post('http://127.0.0.1:8000/upload-github', { repo_url: githubUrl });
      setGithubStatus('');
      setGithubUrl('');
      toast.success('GitHub repository ingested successfully!', { id: toastId });
    } catch (err) {
      console.error(err);
      setGithubStatus('');
      toast.error('Failed to clone GitHub repository. Please check the URL.', { id: toastId });
    }
  }

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    const userMessage = { type: 'user', text: query };
    setChatHistory(prev => [...prev, userMessage]);
    const currentQuery = query;
    setQuery('');
    setIsQuerying(true);
    
    try {
      if (chatMode === 'showdown') {
        const res = await axios.post('http://127.0.0.1:8000/query', { query: currentQuery });
        const results = res.data;
        
        const botMessage = { 
          type: 'bot', 
          mode: 'showdown',
          vectorText: results.vector_rag?.answer,
          graphText: results.graph_rag?.answer,
          results: results
        };
        
        setChatHistory(prev => [...prev, botMessage]);
        setActiveMetrics(results);
      } else {
        const res = await axios.post('http://127.0.0.1:8000/generate-code', { request: currentQuery });
        const generatedCode = res.data.generated_code;
        
        const botMessage = {
          type: 'bot',
          mode: 'agent',
          text: generatedCode
        };
        setChatHistory(prev => [...prev, botMessage]);
      }
    } catch (err) {
      console.error(err);
      
      let errorMsg = "An error occurred while connecting to the server.";
      let advice = "Please check if your backend is running.";
      
      if (err.response?.status === 429) {
        errorMsg = "API Rate Limit Exceeded!";
        advice = "You have made too many requests. Please wait a minute before trying again.";
      } else if (err.response?.status >= 500) {
        errorMsg = "Backend Server Error";
        advice = "Check your backend terminal for the crash traceback.";
      }
      
      toast.error(
        <div>
          <strong>{errorMsg}</strong>
          <div className="text-sm opacity-90 mt-1">{advice}</div>
        </div>, 
        { duration: 5000 }
      );
      
      setChatHistory(prev => [...prev, { type: 'bot', text: `Error: ${errorMsg}\n\n${advice}`, error: true }]);
    } finally {
      setIsQuerying(false);
    }
  };

  const ScoreCard = ({ title, score, icon: Icon, color }) => (
    <div className={`p-3 rounded-lg border ${color} bg-surface/50 flex flex-col gap-1`}>
      <div className="flex items-center gap-1.5 text-slate-400 font-medium text-xs">
        <Icon size={14} />
        <span>{title}</span>
      </div>
      <div className="text-xl font-bold">{score !== undefined ? score : '-'}</div>
    </div>
  );

  return (
    <div className="min-h-screen p-6 flex flex-col items-center">
      <Toaster position="bottom-right" toastOptions={{ 
        style: { background: '#1e293b', color: '#f8fafc', border: '1px solid #334155' } 
      }} />
      
      {/* Header */}
      <header className="w-full max-w-[1400px] flex justify-between items-center mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/20 rounded-lg text-primary">
            <Code2 size={28} />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Code<span className="gradient-text">Lens</span></h1>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Activity size={16} className="text-emerald-400" /> API Online
        </div>
      </header>

      <main className="w-full max-w-[1400px] grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-140px)]">
        
        {/* Left Sidebar */}
        <div className="flex flex-col gap-6 lg:col-span-1 h-full overflow-y-auto pr-2 custom-scrollbar">
          
          <section className="glass-panel p-5 flex flex-col gap-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Database size={18} className="text-secondary" />
              Knowledge Base
            </h2>
            
            <div className="border border-dashed border-border rounded-lg p-4 flex flex-col items-center justify-center gap-2 hover:border-primary/50 transition-colors cursor-pointer relative bg-surface/30">
              <input 
                type="file" 
                className="absolute inset-0 opacity-0 cursor-pointer" 
                onChange={handleFileChange}
              />
              <Upload size={20} className="text-slate-400" />
              <span className="text-xs text-slate-300 font-medium text-center">
                {file ? file.name : "Upload ZIP codebase"}
              </span>
            </div>
            
            <button 
              onClick={handleUpload}
              disabled={!file || uploadStatus === 'uploading'}
              className="w-full py-2 bg-primary hover:bg-blue-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-sm font-medium transition-colors"
            >
              {uploadStatus === 'uploading' ? 'Processing...' : uploadStatus === 'success' ? 'Uploaded!' : 'Ingest ZIP'}
            </button>

            <div className="relative flex py-2 items-center">
                <div className="flex-grow border-t border-border"></div>
                <span className="flex-shrink-0 mx-4 text-slate-500 text-xs">OR</span>
                <div className="flex-grow border-t border-border"></div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="relative">
                <GitBranch size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input 
                  type="text" 
                  value={githubUrl}
                  onChange={(e) => setGithubUrl(e.target.value)}
                  placeholder="https://github.com/..."
                  className="w-full bg-surface border border-border rounded-lg py-2 pl-9 pr-3 text-sm focus:border-secondary focus:outline-none"
                />
              </div>
              <button 
                onClick={handleGithubUpload}
                disabled={!githubUrl || githubStatus === 'uploading'}
                className="w-full py-2 bg-secondary hover:bg-indigo-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-sm font-medium transition-colors"
              >
                {githubStatus === 'uploading' ? 'Cloning...' : githubStatus === 'success' ? 'Cloned!' : 'Ingest GitHub'}
              </button>
            </div>

          </section>

          {/* Architecture Info */}
          <section className="glass-panel p-5">
            <h3 className="text-xs font-semibold mb-3 text-slate-300 uppercase tracking-wider">Metrics Guide</h3>
            <ul className="space-y-3 text-xs text-slate-400">
              <li className="flex gap-2">
                <CheckCircle size={14} className="text-emerald-400 shrink-0 mt-0.5" />
                <span>Accuracy measures if the generated answer correctly resolves the query.</span>
              </li>
              <li className="flex gap-2">
                <AlertTriangle size={14} className="text-amber-400 shrink-0 mt-0.5" />
                <span>Hallucination score detects non-factual or fabricated information.</span>
              </li>
            </ul>
          </section>

        </div>

        {/* Center Content - Chat Interface */}
        <div className="flex flex-col lg:col-span-2 glass-panel overflow-hidden h-full">
          {/* Chat History */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
            {chatHistory.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4">
                <div className="w-16 h-16 rounded-full bg-surface flex items-center justify-center mb-2">
                  <Wand2 size={32} className="text-primary/50" />
                </div>
                <h3 className="text-lg font-medium text-slate-300">Start exploring your codebase</h3>
                <p className="text-sm text-center max-w-sm">
                  Ask multi-hop questions to compare Vector RAG vs AST Graph RAG performance and accuracy.
                </p>
              </div>
            ) : (
              chatHistory.map((msg, idx) => (
                <div key={idx} className={`flex gap-4 ${msg.type === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.type === 'user' ? 'bg-primary text-white' : 'bg-surface border border-border text-emerald-400'}`}>
                    {msg.type === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className={`flex flex-col max-w-[85%] ${msg.type === 'user' ? 'items-end' : 'items-start'}`}>
                    <div className={`p-4 rounded-2xl text-sm ${msg.type === 'user' ? 'bg-primary text-white rounded-tr-sm' : 'bg-surface border border-border text-slate-300 rounded-tl-sm'}`}>
                      {msg.type === 'bot' && !msg.error ? (
                        msg.mode === 'showdown' ? (
                         <div className="flex flex-col gap-4">
                            <div>
                              <div className="text-xs font-semibold text-emerald-400 mb-1 flex items-center gap-1"><Network size={12}/> AST Graph-RAG Answer:</div>
                              <p className="whitespace-pre-wrap leading-relaxed">{msg.graphText}</p>
                            </div>
                            <div className="border-t border-border/50 pt-3">
                              <div className="text-xs font-semibold text-blue-400 mb-1 flex items-center gap-1"><Database size={12}/> Vector-RAG Answer:</div>
                              <p className="whitespace-pre-wrap leading-relaxed">{msg.vectorText}</p>
                            </div>
                         </div>
                        ) : (
                          <div className="flex flex-col gap-2">
                             <div className="text-xs font-semibold text-secondary mb-1 flex items-center gap-1"><Wand2 size={12}/> Agent Generated Code:</div>
                             <pre className="whitespace-pre-wrap font-mono text-xs bg-slate-900 p-3 rounded border border-border text-emerald-400 overflow-x-auto">{msg.text}</pre>
                          </div>
                        )
                      ) : (
                        <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
            
            {isQuerying && (
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-surface border border-border text-emerald-400 flex items-center justify-center shrink-0">
                  <Bot size={16} />
                </div>
                <div className="bg-surface border border-border p-4 rounded-2xl rounded-tl-sm flex items-center gap-2">
                  <Activity size={16} className="text-primary animate-pulse" />
                  <span className="text-sm text-slate-400">Analyzing codebase...</span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Chat Input */}
          <div className="p-4 border-t border-border bg-background/50 flex flex-col gap-3">
            
            <div className="flex gap-2">
              <button 
                onClick={() => setChatMode('showdown')}
                className={`text-xs px-3 py-1.5 rounded-full font-medium transition-colors flex items-center gap-1 ${chatMode === 'showdown' ? 'bg-primary text-white' : 'bg-surface border border-border text-slate-400 hover:text-slate-300'}`}
              >
                <Zap size={12} /> Hallucination Showdown
              </button>
              <button 
                onClick={() => setChatMode('agent')}
                className={`text-xs px-3 py-1.5 rounded-full font-medium transition-colors flex items-center gap-1 ${chatMode === 'agent' ? 'bg-secondary text-white' : 'bg-surface border border-border text-slate-400 hover:text-slate-300'}`}
              >
                <Wand2 size={12} /> Code Agent
              </button>
            </div>

            <form onSubmit={handleQuery} className="relative flex items-center">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={chatMode === 'showdown' ? "Ask about your code..." : "Request a new feature (e.g. Add dark mode)..."}
                className="w-full bg-surface border border-border rounded-full py-3 pl-5 pr-14 text-sm focus:outline-none focus:border-primary transition-colors"
                disabled={isQuerying}
              />
              <button
                type="submit"
                disabled={!query.trim() || isQuerying}
                className="absolute right-2 p-2 bg-primary text-white rounded-full hover:bg-blue-600 disabled:opacity-50 disabled:hover:bg-primary transition-colors"
              >
                <Send size={16} />
              </button>
            </form>
          </div>
        </div>

        {/* Right Sidebar - Metrics */}
        <div className="flex flex-col gap-6 lg:col-span-1 h-full overflow-y-auto pl-2 custom-scrollbar">
          <section className="glass-panel p-5 flex flex-col gap-4 h-full">
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-2">
              <Activity size={18} className="text-amber-400" />
              Live Showdown Metrics
            </h2>
            
            {!activeMetrics ? (
              <div className="flex-1 flex items-center justify-center text-slate-500 text-sm text-center p-4 border border-dashed border-border rounded-lg bg-surface/30">
                Send a query in the chat to see real-time hallucination and accuracy comparisons.
              </div>
            ) : (
              <div className="flex flex-col gap-6">
                
                {/* Graph RAG Metrics */}
                <div className="flex flex-col gap-3 relative">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]"></div>
                    <h3 className="font-bold text-slate-200">AST Graph-RAG</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <ScoreCard title="Accuracy" score={activeMetrics.graph_rag?.accuracy_score} icon={CheckCircle} color="border-emerald-500/30 text-emerald-400 bg-emerald-500/5" />
                    <ScoreCard title="Hallucination" score={activeMetrics.graph_rag?.hallucination_score} icon={AlertTriangle} color="border-amber-500/30 text-amber-400 bg-amber-500/5" />
                  </div>
                </div>

                <div className="w-full h-px bg-border my-2"></div>

                {/* Vector RAG Metrics */}
                <div className="flex flex-col gap-3 relative">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]"></div>
                    <h3 className="font-bold text-slate-200">Vector-RAG</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <ScoreCard title="Accuracy" score={activeMetrics.vector_rag?.accuracy_score} icon={CheckCircle} color="border-blue-500/30 text-blue-400 bg-blue-500/5" />
                    <ScoreCard title="Hallucination" score={activeMetrics.vector_rag?.hallucination_score} icon={AlertTriangle} color="border-amber-500/30 text-amber-400 bg-amber-500/5" />
                  </div>
                </div>

              </div>
            )}
          </section>
        </div>

      </main>
    </div>
  );
}

export default App;
