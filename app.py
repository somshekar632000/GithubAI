<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Repository AI Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios@1.6.7/dist/axios.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.22.5/Babel.min.js"></script>
    <style>
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 50%, #81d4fa 100%);
            min-height: 100vh;
            color: #1e293b;
            margin: 0;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect } = React;

        // Main App Component
        function App() {
            const [activeTab, setActiveTab] = useState('chat');
            const [chatHistory, setChatHistory] = useState([]);
            const [chatMessage, setChatMessage] = useState('');
            const [githubUrl, setGithubUrl] = useState('');
            const [indexName, setIndexName] = useState('');
            const [processingStatus, setProcessingStatus] = useState('');
            const [existingIndexes, setExistingIndexes] = useState([]);
            const [selectedIndex, setSelectedIndex] = useState('');
            const [indexInfo, setIndexInfo] = useState('Click "Load Available Repositories" to see your processed repositories');
            const [connectionStatus, setConnectionStatus] = useState('');
            const [isProcessing, setIsProcessing] = useState(false);
            const [isConnecting, setIsConnecting] = useState(false);

            // Fetch API status on mount
            useEffect(() => {
                axios.get('/api/status').then(response => {
                    setProcessingStatus(response.data.status);
                }).catch(error => {
                    setProcessingStatus(`Error fetching status: ${error.message}`);
                });
            }, []);

            // Handle chat submission
            const handleChatSubmit = async () => {
                if (!chatMessage.trim()) return;
                try {
                    const response = await axios.post('/api/chat', { message: chatMessage, history: chatHistory });
                    setChatHistory(response.data.history);
                    setChatMessage('');
                } catch (error) {
                    setChatHistory([...chatHistory, [chatMessage, `Error: ${error.message}]`]]);
                    setChatMessage('');
                }
            };

            // Handle process repository
            const handleProcessRepo = async () => {
                if (!githubUrl || !indexName) return;
                setIsProcessing(true);
                try {
                    const response = await axios.post('/api/process', { github_url: githubUrl, index_name: indexName });
                    setProcessingStatus(response.data.status);
                    setIsProcessing(false);
                } catch (error) {
                    setProcessingStatus(`Error: ${error.message}`);
                    setIsProcessing(false);
                }
            };

            // Handle refresh indexes
            const handleRefreshIndexes = async () => {
                try {
                    const response = await axios.get('/api/refresh_indexes');
                    setExistingIndexes(response.data.choices || []);
                    setIndexInfo(response.data.info);
                } catch (error) {
                    setExistingIndexes([]);
                    setIndexInfo(`Error: ${error.message}`);
                }
            };

            // Handle connect to index
            const handleConnectIndex = async () => {
                if (!selectedIndex) {
                    setConnectionStatus('Please select a repository first.');
                    return;
                }
                setIsConnecting(true);
                try {
                    const response = await axios.post('/api/connect_index', { index_name: selectedIndex });
                    setConnectionStatus(response.data.status);
                    setIsConnecting(false);
                } catch (error) {
                    setConnectionStatus(`Error: ${error.message}`);
                    setIsConnecting(false);
                }
            };

            // Clear chat history
            const clearChat = () => {
                setChatHistory([]);
            };

            return (
                <div className="max-w-6xl mx-auto p-5 bg-white bg-opacity-95 backdrop-blur-xl rounded-3xl shadow-2xl my-5">
                    {/* Header */}
                    <div className="text-center py-8 border-b border-gray-200/20 mb-8">
                        <h1 className="text-5xl font-bold bg-gradient-to-r from-sky-500 to-indigo-600 bg-clip-text text-transparent">
                            Github AI Assistant
                        </h1>
                        <p className="text-gray-600 text-base mt-3 max-w-xl mx-auto">
                            A Learning tool to Accelerate the understanding of large Github Repositories. Get instant insights into code functionality and code structure.
                        </p>
                    </div>

                    {/* API Status */}
                    <div className="flex justify-center gap-4 mb-8 flex-wrap">
                        <div className="bg-white bg-opacity-80 backdrop-blur-md border border-white/30 rounded-xl px-5 py-3 text-sm font-medium text-gray-700 shadow-md hover:shadow-lg transition-all">
                            {processingStatus.includes('LLM Connected') ? '✅ LLM Connected' : '❌ LLM Not Connected'}
                        </div>
                        <div className="bg-white bg-opacity-80 backdrop-blur-md border border-white/30 rounded-xl px-5 py-3 text-sm font-medium text-gray-700 shadow-md hover:shadow-lg transition-all">
                            {processingStatus.includes('Database Connected') ? '✅ Database Connected' : '❌ Database Not Connected'}
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="bg-white bg-opacity-60 backdrop-blur-md rounded-2xl p-1.5 mb-6 shadow-md">
                        <div className="flex gap-1">
                            <button
                                className={`flex-1 py-3 px-6 rounded-xl text-sm font-medium transition-all ${
                                    activeTab === 'chat' ? 'bg-white text-sky-500 shadow-sm' : 'text-gray-600 hover:bg-white/70'
                                }`}
                                onClick={() => setActiveTab('chat')}
                            >
                                💬 Chat
                            </button>
                            <button
                                className={`flex-1 py-3 px-6 rounded-xl text-sm font-medium transition-all ${
                                    activeTab === 'process' ? 'bg-white text-sky-500 shadow-sm' : 'text-gray-600 hover:bg-white/70'
                                }`}
                                onClick={() => setActiveTab('process')}
                            >
                                🚀 Process Repository
                            </button>
                            <button
                                className={`flex-1 py-3 px-6 rounded-xl text-sm font-medium transition-all ${
                                    activeTab === 'connect' ? 'bg-white text-sky-500 shadow-sm' : 'text-gray-600 hover:bg-white/70'
                                }`}
                                onClick={() => setActiveTab('connect')}
                            >
                                📂 Connect to Repository
                            </button>
                        </div>
                    </div>

                    {/* Tab Content */}
                    {activeTab === 'chat' && (
                        <div className="bg-white bg-opacity-70 backdrop-blur-md rounded-2xl p-6 shadow-lg">
                            <div className="h-[500px] overflow-y-auto mb-4">
                                {chatHistory.map((chat, index) => (
                                    <div key={index} className="mb-4 animate-fadeInUp">
                                        <div className="bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-2xl rounded-br-sm p-4 max-w-[75%] ml-auto shadow-md">
                                            {chat[0]}
                                        </div>
                                        <div className="bg-white bg-opacity-90 backdrop-blur-md border border-white/30 rounded-2xl rounded-bl-sm p-4 max-w-[85%] mr-auto mt-2 shadow-md">
                                            {chat[1]}
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="bg-white bg-opacity-90 backdrop-blur-md border border-white/30 rounded-2xl p-5 shadow-lg flex gap-3">
                                <textarea
                                    className="flex-1 min-h-[48px] max-h-[120px] border-2 border-gray-200/20 rounded-2xl p-4 text-sm text-gray-800 bg-white bg-opacity-90 backdrop-blur-md focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 resize-none"
                                    placeholder="Ask me anything about the repository..."
                                    value={chatMessage}
                                    onChange={(e) => setChatMessage(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleChatSubmit()}
                                />
                                <button
                                    className="min-w-[48px] h-12 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-2xl flex items-center justify-center font-semibold shadow-md hover:shadow-lg hover:from-sky-600 hover:to-blue-700 transition-all"
                                    onClick={handleChatSubmit}
                                >
                                    Send
                                </button>
                            </div>
                            <button
                                className="mt-3 bg-transparent border-2 border-gray-200/20 rounded-xl px-4 py-2 text-sm text-gray-600 hover:bg-gray-50/80 hover:border-gray-300 transition-all"
                                onClick={clearChat}
                            >
                                Clear conversation
                            </button>
                            <div className="mt-4 bg-gray-50/90 backdrop-blur-md border border-gray-200/50 rounded-2xl p-5">
                                <h4 className="text-sm font-semibold text-gray-700 mb-3">💡 Example questions</h4>
                                <ul className="text-sm text-gray-600 list-disc pl-5">
                                    <li>What is the main purpose of this repository?</li>
                                    <li><code>explain filename.py</code> or <code>filename.ipynb</code></li>
                                    <li><code>explain filepath</code></li>
                                    <li><code>Explain Cell 3 in filename.ipynb</code> or <code>filepath</code></li>
                                    <li><code>explain line 55 to line 65 in filename.py</code> or <code>filepath</code></li>
                                    <li><code>explain some_function</code> or <code>class class_name</code></li>
                                    <li><code>explain some_function</code> or <code>class class_name in filename.py/filename.ipynb/filepath</code></li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {activeTab === 'process' && (
                        <div className="bg-white bg-opacity-70 backdrop-blur-md rounded-2xl p-8 shadow-lg">
                            <h3 className="text-xl font-semibold text-gray-800 mb-2">Process a New Repository</h3>
                            <p className="text-gray-600 mb-6">Index a GitHub repository to enable AI-powered chat and analysis.</p>
                            <div className="space-y-6">
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">GitHub Repository URL</label>
                                    <input
                                        type="text"
                                        className="w-full p-4 border-2 border-gray-200/20 rounded-xl text-sm text-gray-800 bg-white bg-opacity-90 backdrop-blur-md focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10"
                                        placeholder="https://github.com/username/repository"
                                        value={githubUrl}
                                        onChange={(e) => setGithubUrl(e.target.value)}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">Index Name</label>
                                    <input
                                        type="text"
                                        className="w-full p-4 border-2 border-gray-200/20 rounded-xl text-sm text-gray-800 bg-white bg-opacity-90 backdrop-blur-md focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10"
                                        placeholder="Enter Index name"
                                        value={indexName}
                                        onChange={(e) => setIndexName(e.target.value)}
                                    />
                                </div>
                                <button
                                    className={`w-full bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-xl p-4 text-sm font-semibold shadow-md hover:shadow-lg hover:from-sky-600 hover:to-blue-700 transition-all ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
                                    onClick={handleProcessRepo}
                                    disabled={isProcessing}
                                >
                                    {isProcessing ? 'Processing...' : '🚀 Start Processing'}
                                </button>
                                <button
                                    className="w-full bg-white bg-opacity-90 border-2 border-gray-200/20 rounded-xl p-4 text-sm font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-all"
                                    onClick={() => axios.get('/api/status').then(res => setProcessingStatus(res.data.status))}
                                >
                                    🔄 Refresh Status
                                </button>
                                <textarea
                                    className="w-full h-40 p-4 border-2 border-gray-200/20 rounded-xl text-sm text-gray-600 bg-gray-50 bg-opacity-90 backdrop-blur-md font-mono focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10"
                                    value={processingStatus}
                                    readOnly
                                />
                            </div>
                            <div className="mt-4 bg-gray-50/90 backdrop-blur-md border border-gray-200/50 rounded-2xl p-5">
                                <h4 className="text-sm font-semibold text-gray-700 mb-3">ℹ️ Processing Information</h4>
                                <ul className="text-sm text-gray-600 list-disc pl-5">
                                    <li><strong>Repository Analysis:</strong> Clone and analyze repository structure</li>
                                    <li><strong>Code Embedding:</strong> Generate AI embeddings for code chunks</li>
                                    <li><strong>Vector Storage:</strong> Store embeddings in Vector database</li>
                                    <li><strong>Index Creation:</strong> Create searchable index for fast retrieval</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {activeTab === 'connect' && (
                        <div className="bg-white bg-opacity-70 backdrop-blur-md rounded-2xl p-8 shadow-lg">
                            <h3 className="text-xl font-semibold text-gray-800 mb-2">Connect to Previously Processed Repository</h3>
                            <p className="text-gray-600 mb-6">Instantly connect to repositories you've already processed.</p>
                            <div className="space-y-6">
                                <button
                                    className="w-full bg-white bg-opacity-90 border-2 border-gray-200/20 rounded-xl p-4 text-sm font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-all"
                                    onClick={handleRefreshIndexes}
                                >
                                    🔄 Load Available Repositories
                                </button>
                                <textarea
                                    className="w-full h-20 p-4 border-2 border-gray-200/20 rounded-xl text-sm text-gray-600 bg-gray-50 bg-opacity-90 backdrop-blur-md font-mono focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10"
                                    value={indexInfo}
                                    readOnly
                                />
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 mb-2">Select Repository</label>
                                    <select
                                        className="w-full p-4 border-2 border-gray-200/20 rounded-xl text-sm text-gray-800 bg-white bg-opacity-90 backdrop-blur-md focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 cursor-pointer"
                                        value={selectedIndex}
                                        onChange={(e) => setSelectedIndex(e.target.value)}
                                    >
                                        <option value="">Select a repository</option>
                                        {existingIndexes.map(index => (
                                            <option key={index} value={index}>{index}</option>
                                        ))}
                                    </select>
                                </div>
                                <button
                                    className={`w-full bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-xl p-4 text-sm font-semibold shadow-md hover:shadow-lg hover:from-sky-600 hover:to-blue-700 transition-all ${isConnecting ? 'opacity-50 cursor-not-allowed' : ''}`}
                                    onClick={handleConnectIndex}
                                    disabled={isConnecting}
                                >
                                    {isConnecting ? 'Connecting...' : '🔗 Connect to Repository'}
                                </button>
                                <textarea
                                    className="w-full h-20 p-4 border-2 border-gray-200/20 rounded-xl text-sm text-gray-600 bg-gray-50 bg-opacity-90 backdrop-blur-md font-mono focus:outline-none focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10"
                                    value={connectionStatus}
                                    readOnly
                                />
                            </div>
                            <div className="mt-4 bg-gray-50/90 backdrop-blur-md border border-gray-200/50 rounded-2xl p-5">
                                <h4 className="text-sm font-semibold text-gray-700 mb-3">✨ Benefits of using existing repositories</h4>
                                <ul className="text-sm text-gray-600 list-disc pl-5">
                                    <li><strong>⚡ Instant Access:</strong> No waiting for processing</li>
                                    <li><strong>💰 Time and effective:</strong> No re-embedding required saves the most valuable resource: TIME</li>
                                    <li><strong>🔄 Previous files:</strong> All your previous files that are processed preserved</li>
                                    <li><strong>🚀 Fast Startup:</strong> Begin chatting immediately</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {/* Developer Card */}
                    <div className="fixed bottom-5 right-5 bg-white bg-opacity-90 backdrop-blur-md rounded-2xl p-4 shadow-lg hover:shadow-xl transition-all z-50">
                        <div className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                            👨‍💻 Somshekar M
                        </div>
                        <a
                            href="https://www.linkedin.com/in/somshekar-m"
                            target="_blank"
                            className="flex items-center gap-1.5 text-sm text-sky-500 hover:text-sky-600 bg-sky-500/10 hover:bg-sky-500/20 rounded-lg px-2 py-1 mt-1 transition-all"
                        >
                            🔗 LinkedIn Profile
                        </a>
                    </div>
                </div>
            );
        }

        // Render the app
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>