import { useCallback, useEffect, useMemo, useState } from "react"

type Role = "user" | "ai"
interface Message {
  id: string
  role: Role
  content: string
}

interface DefectReport {
  defect_id: string
  description: string
  ratt_branch: string
  likelihood: number
  severity: number
  risk_score: number
  long_term_consequence: string
}

interface ChatResponse {
  session_id: string
  response: string
  conversation_length: number
}

interface AnalyzeResponse {
  session_id: string
  story?: string
  top_defect?: DefectReport
}

const API_BASE = "http://localhost:8000"

const ChatWindow = ({ messages }: { messages: Message[] }) => (
  <div className="flex-1 overflow-y-auto rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
    {messages.length === 0 && (
      <p className="text-sm text-slate-500">还没有对话，先介绍一下你的世界吧。</p>
    )}
    <div className="space-y-3">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={`rounded-xl px-4 py-2 text-sm leading-relaxed ${
            msg.role === "user" ? "bg-indigo-50 text-slate-900" : "bg-slate-900 text-white"
          }`}
        >
          <p className="font-semibold">{msg.role === "user" ? "你" : "SEE 模块"}</p>
          <p>{msg.content}</p>
        </div>
      ))}
    </div>
  </div>
)

const MessageInput = ({
  disabled,
  onSend,
}: {
  disabled: boolean
  onSend: (message: string) => Promise<void>
}) => {
  const [value, setValue] = useState("")

  const handleSend = async () => {
    if (!value.trim()) return
    const message = value.trim()
    setValue("")
    await onSend(message)
  }

  return (
    <div className="mt-4 flex gap-3">
      <textarea
        className="min-h-[80px] flex-1 rounded-2xl border border-slate-200 p-4 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
        placeholder="描述你的世界、规则或新灵感……"
        value={value}
        onChange={(evt) => setValue(evt.target.value)}
        disabled={disabled}
      />
      <button
        className="h-[80px] rounded-2xl bg-indigo-600 px-6 text-white shadow-lg transition hover:bg-indigo-700 disabled:opacity-50"
        onClick={handleSend}
        disabled={disabled}
      >
        发送
      </button>
    </div>
  )
}

const AnalyzeButton = ({ disabled, onAnalyze }: { disabled: boolean; onAnalyze: () => Promise<void> }) => (
  <button
    className="mt-6 w-full rounded-2xl bg-rose-600 px-4 py-3 text-lg font-semibold text-white shadow-lg transition hover:bg-rose-700 disabled:opacity-60"
    onClick={onAnalyze}
    disabled={disabled}
  >
    分析我的世界
  </button>
)

const StoryDisplay = ({
  story,
  defect,
}: {
  story?: string
  defect?: DefectReport
}) => {
  if (!story) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
        尚未生成故事。点击“分析我的世界”即可触发 RATT + CDNG 工作流。
      </div>
    )
  }

  return (
    <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      {defect && (
        <div className="rounded-2xl bg-slate-900 p-4 text-white">
          <p className="text-xs uppercase tracking-widest text-slate-300">检测到的核心缺陷</p>
          <p className="text-lg font-semibold">{defect.description}</p>
          <p className="mt-2 text-sm text-slate-200">{defect.long_term_consequence}</p>
          <p className="mt-2 text-xs">风险评分：{defect.risk_score}（可能性 {defect.likelihood} / 影响 {defect.severity}）</p>
        </div>
      )}
      <article className="whitespace-pre-line text-slate-800">{story}</article>
    </div>
  )
}

function App() {
  const [sessionId, setSessionId] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [story, setStory] = useState<string | undefined>()
  const [topDefect, setTopDefect] = useState<DefectReport | undefined>()
  const [seeVariant, setSeeVariant] = useState<"default" | "simple">("default")
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const initSession = useCallback(async () => {
    const res = await fetch(`${API_BASE}/api/v1/session/init`, { method: "POST" })
    if (!res.ok) {
      throw new Error("session init failed")
    }
    const data = await res.json()
    setSessionId(data.session_id)
    // reset local buffers
    setMessages([])
    setStory(undefined)
    setTopDefect(undefined)
  }, [])

  useEffect(() => {
    initSession().catch((err) => {
      console.error(err)
      setError("无法初始化会话，请确认后端是否已启动。")
    })
  }, [initSession])

  const apiHeaders = useMemo(
    () => ({
      "Content-Type": "application/json",
    }),
    []
  )

  const handleSend = useCallback(
    async (message: string) => {
      if (!sessionId) return
      setLoading(true)
      setError(null)
      try {
        const endpoint = seeVariant === "simple" ? "/api/v1/chat_simple" : "/api/v1/chat"
        const res = await fetch(`${API_BASE}${endpoint}`, {
          method: "POST",
          headers: apiHeaders,
          body: JSON.stringify({ session_id: sessionId, message }),
        })
        if (!res.ok) {
          throw new Error("Chat endpoint failed")
        }
        const data = (await res.json()) as ChatResponse
        setMessages((prev) => [
          ...prev,
          { id: crypto.randomUUID(), role: "user", content: message },
          { id: crypto.randomUUID(), role: "ai", content: data.response },
        ])
      } catch (err) {
        console.error(err)
        setError("发送失败，请稍后再试。")
      } finally {
        setLoading(false)
      }
    },
    [apiHeaders, seeVariant, sessionId]
  )

  const handleAnalyze = useCallback(async () => {
    if (!sessionId) return
    setAnalyzing(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/v1/analyze`, {
        method: "POST",
        headers: apiHeaders,
        body: JSON.stringify({ session_id: sessionId }),
      })
      if (!res.ok) {
        throw new Error("Analyze endpoint failed")
      }
      const data = (await res.json()) as AnalyzeResponse
      setStory(data.story)
      setTopDefect(data.top_defect)
    } catch (err) {
      console.error(err)
      setError("分析失败，请确认后端是否运行。")
    } finally {
      setAnalyzing(false)
    }
  }, [apiHeaders, sessionId])

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <header>
        <p className="text-sm font-semibold text-indigo-600">Creative Workshop · V1</p>
        <h1 className="mt-1 text-3xl font-bold text-slate-900">创意工坊：苏格拉底反馈回路</h1>
        <p className="mt-2 text-sm text-slate-500">
          与 SEE 对话来精炼你的世界，然后让 CDA + CDNG 揭示潜在缺陷，并以故事的形式呈现。
        </p>
        <div className="mt-3 inline-flex items-center gap-2 rounded-full bg-slate-100 p-1 text-xs text-slate-600">
          <span className="px-2">SEE 版本</span>
          <button
            type="button"
            onClick={() => setSeeVariant("default")}
            className={`rounded-full px-3 py-1 ${
              seeVariant === "default" ? "bg-white text-slate-900 shadow" : "text-slate-500"
            }`}
          >
            标准（含空白/冲突分析）
          </button>
          <button
            type="button"
            onClick={() => setSeeVariant("simple")}
            className={`rounded-full px-3 py-1 ${
              seeVariant === "simple" ? "bg-white text-slate-900 shadow" : "text-slate-500"
            }`}
          >
            简化（仅历史 + 当前输入）
          </button>
        </div>
        <div className="mt-2 flex items-center gap-3 text-xs text-slate-500">
          <span className="rounded-full bg-slate-100 px-3 py-1">Session: {sessionId || "未初始化"}</span>
          <button
            type="button"
            onClick={initSession}
            className="rounded-full bg-white px-3 py-1 text-blue-600 shadow-sm ring-1 ring-slate-200 hover:bg-blue-50"
          >
            新建会话
          </button>
        </div>
      </header>

      {error && <p className="rounded-2xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}

      <div className="grid flex-1 gap-6 lg:grid-cols-2">
        <div className="flex flex-col">
          <ChatWindow messages={messages} />
          <MessageInput disabled={loading || !sessionId} onSend={handleSend} />
          <AnalyzeButton disabled={analyzing || !sessionId} onAnalyze={handleAnalyze} />
        </div>
        <div className="flex flex-col gap-4">
          <StoryDisplay story={story} defect={topDefect} />
        </div>
      </div>
    </div>
  )
}

export default App
