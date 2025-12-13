import { Component, type ErrorInfo, type ReactNode } from "react"

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  message?: string
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, message: undefined }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, message: error?.message || "Unknown error" }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Uncaught error in React tree:", error, info?.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="mx-auto max-w-2xl rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800">
          <p className="font-semibold">前端运行时出错，页面已暂停渲染。</p>
          <p className="mt-2 break-all">{this.state.message}</p>
          <p className="mt-4 text-xs text-rose-600">
            刷新页面重试，或查看浏览器 Console 获取更多信息。
          </p>
        </div>
      )
    }
    return this.props.children
  }
}
