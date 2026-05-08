package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

type proxyConfig struct {
	listenAddr      string
	upstreamBaseURL string
	apiKey          string
	logFile         string
	logFormat       string
	msgDir          string
	timeout         time.Duration
}

type proxyServer struct {
	client          *http.Client
	upstreamChatURL string
	apiKey          string
	logger          *slog.Logger
	msgDir          string
	nextID          atomic.Uint64
}

type exchangeLog struct {
	RequestID       uint64            `json:"requestId"`
	Method          string            `json:"method"`
	Path            string            `json:"path"`
	RemoteAddr      string            `json:"remoteAddr"`
	UpstreamURL     string            `json:"upstreamUrl"`
	StatusCode      int               `json:"statusCode"`
	Duration        time.Duration     `json:"duration"`
	RequestBody     string            `json:"requestBody"`
	ResponseBody    string            `json:"responseBody"`
	Error           string            `json:"error,omitempty"`
	ContentType     string            `json:"contentType,omitempty"`
	UpstreamHeaders map[string]string `json:"upstreamHeaders,omitempty"`
}

// maxRequestBodySize 请求体最大允许大小（50MB），防止恶意或异常请求导致 OOM
const maxRequestBodySize = 50 << 20

func main() {
	cfg, err := parseFlags()
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	logFile, err := os.OpenFile(cfg.logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "open log file %q: %v\n", cfg.logFile, err)
		os.Exit(1)
	}
	defer func() {
		_ = logFile.Close()
	}()

	var logger *slog.Logger
	if cfg.logFormat == "json" {
		logger = slog.New(slog.NewJSONHandler(logFile, &slog.HandlerOptions{Level: slog.LevelInfo}))
	} else {
		logger = slog.New(slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelInfo}))
	}
	slog.SetDefault(logger)

	upstreamChatURL, err := buildUpstreamChatURL(cfg.upstreamBaseURL)
	if err != nil {
		logger.Error("invalid upstream base url", "error", err)
		os.Exit(1)
	}

	server := &proxyServer{
		client: &http.Client{
			Timeout: cfg.timeout,
			Transport: &http.Transport{
				Proxy:               http.ProxyFromEnvironment,
				DialContext:         (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
				ForceAttemptHTTP2:   true,
				MaxIdleConns:        100,
				IdleConnTimeout:     90 * time.Second,
				TLSHandshakeTimeout: 10 * time.Second,
				DisableCompression:  true,
			},
		},
		upstreamChatURL: upstreamChatURL,
		apiKey:          cfg.apiKey,
		logger:          logger,
		msgDir:          cfg.msgDir,
	}

	if cfg.msgDir != "" {
		if err := os.MkdirAll(cfg.msgDir, 0o755); err != nil {
			logger.Error("failed to create message directory", "dir", cfg.msgDir, "error", err)
			os.Exit(1)
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", server.handleChatCompletions)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})

	httpServer := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("shutdown failed", "error", err)
		}
	}()

	logger.Info("proxy server starting", "listenAddr", cfg.listenAddr, "upstreamChatURL", upstreamChatURL, "logFile", cfg.logFile)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("server stopped with error", "error", err)
		os.Exit(1)
	}
}

func parseFlags() (proxyConfig, error) {
	var cfg proxyConfig
	flag.StringVar(&cfg.listenAddr, "listen", ":8080", "listen address for the proxy server")
	flag.StringVar(&cfg.upstreamBaseURL, "upstream-base-url", "", "OpenAI-compatible upstream base URL, e.g. https://api.example.com/v1")
	flag.StringVar(&cfg.apiKey, "api-key", "", "upstream API key")
	flag.StringVar(&cfg.logFile, "log-file", "llm_proxy.log", "log file path in the current directory")
	flag.StringVar(&cfg.logFormat, "log-format", "text", "log format: text (human-readable) or json")
	flag.StringVar(&cfg.msgDir, "msg-dir", "messages", "directory to store structured message text files (empty to disable)")
	flag.DurationVar(&cfg.timeout, "timeout", 180*time.Second, "upstream request timeout")
	flag.Parse()

	cfg.listenAddr = strings.TrimSpace(cfg.listenAddr)
	cfg.upstreamBaseURL = strings.TrimSpace(cfg.upstreamBaseURL)
	cfg.apiKey = strings.TrimSpace(cfg.apiKey)
	cfg.logFile = strings.TrimSpace(cfg.logFile)
	cfg.logFormat = strings.ToLower(strings.TrimSpace(cfg.logFormat))
	cfg.msgDir = strings.TrimSpace(cfg.msgDir)

	if cfg.listenAddr == "" {
		cfg.listenAddr = ":8080"
	}
	if cfg.upstreamBaseURL == "" {
		return proxyConfig{}, fmt.Errorf("missing required flag: --upstream-base-url")
	}
	if cfg.apiKey == "" {
		return proxyConfig{}, fmt.Errorf("missing required flag: --api-key")
	}
	if cfg.logFile == "" {
		cfg.logFile = "llm_proxy.log"
	}
	if cfg.logFormat != "text" && cfg.logFormat != "json" {
		return proxyConfig{}, fmt.Errorf("--log-format must be either 'text' or 'json'")
	}
	if cfg.timeout <= 0 {
		return proxyConfig{}, fmt.Errorf("--timeout must be greater than zero")
	}

	return cfg, nil
}

func buildUpstreamChatURL(baseURL string) (string, error) {
	trimmed := strings.TrimSpace(baseURL)
	if trimmed == "" {
		return "", fmt.Errorf("empty upstream base url")
	}
	joined, err := url.JoinPath(trimmed, "chat/completions")
	if err != nil {
		return "", err
	}
	return joined, nil
}

func (s *proxyServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/chat/completions" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}

	requestID := s.nextID.Add(1)
	startedAt := time.Now()
	requestBody, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBodySize))
	if err != nil {
		s.logExchange(exchangeLog{
			RequestID:   requestID,
			Method:      r.Method,
			Path:        r.URL.Path,
			RemoteAddr:  r.RemoteAddr,
			UpstreamURL: s.upstreamChatURL,
			Duration:    time.Since(startedAt),
			Error:       fmt.Sprintf("read request body: %v", err),
		})
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	_ = r.Body.Close()

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, s.upstreamChatURL, bytes.NewReader(requestBody))
	if err != nil {
		s.logExchange(exchangeLog{
			RequestID:   requestID,
			Method:      r.Method,
			Path:        r.URL.Path,
			RemoteAddr:  r.RemoteAddr,
			UpstreamURL: s.upstreamChatURL,
			Duration:    time.Since(startedAt),
			RequestBody: string(requestBody),
			Error:       fmt.Sprintf("create upstream request: %v", err),
		})
		http.Error(w, "failed to create upstream request", http.StatusInternalServerError)
		return
	}
	// 确保使用上游 URL 的 Host，而非客户端原始 Host
	// （http.NewRequestWithContext 已自动设置，此行为防御性编码）
	upstreamReq.Host = ""
	copyRequestHeaders(upstreamReq.Header, r.Header)
	upstreamReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	if upstreamReq.Header.Get("Content-Type") == "" {
		upstreamReq.Header.Set("Content-Type", "application/json")
	}
	upstreamReq.ContentLength = int64(len(requestBody))

	upstreamResp, err := s.client.Do(upstreamReq)
	if err != nil {
		s.logExchange(exchangeLog{
			RequestID:   requestID,
			Method:      r.Method,
			Path:        r.URL.Path,
			RemoteAddr:  r.RemoteAddr,
			UpstreamURL: s.upstreamChatURL,
			Duration:    time.Since(startedAt),
			RequestBody: string(requestBody),
			Error:       fmt.Sprintf("call upstream: %v", err),
		})
		http.Error(w, "failed to call upstream", http.StatusBadGateway)
		return
	}
	defer func() {
		_ = upstreamResp.Body.Close()
	}()

	copyResponseHeaders(w.Header(), upstreamResp.Header)
	w.WriteHeader(upstreamResp.StatusCode)

	responseBuffer := &strings.Builder{}
	writer := &flushWriter{w: w}
	copyErr := copyResponseBody(writer, upstreamResp.Body, responseBuffer)
	if copyErr != nil && !errors.Is(copyErr, context.Canceled) {
		s.logExchange(exchangeLog{
			RequestID:       requestID,
			Method:          r.Method,
			Path:            r.URL.Path,
			RemoteAddr:      r.RemoteAddr,
			UpstreamURL:     s.upstreamChatURL,
			StatusCode:      upstreamResp.StatusCode,
			Duration:        time.Since(startedAt),
			RequestBody:     string(requestBody),
			ResponseBody:    responseBuffer.String(),
			Error:           fmt.Sprintf("copy upstream response: %v", copyErr),
			ContentType:     upstreamResp.Header.Get("Content-Type"),
			UpstreamHeaders: selectedHeaders(upstreamResp.Header),
		})
		return
	}

	s.logExchange(exchangeLog{
		RequestID:       requestID,
		Method:          r.Method,
		Path:            r.URL.Path,
		RemoteAddr:      r.RemoteAddr,
		UpstreamURL:     s.upstreamChatURL,
		StatusCode:      upstreamResp.StatusCode,
		Duration:        time.Since(startedAt),
		RequestBody:     string(requestBody),
		ResponseBody:    responseBuffer.String(),
		ContentType:     upstreamResp.Header.Get("Content-Type"),
		UpstreamHeaders: selectedHeaders(upstreamResp.Header),
	})
}

// chatRequest 用于一次性解析请求体中的 model 和 messages
type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

// parseChatRequest 一次性解析请求体，避免多次 JSON 反序列化
func parseChatRequest(body string) chatRequest {
	var req chatRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		return chatRequest{}
	}
	return req
}

// extractModelFromRequest 从请求body中提取模型名称，方便快速查看
func extractModelFromRequest(body string) string {
	return parseChatRequest(body).Model
}

// toolCallInfo 用于解析消息中的 tool_calls 字段
type toolCallInfo struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// chatMessage 用于解析请求中的消息
type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []toolCallInfo `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	// 兼容旧版 function_call
	FunctionCall *struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function_call,omitempty"`
}

// extractMessagesFromRequest 从已解析的 chatRequest 中提取对话消息，方便快速查看
func extractMessagesFromRequest(req chatRequest) string {
	if len(req.Messages) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 💬 PROMPT DETAILS\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	for i, msg := range req.Messages {
		roleLabel := strings.ToUpper(msg.Role)
		if msg.Name != "" {
			roleLabel = fmt.Sprintf("%s (%s)", roleLabel, msg.Name)
		}

		icon := "👤"
		switch strings.ToLower(msg.Role) {
		case "system":
			icon = "⚙️ "
		case "assistant":
			icon = "🤖"
		case "tool", "function":
			icon = "🛠️ "
		}

		sb.WriteString(fmt.Sprintf("│ %s %-15s\n", icon, roleLabel))

		// 如果是 tool 消息，显示对应的 tool_call_id
		if msg.ToolCallID != "" {
			sb.WriteString(fmt.Sprintf("│   📎 tool_call_id: %s\n", msg.ToolCallID))
			sb.WriteString("│\n")
		}

		// 如果是 assistant 消息且包含 tool_calls，显示工具调用信息
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				sb.WriteString(fmt.Sprintf("│   🔧 Tool Call: %s (id: %s)\n", tc.Function.Name, tc.ID))
				// 尝试格式化 arguments
				var argsPretty string
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err == nil {
					if pretty, err := json.MarshalIndent(argsMap, "│     ", "  "); err == nil {
						argsPretty = string(pretty)
					}
				}
				if argsPretty == "" {
					argsPretty = tc.Function.Arguments
				}
				for _, line := range strings.Split(argsPretty, "\n") {
					sb.WriteString(fmt.Sprintf("│     %s\n", line))
				}
			}
			// 如果同时有 content，也显示
			if strings.TrimSpace(msg.Content) != "" {
				sb.WriteString("│\n")
				sb.WriteString("│   📝 Content:\n")
				for _, line := range strings.Split(strings.TrimSpace(msg.Content), "\n") {
					if line == "" {
						sb.WriteString("│\n")
						continue
					}
					sb.WriteString(fmt.Sprintf("│     %s\n", line))
				}
			}
		} else if msg.FunctionCall != nil {
			// 兼容旧版 function_call
			sb.WriteString(fmt.Sprintf("│   🔧 Function Call: %s\n", msg.FunctionCall.Name))
			var argsPretty string
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(msg.FunctionCall.Arguments), &argsMap); err == nil {
				if pretty, err := json.MarshalIndent(argsMap, "│     ", "  "); err == nil {
					argsPretty = string(pretty)
				}
			}
			if argsPretty == "" {
				argsPretty = msg.FunctionCall.Arguments
			}
			for _, line := range strings.Split(argsPretty, "\n") {
				sb.WriteString(fmt.Sprintf("│     %s\n", line))
			}
		} else {
			// 普通消息，格式化处理内容中的换行，增加缩进
			lines := strings.Split(strings.TrimSpace(msg.Content), "\n")
			for _, line := range lines {
				if line == "" {
					sb.WriteString("│\n")
					continue
				}
				sb.WriteString(fmt.Sprintf("│   %s\n", line))
			}
		}

		if i < len(req.Messages)-1 {
			sb.WriteString("│\n")
		}
	}
	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

// responseInfo 保存从响应中提取的结构化信息
type responseInfo struct {
	Content      string // 模型返回的文本内容
	ToolCalls    []toolCallInfo
	FinishReason string
	Usage        *usageInfo
	RawContent   string // 原始响应文本（用于保存到文件）
}

type usageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// extractResponseInfo 从响应body中提取模型返回的完整信息，包括工具调用、token用量等
func extractResponseInfo(body string) responseInfo {
	info := responseInfo{RawContent: body}

	// 处理流式响应的情况
	if strings.HasPrefix(body, "data: ") {
		info.extractFromStream(body)
	} else {
		info.extractFromNonStream(body)
	}

	return info
}

// streamToolCallDelta 用于流式响应中解析 delta.tool_calls，包含 index 字段
type streamToolCallDelta struct {
	Index   int    `json:"index"`
	ID      string `json:"id"`
	Type    string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// extractFromStream 从 SSE 流式响应中提取信息
func (info *responseInfo) extractFromStream(body string) {
	var contentBuilder strings.Builder
	var toolCalls []toolCallInfo
	var finishReason string
	var lastUsage *usageInfo

	lines := strings.Split(body, "\n")
	// 用于合并流式 tool_calls 的临时结构
	streamToolCalls := make(map[int]*toolCallInfo)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == "data: [DONE]" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonStr := strings.TrimPrefix(line, "data: ")

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string               `json:"content"`
					ToolCalls []streamToolCallDelta `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *usageInfo `json:"usage"`
		}
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.Delta.Content != "" {
				contentBuilder.WriteString(choice.Delta.Content)
			}
			// 合并流式 tool_calls（每个 chunk 只包含部分信息）
			for _, tc := range choice.Delta.ToolCalls {
				idx := tc.Index
				if existing, exists := streamToolCalls[idx]; exists {
					// 合并：更新 ID、name、追加 arguments
					if tc.ID != "" {
						existing.ID = tc.ID
					}
					if tc.Type != "" {
						existing.Type = tc.Type
					}
					if tc.Function.Name != "" {
						existing.Function.Name += tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						existing.Function.Arguments += tc.Function.Arguments
					}
				} else {
					streamToolCalls[idx] = &toolCallInfo{
						ID:   tc.ID,
						Type: tc.Type,
						Function: struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						}{
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
					}
				}
			}
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}
		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}
	}

	info.Content = contentBuilder.String()
	// 将 map 转为 slice（按 key 排序）
	if len(streamToolCalls) > 0 {
		indices := make([]int, 0, len(streamToolCalls))
		for k := range streamToolCalls {
			indices = append(indices, k)
		}
		sort.Ints(indices)
		for _, k := range indices {
			toolCalls = append(toolCalls, *streamToolCalls[k])
		}
	}
	info.ToolCalls = toolCalls
	info.FinishReason = finishReason
	info.Usage = lastUsage
}

// extractFromNonStream 从普通（非流式）响应中提取信息
func (info *responseInfo) extractFromNonStream(body string) {
	var resp struct {
		Choices []struct {
			Message struct {
				Content   string         `json:"content"`
				ToolCalls []toolCallInfo `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usageInfo `json:"usage"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return
	}
	if len(resp.Choices) > 0 {
		info.Content = resp.Choices[0].Message.Content
		info.ToolCalls = resp.Choices[0].Message.ToolCalls
		info.FinishReason = resp.Choices[0].FinishReason
	}
	info.Usage = resp.Usage
}

// formatResponseContent 格式化响应内容为可读文本
func formatResponseContent(info responseInfo) string {
	if info.Content == "" && len(info.ToolCalls) == 0 && info.Usage == nil {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 🤖 ASSISTANT RESPONSE\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	// 文本内容
	if info.Content != "" {
		lines := strings.Split(strings.TrimSpace(info.Content), "\n")
		for _, line := range lines {
			if line == "" {
				sb.WriteString("│\n")
				continue
			}
			sb.WriteString(fmt.Sprintf("│   %s\n", line))
		}
	}

	// 工具调用
	if len(info.ToolCalls) > 0 {
		if info.Content != "" {
			sb.WriteString("│\n")
		}
		sb.WriteString("│   🔧 TOOL CALLS:\n")
		sb.WriteString("│\n")
		for i, tc := range info.ToolCalls {
			sb.WriteString(fmt.Sprintf("│   ┌─ Tool Call #%d ─────────────────────────────────────────────────────────────────────────────\n", i+1))
			sb.WriteString(fmt.Sprintf("│   │ ID:       %s\n", tc.ID))
			sb.WriteString(fmt.Sprintf("│   │ Type:     %s\n", tc.Type))
			sb.WriteString(fmt.Sprintf("│   │ Function: %s\n", tc.Function.Name))
			// 格式化 arguments
			var argsPretty string
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err == nil {
				if pretty, err := json.MarshalIndent(argsMap, "│   │ Args:     ", "  "); err == nil {
					argsPretty = string(pretty)
				}
			}
			if argsPretty == "" {
				argsPretty = tc.Function.Arguments
			}
			if argsPretty != "" {
				sb.WriteString("│   │ Args:\n")
				for _, line := range strings.Split(argsPretty, "\n") {
					sb.WriteString(fmt.Sprintf("│   │   %s\n", line))
				}
			}
			sb.WriteString("│   └────────────────────────────────────────────────────────────────────────────────────────────\n")
		}
	}

	// Finish Reason
	if info.FinishReason != "" {
		sb.WriteString("│\n")
		sb.WriteString(fmt.Sprintf("│   🏁 Finish Reason: %s\n", info.FinishReason))
	}

	// Token 用量
	if info.Usage != nil {
		sb.WriteString("│\n")
		sb.WriteString("│   📊 Token Usage:\n")
		sb.WriteString(fmt.Sprintf("│      Prompt Tokens:     %d\n", info.Usage.PromptTokens))
		sb.WriteString(fmt.Sprintf("│      Completion Tokens: %d\n", info.Usage.CompletionTokens))
		sb.WriteString(fmt.Sprintf("│      Total Tokens:      %d\n", info.Usage.TotalTokens))
	}

	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

func (s *proxyServer) logExchange(entry exchangeLog) {
	// 一次性解析请求体，避免多次 JSON 反序列化
	chatReq := parseChatRequest(entry.RequestBody)
	model := chatReq.Model
	// 提取对话消息
	messages := extractMessagesFromRequest(chatReq)
	// 提取响应信息（包括工具调用、token用量等）
	responseInfo := extractResponseInfo(entry.ResponseBody)
	// 格式化响应内容
	responseContent := formatResponseContent(responseInfo)

	// 基本信息放在最前面
	attrs := []any{
		slog.Uint64("requestId", entry.RequestID),
		slog.Int("statusCode", entry.StatusCode),
		slog.String("model", model),
		slog.Duration("duration", entry.Duration),
		slog.String("remoteAddr", entry.RemoteAddr),
	}

	// 核心内容放在中间，最容易看到
	if messages != "" {
		attrs = append(attrs, slog.String("messages", messages))
	}

	if responseContent != "" {
		attrs = append(attrs, slog.String("responseContent", responseContent))
	}

	// 将消息保存到单独的文件中
	if s.msgDir != "" {
		s.saveMessageToFile(entry.RequestID, model, messages, responseContent, responseInfo)
	}

	// 详细信息放在最后
	attrs = append(attrs,
		slog.String("method", entry.Method),
		slog.String("path", entry.Path),
		slog.String("upstreamUrl", entry.UpstreamURL),
	)

	if entry.ContentType != "" {
		attrs = append(attrs, slog.String("contentType", entry.ContentType))
	}
	if len(entry.UpstreamHeaders) > 0 {
		attrs = append(attrs, slog.Any("upstreamHeaders", entry.UpstreamHeaders))
	}
	if entry.Error != "" {
		attrs = append(attrs, slog.String("error", entry.Error))
		s.logger.Error("❌ 请求失败", attrs...)
		return
	}
	s.logger.Info("✅ 请求成功", attrs...)
}

func (s *proxyServer) saveMessageToFile(reqID uint64, model, messages, response string, respInfo responseInfo) {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%04d.txt", timestamp, reqID)
	filePath := filepath.Join(s.msgDir, filename)

	var content strings.Builder
	content.WriteString("╔══════════════════════════════════════════════════════════════════════════════════════════════════\n")
	content.WriteString(fmt.Sprintf("║ 🆔 Request ID: %d\n", reqID))
	content.WriteString(fmt.Sprintf("║ ⏰ Timestamp:  %s\n", time.Now().Format(time.RFC3339)))
	content.WriteString(fmt.Sprintf("║ 🧠 Model:      %s\n", model))
	if respInfo.FinishReason != "" {
		content.WriteString(fmt.Sprintf("║ 🏁 Finish:     %s\n", respInfo.FinishReason))
	}
	if respInfo.Usage != nil {
		content.WriteString(fmt.Sprintf("║ 📊 Tokens:     prompt=%d, completion=%d, total=%d\n",
			respInfo.Usage.PromptTokens, respInfo.Usage.CompletionTokens, respInfo.Usage.TotalTokens))
	}
	if len(respInfo.ToolCalls) > 0 {
		content.WriteString(fmt.Sprintf("║ 🔧 Tool Calls: %d\n", len(respInfo.ToolCalls)))
		for i, tc := range respInfo.ToolCalls {
			content.WriteString(fmt.Sprintf("║    #%d %s (id: %s)\n", i+1, tc.Function.Name, tc.ID))
		}
	}
	content.WriteString("╚══════════════════════════════════════════════════════════════════════════════════════════════════\n\n")

	if messages != "" {
		content.WriteString(messages)
		content.WriteString("\n\n")
	}

	if response != "" {
		content.WriteString(response)
		content.WriteString("\n")
	}

	if err := os.WriteFile(filePath, []byte(content.String()), 0o644); err != nil {
		s.logger.Error("failed to save message to file", "path", filePath, "error", err)
	}
}

type flushWriter struct {
	w http.ResponseWriter
}

func (f *flushWriter) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	if flusher, ok := f.w.(http.Flusher); ok {
		flusher.Flush()
	}
	return n, err
}

func copyRequestHeaders(dst, src http.Header) {
	for key, values := range src {
		if isHopByHopHeader(key) || strings.EqualFold(key, "Authorization") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for key, values := range src {
		if isHopByHopHeader(key) {
			continue
		}
		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func isHopByHopHeader(key string) bool {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "connection", "proxy-connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func copyResponseBody(dst io.Writer, src io.Reader, buf *strings.Builder) error {
	data := make([]byte, 32*1024)
	for {
		n, err := src.Read(data)
		if n > 0 {
			chunk := data[:n]
			if buf != nil {
				_, _ = buf.Write(chunk)
			}
			written, writeErr := dst.Write(chunk)
			if writeErr != nil {
				return writeErr
			}
			if written != len(chunk) {
				return io.ErrShortWrite
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

func selectedHeaders(h http.Header) map[string]string {
	selected := make(map[string]string)
	for _, key := range []string{"Content-Type", "Transfer-Encoding", "Content-Length", "OpenAI-Model", "X-Request-Id"} {
		if value := h.Get(key); value != "" {
			selected[key] = value
		}
	}
	if len(selected) == 0 {
		return nil
	}
	return selected
}
