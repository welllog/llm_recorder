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
	requestBody, err := io.ReadAll(r.Body)
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

// extractModelFromRequest 从请求body中提取模型名称，方便快速查看
func extractModelFromRequest(body string) string {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		return ""
	}
	return req.Model
}

// extractMessagesFromRequest 从请求body中提取对话消息，方便快速查看
func extractMessagesFromRequest(body string) string {
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
			Name    string `json:"name,omitempty"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(body), &req); err != nil {
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

		// 格式化处理内容中的换行，增加缩进
		lines := strings.Split(strings.TrimSpace(msg.Content), "\n")
		for _, line := range lines {
			if line == "" {
				sb.WriteString("│\n")
				continue
			}
			sb.WriteString(fmt.Sprintf("│   %s\n", line))
		}

		if i < len(req.Messages)-1 {
			sb.WriteString("│\n")
		}
	}
	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

// extractResponseContent 从响应body中提取模型返回的内容，方便快速查看
func extractResponseContent(body string) string {
	var content string
	// 处理流式响应的情况
	if strings.HasPrefix(body, "data: ") {
		var sb strings.Builder
		lines := strings.Split(body, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || line == "data: [DONE]" {
				continue
			}
			if strings.HasPrefix(line, "data: ") {
				jsonStr := strings.TrimPrefix(line, "data: ")
				var resp struct {
					Choices []struct {
						Delta struct {
							Content string `json:"content"`
						} `json:"delta"`
					} `json:"choices"`
				}
				if err := json.Unmarshal([]byte(jsonStr), &resp); err == nil && len(resp.Choices) > 0 {
					sb.WriteString(resp.Choices[0].Delta.Content)
				}
			}
		}
		content = sb.String()
	} else {
		// 处理普通响应
		var resp struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(body), &resp); err == nil && len(resp.Choices) > 0 {
			content = resp.Choices[0].Message.Content
		}
	}

	if content == "" {
		return ""
	}

	// 美化格式输出
	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 🤖 ASSISTANT RESPONSE\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	lines := strings.Split(strings.TrimSpace(content), "\n")
	for _, line := range lines {
		if line == "" {
			sb.WriteString("│\n")
			continue
		}
		sb.WriteString(fmt.Sprintf("│   %s\n", line))
	}
	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

func (s *proxyServer) logExchange(entry exchangeLog) {
	// 提取模型名称
	model := extractModelFromRequest(entry.RequestBody)
	// 提取对话消息
	messages := extractMessagesFromRequest(entry.RequestBody)
	// 提取响应内容
	responseContent := extractResponseContent(entry.ResponseBody)

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
		s.saveMessageToFile(entry.RequestID, model, messages, responseContent)
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

func (s *proxyServer) saveMessageToFile(reqID uint64, model, messages, response string) {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%04d.txt", timestamp, reqID)
	filePath := filepath.Join(s.msgDir, filename)

	var content strings.Builder
	content.WriteString("╔══════════════════════════════════════════════════════════════════════════════════════════════════\n")
	content.WriteString(fmt.Sprintf("║ 🆔 Request ID: %d\n", reqID))
	content.WriteString(fmt.Sprintf("║ ⏰ Timestamp:  %s\n", time.Now().Format(time.RFC3339)))
	content.WriteString(fmt.Sprintf("║ 🧠 Model:      %s\n", model))
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
