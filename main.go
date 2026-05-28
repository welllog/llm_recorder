package main

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode"
)

type proxyConfig struct {
	listenAddr                       string
	openaiUpstreamBaseURL            string
	openaiAPIKey                     string
	openaiResponsesUpstreamBaseURL   string
	openaiResponsesAPIKey            string
	anthropicUpstreamBaseURL         string
	anthropicAPIKey                  string
	logFile                          string
	logFormat                        string
	msgDir                           string
	timeout                          time.Duration
	maxResponseSize                  int
	disableFullLogging               bool
	enableCORS                       bool
	maxLogFieldSize                  int
	enableRequestDecompression       bool
	createDateSubdirs                bool
	healthzPath                      string
}

type upstreamTarget struct {
	provider    string
	upstreamURL string
	apiKey      string
}

type proxyServer struct {
	client                     *http.Client
	routeTargets               map[string]upstreamTarget
	logger                     *slog.Logger
	msgDir                     string
	nextID                     atomic.Uint64
	maxResponseSize            int
	enableCORS                 bool
	enableRequestDecompression bool
	createDateSubdirs          bool
	maxLogFieldSize            int
	disableFullLogging         bool
	wg                         sync.WaitGroup
}

type exchangeLog struct {
	RequestID       uint64            `json:"requestId"`
	Provider        string            `json:"provider"`
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

const (
	providerOpenAI          = "openai"
	providerAnthropic       = "anthropic"
	providerOpenAIResponses = "openai-responses"
)

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

	routeTargets, err := buildRouteTargets(cfg)
	if err != nil {
		logger.Error("invalid upstream configuration", "error", err)
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
				DisableCompression:  false, // 启用自动解压，方便日志记录
			},
		},
		routeTargets:               routeTargets,
		logger:                     logger,
		msgDir:                     cfg.msgDir,
		maxResponseSize:            cfg.maxResponseSize,
		enableCORS:                 cfg.enableCORS,
		enableRequestDecompression: cfg.enableRequestDecompression,
		createDateSubdirs:          cfg.createDateSubdirs,
		maxLogFieldSize:            cfg.maxLogFieldSize,
		disableFullLogging:         cfg.disableFullLogging,
	}

	if cfg.msgDir != "" {
		if err := os.MkdirAll(cfg.msgDir, 0o755); err != nil {
			logger.Error("failed to create message directory", "dir", cfg.msgDir, "error", err)
			os.Exit(1)
		}
	}

	mux := newProxyMux(server, cfg)

	httpServer := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		<-ctx.Done()
		logger.Info("shutdown signal received, waiting for active requests to complete")
		// 先关闭server，不再接受新请求
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("shutdown failed", "error", err)
		}
		// 等待所有正在处理的请求完成
		waitCtx, waitCancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer waitCancel()
		done := make(chan struct{})
		go func() {
			server.wg.Wait()
			close(done)
		}()
		select {
		case <-done:
			logger.Info("all requests completed successfully, exiting")
		case <-waitCtx.Done():
			logger.Warn("timed out waiting for requests to complete, force exiting")
		}
	}()

	logger.Info(
		"proxy server starting",
		"listenAddr", cfg.listenAddr,
		"openaiEnabled", routeTargets["/v1/chat/completions"].upstreamURL != "",
		"openaiResponsesEnabled", routeTargets["/v1/responses"].upstreamURL != "",
		"anthropicEnabled", routeTargets["/v1/messages"].upstreamURL != "",
		"logFile", cfg.logFile,
	)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("server stopped with error", "error", err)
		os.Exit(1)
	}
}

func parseFlags() (proxyConfig, error) {
	var cfg proxyConfig
	flag.StringVar(&cfg.listenAddr, "listen", ":8080", "listen address for the proxy server")
	flag.StringVar(&cfg.openaiUpstreamBaseURL, "openai-upstream-base-url", "", "OpenAI upstream base URL, e.g. https://api.openai.com/v1")
	flag.StringVar(&cfg.openaiAPIKey, "openai-api-key", "", "OpenAI upstream API key")
	flag.StringVar(&cfg.openaiResponsesUpstreamBaseURL, "openai-responses-upstream-base-url", "", "OpenAI Responses API upstream base URL (falls back to openai-upstream-base-url if empty)")
	flag.StringVar(&cfg.openaiResponsesAPIKey, "openai-responses-api-key", "", "OpenAI Responses API key (falls back to openai-api-key if empty)")
	flag.StringVar(&cfg.anthropicUpstreamBaseURL, "anthropic-upstream-base-url", "", "Anthropic upstream base URL, e.g. https://api.anthropic.com")
	flag.StringVar(&cfg.anthropicAPIKey, "anthropic-api-key", "", "Anthropic upstream API key")
	flag.StringVar(&cfg.logFile, "log-file", "llm_proxy.log", "log file path in the current directory")
	flag.StringVar(&cfg.logFormat, "log-format", "text", "log format: text (human-readable) or json")
	flag.StringVar(&cfg.msgDir, "msg-dir", "messages", "directory to store structured message text files (empty to disable)")
	flag.DurationVar(&cfg.timeout, "timeout", 180*time.Second, "upstream request timeout")
	flag.IntVar(&cfg.maxResponseSize, "max-response-size", 50*1024*1024, "Maximum response body size to log in bytes")
	flag.BoolVar(&cfg.disableFullLogging, "disable-full-logging", false, "Disable logging full request/response bodies for sensitive data")
	flag.BoolVar(&cfg.enableCORS, "enable-cors", false, "Enable CORS headers for browser access")
	flag.IntVar(&cfg.maxLogFieldSize, "max-log-field-size", 100*1024, "Maximum length of log fields in bytes, longer fields will be truncated")
	flag.BoolVar(&cfg.enableRequestDecompression, "enable-request-decompression", true, "Enable decompression of gzip/deflate compressed request bodies")
	flag.BoolVar(&cfg.createDateSubdirs, "create-date-subdirs", true, "Create date-based subdirectories for message files")
	flag.StringVar(&cfg.healthzPath, "healthz-path", "/healthz", "Path for health check endpoint")
	flag.Parse()

	// Trim whitespace from all string fields
	cfg.listenAddr = strings.TrimSpace(cfg.listenAddr)
	cfg.openaiUpstreamBaseURL = strings.TrimSpace(cfg.openaiUpstreamBaseURL)
	cfg.openaiAPIKey = strings.TrimSpace(cfg.openaiAPIKey)
	cfg.openaiResponsesUpstreamBaseURL = strings.TrimSpace(cfg.openaiResponsesUpstreamBaseURL)
	cfg.openaiResponsesAPIKey = strings.TrimSpace(cfg.openaiResponsesAPIKey)
	cfg.anthropicUpstreamBaseURL = strings.TrimSpace(cfg.anthropicUpstreamBaseURL)
	cfg.anthropicAPIKey = strings.TrimSpace(cfg.anthropicAPIKey)
	cfg.logFile = strings.TrimSpace(cfg.logFile)
	cfg.logFormat = strings.ToLower(strings.TrimSpace(cfg.logFormat))
	cfg.msgDir = strings.TrimSpace(cfg.msgDir)
	cfg.healthzPath = strings.TrimSpace(cfg.healthzPath)

	// Read from environment variables if flags not set
	if cfg.openaiUpstreamBaseURL == "" {
		cfg.openaiUpstreamBaseURL = strings.TrimSpace(os.Getenv("OPENAI_UPSTREAM_BASE_URL"))
	}
	if cfg.openaiAPIKey == "" {
		cfg.openaiAPIKey = strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	if cfg.openaiResponsesUpstreamBaseURL == "" {
		cfg.openaiResponsesUpstreamBaseURL = strings.TrimSpace(os.Getenv("OPENAI_RESPONSES_UPSTREAM_BASE_URL"))
	}
	if cfg.openaiResponsesAPIKey == "" {
		cfg.openaiResponsesAPIKey = strings.TrimSpace(os.Getenv("OPENAI_RESPONSES_API_KEY"))
	}
	if cfg.anthropicUpstreamBaseURL == "" {
		cfg.anthropicUpstreamBaseURL = strings.TrimSpace(os.Getenv("ANTHROPIC_UPSTREAM_BASE_URL"))
	}
	if cfg.anthropicAPIKey == "" {
		cfg.anthropicAPIKey = strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	}
	if cfg.logFile == "" {
		cfg.logFile = strings.TrimSpace(os.Getenv("LLM_PROXY_LOG_FILE"))
	}
	if cfg.logFormat == "" {
		cfg.logFormat = strings.ToLower(strings.TrimSpace(os.Getenv("LLM_PROXY_LOG_FORMAT")))
	}
	if cfg.msgDir == "" {
		cfg.msgDir = strings.TrimSpace(os.Getenv("LLM_PROXY_MSG_DIR"))
	}
	if timeoutStr := os.Getenv("LLM_PROXY_TIMEOUT"); timeoutStr != "" {
		if t, err := time.ParseDuration(timeoutStr); err == nil {
			cfg.timeout = t
		}
	}
	if maxRespSizeStr := os.Getenv("LLM_PROXY_MAX_RESPONSE_SIZE"); maxRespSizeStr != "" {
		if s, err := strconv.Atoi(maxRespSizeStr); err == nil {
			cfg.maxResponseSize = s
		}
	}
	if disableFullLogStr := os.Getenv("LLM_PROXY_DISABLE_FULL_LOGGING"); disableFullLogStr != "" {
		if b, err := strconv.ParseBool(disableFullLogStr); err == nil {
			cfg.disableFullLogging = b
		}
	}
	if enableCorsStr := os.Getenv("LLM_PROXY_ENABLE_CORS"); enableCorsStr != "" {
		if b, err := strconv.ParseBool(enableCorsStr); err == nil {
			cfg.enableCORS = b
		}
	}
	if maxLogFieldSizeStr := os.Getenv("LLM_PROXY_MAX_LOG_FIELD_SIZE"); maxLogFieldSizeStr != "" {
		if s, err := strconv.Atoi(maxLogFieldSizeStr); err == nil {
			cfg.maxLogFieldSize = s
		}
	}
	if enableReqDecompStr := os.Getenv("LLM_PROXY_ENABLE_REQUEST_DECOMPRESSION"); enableReqDecompStr != "" {
		if b, err := strconv.ParseBool(enableReqDecompStr); err == nil {
			cfg.enableRequestDecompression = b
		}
	}
	if createDateSubdirsStr := os.Getenv("LLM_PROXY_CREATE_DATE_SUBDIRS"); createDateSubdirsStr != "" {
		if b, err := strconv.ParseBool(createDateSubdirsStr); err == nil {
			cfg.createDateSubdirs = b
		}
	}
	if healthzPathStr := os.Getenv("LLM_PROXY_HEALTHZ_PATH"); healthzPathStr != "" {
		cfg.healthzPath = strings.TrimSpace(healthzPathStr)
	}

	// Default values
	if cfg.listenAddr == "" {
		cfg.listenAddr = ":8080"
	}
	if cfg.logFile == "" {
		cfg.logFile = "llm_proxy.log"
	}
	if cfg.logFormat == "" {
		cfg.logFormat = "text"
	}
	if cfg.maxResponseSize <= 0 {
		cfg.maxResponseSize = 50 * 1024 * 1024
	}
	if cfg.maxLogFieldSize <= 0 {
		cfg.maxLogFieldSize = 100 * 1024
	}
	if cfg.healthzPath == "" {
		cfg.healthzPath = "/healthz"
	}

	// Validate configuration
	hasOpenAI := cfg.openaiUpstreamBaseURL != "" && cfg.openaiAPIKey != ""
	hasAnthropic := cfg.anthropicUpstreamBaseURL != "" && cfg.anthropicAPIKey != ""
	if !hasOpenAI && !hasAnthropic {
		return proxyConfig{}, fmt.Errorf("at least one upstream (OpenAI or Anthropic) must be fully configured")
	}
	if cfg.logFormat != "text" && cfg.logFormat != "json" {
		return proxyConfig{}, fmt.Errorf("--log-format must be either 'text' or 'json'")
	}
	if cfg.timeout <= 0 {
		return proxyConfig{}, fmt.Errorf("--timeout must be greater than zero")
	}

	return cfg, nil
}

func buildUpstreamURL(baseURL, provider string) (string, error) {
	trimmed := strings.TrimSpace(baseURL)
	if trimmed == "" {
		return "", fmt.Errorf("empty upstream base url")
	}
	// 目标路径
	targetPath := "chat/completions"
	if provider == providerAnthropic {
		targetPath = "v1/messages"
	} else if provider == providerOpenAIResponses {
		targetPath = "responses"
	}
	// 检查是否已经包含目标路径，避免重复添加
	parsed, err := url.Parse(trimmed)
	if err != nil {
		return "", err
	}
	if strings.HasSuffix(strings.TrimSuffix(parsed.Path, "/"), targetPath) {
		return trimmed, nil
	}
	// 拼接路径
	joined, err := url.JoinPath(trimmed, targetPath)
	if err != nil {
		return "", err
	}
	return joined, nil
}

func buildRouteTargets(cfg proxyConfig) (map[string]upstreamTarget, error) {
	targets := make(map[string]upstreamTarget, 3)

	if cfg.openaiUpstreamBaseURL != "" && cfg.openaiAPIKey != "" {
		upstreamURL, err := buildUpstreamURL(cfg.openaiUpstreamBaseURL, providerOpenAI)
		if err != nil {
			return nil, fmt.Errorf("openai upstream: %w", err)
		}
		targets["/v1/chat/completions"] = upstreamTarget{
			provider:    providerOpenAI,
			upstreamURL: upstreamURL,
			apiKey:      cfg.openaiAPIKey,
		}
	}

	// Responses API: 使用独立配置，未配置时 fallback 到 OpenAI 配置
	responsesBaseURL := cfg.openaiResponsesUpstreamBaseURL
	if responsesBaseURL == "" {
		responsesBaseURL = cfg.openaiUpstreamBaseURL
	}
	responsesAPIKey := cfg.openaiResponsesAPIKey
	if responsesAPIKey == "" {
		responsesAPIKey = cfg.openaiAPIKey
	}
	if responsesBaseURL != "" && responsesAPIKey != "" {
		responsesUpstreamURL, err := buildUpstreamURL(responsesBaseURL, providerOpenAIResponses)
		if err != nil {
			return nil, fmt.Errorf("openai-responses upstream: %w", err)
		}
		targets["/v1/responses"] = upstreamTarget{
			provider:    providerOpenAIResponses,
			upstreamURL: responsesUpstreamURL,
			apiKey:      responsesAPIKey,
		}
	}

	if cfg.anthropicUpstreamBaseURL != "" && cfg.anthropicAPIKey != "" {
		upstreamURL, err := buildUpstreamURL(cfg.anthropicUpstreamBaseURL, providerAnthropic)
		if err != nil {
			return nil, fmt.Errorf("anthropic upstream: %w", err)
		}
		targets["/v1/messages"] = upstreamTarget{
			provider:    providerAnthropic,
			upstreamURL: upstreamURL,
			apiKey:      cfg.anthropicAPIKey,
		}
	}

	return targets, nil
}

func newProxyMux(server *proxyServer, cfg proxyConfig) http.Handler {
	mux := http.NewServeMux()

	// 健康检查端点
	mux.HandleFunc(cfg.healthzPath, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	// 动态添加已配置的上游路由
	if _, ok := server.routeTargets["/v1/chat/completions"]; ok {
		mux.HandleFunc("/v1/chat/completions", server.handleChatCompletions)
	}
	if _, ok := server.routeTargets["/v1/messages"]; ok {
		mux.HandleFunc("/v1/messages", server.handleMessages)
	}
	if _, ok := server.routeTargets["/v1/responses"]; ok {
		mux.HandleFunc("/v1/responses", server.handleResponses)
	}

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})

	if !cfg.enableCORS {
		return mux
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		applyCORSHeaders(w.Header())
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		mux.ServeHTTP(w, r)
	})
}

func applyCORSHeaders(headers http.Header) {
	headers.Set("Access-Control-Allow-Origin", "*")
	headers.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	headers.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Api-Key, anthropic-version, anthropic-beta")
	headers.Set("Access-Control-Expose-Headers", "Content-Type")
	headers.Set("Access-Control-Max-Age", "600")
}

func (s *proxyServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleProxyRequest(w, r, "/v1/chat/completions")
}

func (s *proxyServer) handleMessages(w http.ResponseWriter, r *http.Request) {
	s.handleProxyRequest(w, r, "/v1/messages")
}

func (s *proxyServer) handleResponses(w http.ResponseWriter, r *http.Request) {
	s.handleProxyRequest(w, r, "/v1/responses")
}

func (s *proxyServer) handleProxyRequest(w http.ResponseWriter, r *http.Request, expectedPath string) {
	// 等待组跟踪活跃请求
	s.wg.Add(1)
	defer s.wg.Done()

	if r.URL.Path != expectedPath {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	target, ok := s.routeTargets[expectedPath]
	if !ok {
		http.Error(w, "route target not configured", http.StatusInternalServerError)
		return
	}

	requestID := s.nextID.Add(1)
	startedAt := time.Now()

	// 读取并解压请求体（如果启用）
	var bodyReader io.ReadCloser
	var err error
	if s.enableRequestDecompression {
		bodyReader, err = decompressRequestBody(r, maxRequestBodySize)
		if err != nil {
			s.logExchange(exchangeLog{
				RequestID:   requestID,
				Provider:    target.provider,
				Method:      r.Method,
				Path:        r.URL.Path,
				RemoteAddr:  r.RemoteAddr,
				UpstreamURL: target.upstreamURL,
				Duration:    time.Since(startedAt),
				Error:       fmt.Sprintf("decompress request body: %v", err),
			}, target.provider)
			http.Error(w, "failed to decompress request body", http.StatusBadRequest)
			return
		}
	} else {
		bodyReader = newLimitedReadCloser(r.Body, maxRequestBodySize, r.Body)
	}
	defer func() {
		_ = bodyReader.Close()
	}()

	requestBody, err := io.ReadAll(bodyReader)
	if err != nil {
		s.logExchange(exchangeLog{
			RequestID:   requestID,
			Provider:    target.provider,
			Method:      r.Method,
			Path:        r.URL.Path,
			RemoteAddr:  r.RemoteAddr,
			UpstreamURL: target.upstreamURL,
			Duration:    time.Since(startedAt),
			Error:       fmt.Sprintf("read request body: %v", err),
		}, target.provider)
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target.upstreamURL, bytes.NewReader(requestBody))
	if err != nil {
		s.logExchange(exchangeLog{
			RequestID:   requestID,
			Provider:    target.provider,
			Method:      r.Method,
			Path:        r.URL.Path,
			RemoteAddr:  r.RemoteAddr,
			UpstreamURL: target.upstreamURL,
			Duration:    time.Since(startedAt),
			RequestBody: string(requestBody),
			Error:       fmt.Sprintf("create upstream request: %v", err),
		}, target.provider)
		http.Error(w, "failed to create upstream request", http.StatusInternalServerError)
		return
	}
	// 确保使用上游 URL 的 Host，而非客户端原始 Host
	// （http.NewRequestWithContext 已自动设置，此行为防御性编码）
	upstreamReq.Host = ""
	copyRequestHeaders(upstreamReq.Header, r.Header)
	if target.provider == providerAnthropic {
		upstreamReq.Header.Set("X-Api-Key", target.apiKey)
		upstreamReq.Header.Del("Authorization")
	} else {
		upstreamReq.Header.Del("X-Api-Key")
		upstreamReq.Header.Set("Authorization", "Bearer "+target.apiKey)
	}
	if upstreamReq.Header.Get("Content-Type") == "" {
		upstreamReq.Header.Set("Content-Type", "application/json")
	}
	upstreamReq.ContentLength = int64(len(requestBody))

	upstreamResp, err := s.client.Do(upstreamReq)
	if err != nil {
		s.logExchange(exchangeLog{
			RequestID:   requestID,
			Provider:    target.provider,
			Method:      r.Method,
			Path:        r.URL.Path,
			RemoteAddr:  r.RemoteAddr,
			UpstreamURL: target.upstreamURL,
			Duration:    time.Since(startedAt),
			RequestBody: string(requestBody),
			Error:       fmt.Sprintf("call upstream: %v", err),
		}, target.provider)
		http.Error(w, "failed to call upstream", http.StatusBadGateway)
		return
	}
	defer func() {
		_ = upstreamResp.Body.Close()
	}()

	copyResponseHeaders(w.Header(), upstreamResp.Header)

	// 如果上游返回压缩响应，解压后再转发，以便日志能正确解析
	respBody, decompressed := decompressResponseBody(upstreamResp)
	if decompressed {
		w.Header().Del("Content-Encoding")
		w.Header().Del("Content-Length")
	}
	w.WriteHeader(upstreamResp.StatusCode)

	responseBuffer := &strings.Builder{}
	writer := &flushWriter{w: w}
	copyErr := copyResponseBody(writer, respBody, responseBuffer, s.maxResponseSize)
	if copyErr != nil && !errors.Is(copyErr, context.Canceled) {
		s.logExchange(exchangeLog{
			RequestID:       requestID,
			Provider:        target.provider,
			Method:          r.Method,
			Path:            r.URL.Path,
			RemoteAddr:      r.RemoteAddr,
			UpstreamURL:     target.upstreamURL,
			StatusCode:      upstreamResp.StatusCode,
			Duration:        time.Since(startedAt),
			RequestBody:     string(requestBody),
			ResponseBody:    responseBuffer.String(),
			Error:           fmt.Sprintf("copy upstream response: %v", copyErr),
			ContentType:     upstreamResp.Header.Get("Content-Type"),
			UpstreamHeaders: selectedHeaders(upstreamResp.Header),
		}, target.provider)
		return
	}

	s.logExchange(exchangeLog{
		RequestID:       requestID,
		Provider:        target.provider,
		Method:          r.Method,
		Path:            r.URL.Path,
		RemoteAddr:      r.RemoteAddr,
		UpstreamURL:     target.upstreamURL,
		StatusCode:      upstreamResp.StatusCode,
		Duration:        time.Since(startedAt),
		RequestBody:     string(requestBody),
		ResponseBody:    responseBuffer.String(),
		ContentType:     upstreamResp.Header.Get("Content-Type"),
		UpstreamHeaders: selectedHeaders(upstreamResp.Header),
	}, target.provider)
}

// chatRequest 用于一次性解析请求体中的 model、messages、tools 等
type chatRequest struct {
	Model      string        `json:"model"`
	Messages   []chatMessage `json:"messages"`
	Tools      []toolDef     `json:"tools,omitempty"`
	ToolChoice any           `json:"tool_choice,omitempty"`
}

// toolDef 描述请求中传递给模型的工具定义
type toolDef struct {
	Type     string      `json:"type"`
	Function toolFuncDef `json:"function"`
}

// toolFuncDef 描述工具函数的名称、描述和参数 schema
type toolFuncDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// responsesToolDef 描述 OpenAI Responses API 中的工具定义（扁平结构）
type responsesToolDef struct {
	Type        string `json:"type"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// parseChatRequest 一次性解析请求体，避免多次 JSON 反序列化
func parseChatRequest(body string) chatRequest {
	var req chatRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		return chatRequest{}
	}
	return req
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
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"` // 支持字符串或多模态内容数组
	Name       string          `json:"name,omitempty"`
	ToolCalls  []toolCallInfo  `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
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
			content := parseOpenAIContent(msg.Content)
			if strings.TrimSpace(content) != "" {
				sb.WriteString("│\n")
				sb.WriteString("│   📝 Content:\n")
				for _, line := range strings.Split(strings.TrimSpace(content), "\n") {
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
			// 普通消息，支持多模态内容解析
			content := parseOpenAIContent(msg.Content)
			lines := strings.Split(strings.TrimSpace(content), "\n")
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

// extractToolsFromRequest 从已解析的 chatRequest 中提取工具定义信息
func extractToolsFromRequest(req chatRequest) string {
	if len(req.Tools) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 🔧 TOOL DEFINITIONS\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	for i, tool := range req.Tools {
		sb.WriteString(fmt.Sprintf("│ 📌 Tool #%d: %s (type: %s)\n", i+1, tool.Function.Name, tool.Type))
		if tool.Function.Description != "" {
			// 将描述按行输出，增加缩进
			descLines := strings.Split(strings.TrimSpace(tool.Function.Description), "\n")
			sb.WriteString("│   📝 Description:\n")
			for _, line := range descLines {
				sb.WriteString(fmt.Sprintf("│     %s\n", line))
			}
		}
		if tool.Function.Parameters != nil {
			sb.WriteString("│   📋 Parameters:\n")
			var paramsPretty string
			if pretty, err := json.MarshalIndent(tool.Function.Parameters, "│     ", "  "); err == nil {
				paramsPretty = string(pretty)
			}
			if paramsPretty != "" {
				for _, line := range strings.Split(paramsPretty, "\n") {
					sb.WriteString(fmt.Sprintf("│     %s\n", line))
				}
			}
		}
		if i < len(req.Tools)-1 {
			sb.WriteString("│\n")
		}
	}

	// 显示 tool_choice（如果有）
	if req.ToolChoice != nil {
		sb.WriteString("│\n")
		sb.WriteString("│ 🎯 Tool Choice:\n")
		var tcPretty string
		if pretty, err := json.MarshalIndent(req.ToolChoice, "│   ", "  "); err == nil {
			tcPretty = string(pretty)
		}
		if tcPretty != "" {
			for _, line := range strings.Split(tcPretty, "\n") {
				sb.WriteString(fmt.Sprintf("│   %s\n", line))
			}
		}
	}

	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

type anthropicRequest struct {
	Model    string             `json:"model"`
	System   json.RawMessage    `json:"system,omitempty"`
	Messages []anthropicMessage `json:"messages"`
	Tools    []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema,omitempty"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
}

func parseAnthropicRequest(body string) anthropicRequest {
	var req anthropicRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		return anthropicRequest{}
	}
	return req
}

func formatAnthropicContent(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return trimmed
	}

	var lines []string
	for _, block := range blocks {
		switch block.Type {
		case "text":
			if strings.TrimSpace(block.Text) != "" {
				lines = append(lines, block.Text)
			}
		case "tool_use":
			args := strings.TrimSpace(string(block.Input))
			if args == "" {
				args = "{}"
			}
			lines = append(lines, fmt.Sprintf("[tool_use] %s (id: %s)", block.Name, block.ID))
			lines = append(lines, args)
		case "tool_result":
			result := formatAnthropicContent(block.Content)
			if result == "" {
				result = strings.TrimSpace(string(block.Content))
			}
			lines = append(lines, fmt.Sprintf("[tool_result] tool_use_id=%s", block.ToolUseID))
			if result != "" {
				lines = append(lines, result)
			}
		case "image":
			// 只记录图片类型和大小，不记录二进制数据
			var source map[string]any
			if err := json.Unmarshal(block.Content, &source); err == nil {
				mediaType := source["media_type"]
				lines = append(lines, fmt.Sprintf("[image] type: %v, size: %d bytes", mediaType, len(block.Content)))
			} else {
				lines = append(lines, fmt.Sprintf("[image] size: %d bytes", len(block.Content)))
			}
		default:
			// 其他类型的内容，只记录类型和大小，避免日志过大
			lines = append(lines, fmt.Sprintf("[%s] size: %d bytes", block.Type, len(block.Content)))
		}
	}

	return strings.Join(lines, "\n")
}

func extractMessagesFromAnthropicRequest(req anthropicRequest) string {
	if len(req.Messages) == 0 && len(req.System) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 💬 PROMPT DETAILS\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	if len(req.System) > 0 {
		systemText := formatAnthropicContent(req.System)
		if systemText != "" {
			sb.WriteString("│ ⚙️  SYSTEM\n")
			for _, line := range strings.Split(systemText, "\n") {
				if strings.TrimSpace(line) == "" {
					sb.WriteString("│\n")
					continue
				}
				sb.WriteString(fmt.Sprintf("│   %s\n", line))
			}
			if len(req.Messages) > 0 {
				sb.WriteString("│\n")
			}
		}
	}

	for i, msg := range req.Messages {
		sb.WriteString(fmt.Sprintf("│ 👤 %-15s\n", strings.ToUpper(msg.Role)))
		content := formatAnthropicContent(msg.Content)
		for _, line := range strings.Split(content, "\n") {
			if strings.TrimSpace(line) == "" {
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

func extractToolsFromAnthropicRequest(req anthropicRequest) string {
	if len(req.Tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 🔧 TOOL DEFINITIONS\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	for i, tool := range req.Tools {
		sb.WriteString(fmt.Sprintf("│ 📌 Tool #%d: %s\n", i+1, tool.Name))
		if tool.Description != "" {
			sb.WriteString("│   📝 Description:\n")
			for _, line := range strings.Split(strings.TrimSpace(tool.Description), "\n") {
				sb.WriteString(fmt.Sprintf("│     %s\n", line))
			}
		}
		if tool.InputSchema != nil {
			sb.WriteString("│   📋 Input Schema:\n")
			if pretty, err := json.MarshalIndent(tool.InputSchema, "│     ", "  "); err == nil {
				for _, line := range strings.Split(string(pretty), "\n") {
					sb.WriteString(fmt.Sprintf("│     %s\n", line))
				}
			}
		}
		if i < len(req.Tools)-1 {
			sb.WriteString("│\n")
		}
	}

	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

// responsesInputItem 描述 OpenAI Responses API 输入项
type responsesInputItem struct {
	Type      string                 `json:"type"`
	Role      string                 `json:"role,omitempty"`
	Content   []responsesContentPart `json:"content,omitempty"`
	CallID    string                 `json:"call_id,omitempty"`
	Output    string                 `json:"output,omitempty"`
	Name      string                 `json:"name,omitempty"`
	Arguments string                 `json:"arguments,omitempty"`
}

// responsesContentPart 描述 Responses API 输入内容块
type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// responsesRequest 用于解析 OpenAI Responses API 请求体
type responsesRequest struct {
	Model        string             `json:"model"`
	Input        json.RawMessage    `json:"input"`
	Instructions string             `json:"instructions,omitempty"`
	Tools        []responsesToolDef `json:"tools,omitempty"`
	Stream       bool               `json:"stream,omitempty"`
}

func parseResponsesRequest(body string) responsesRequest {
	var req responsesRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		return responsesRequest{}
	}
	return req
}

func extractMessagesFromResponsesRequest(req responsesRequest) string {
	if len(req.Input) == 0 && req.Instructions == "" {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 💬 PROMPT DETAILS\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	if req.Instructions != "" {
		sb.WriteString("│ ⚙️  INSTRUCTIONS\n")
		for _, line := range strings.Split(strings.TrimSpace(req.Instructions), "\n") {
			if strings.TrimSpace(line) == "" {
				sb.WriteString("│\n")
				continue
			}
			sb.WriteString(fmt.Sprintf("│   %s\n", line))
		}
		sb.WriteString("│\n")
	}

	trimmed := strings.TrimSpace(string(req.Input))
	if trimmed == "" || trimmed == "null" {
		sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
		return sb.String()
	}

	var inputStr string
	if err := json.Unmarshal(req.Input, &inputStr); err == nil {
		sb.WriteString("│ 👤 USER\n")
		for _, line := range strings.Split(strings.TrimSpace(inputStr), "\n") {
			if line == "" {
				sb.WriteString("│\n")
				continue
			}
			sb.WriteString(fmt.Sprintf("│   %s\n", line))
		}
	} else {
		var items []responsesInputItem
		if err := json.Unmarshal(req.Input, &items); err != nil {
			sb.WriteString(fmt.Sprintf("│ [unparseable input: %s]\n", truncateString(trimmed, 200)))
		} else {
			for i, item := range items {
				switch item.Type {
				case "message":
					icon := "👤"
					switch strings.ToLower(item.Role) {
					case "system", "developer":
						icon = "⚙️ "
					case "assistant":
						icon = "🤖"
					}
					sb.WriteString(fmt.Sprintf("│ %s %-15s\n", icon, strings.ToUpper(item.Role)))
					for _, part := range item.Content {
						switch part.Type {
						case "input_text", "output_text":
							for _, line := range strings.Split(strings.TrimSpace(part.Text), "\n") {
								if line == "" {
									sb.WriteString("│\n")
									continue
								}
								sb.WriteString(fmt.Sprintf("│   %s\n", line))
							}
						case "input_image":
							sb.WriteString(fmt.Sprintf("│   [image] url: %s\n", part.ImageURL))
						default:
							sb.WriteString(fmt.Sprintf("│   [%s]\n", part.Type))
						}
					}
				case "function_call_output":
					sb.WriteString(fmt.Sprintf("│ 🛠️  FUNCTION CALL OUTPUT (call_id: %s)\n", item.CallID))
					for _, line := range strings.Split(strings.TrimSpace(item.Output), "\n") {
						if line == "" {
							sb.WriteString("│\n")
							continue
						}
						sb.WriteString(fmt.Sprintf("│   %s\n", line))
					}
				case "function_call":
					sb.WriteString(fmt.Sprintf("│ 🔧 FUNCTION CALL: %s (call_id: %s)\n", item.Name, item.CallID))
					var argsPretty string
					var argsMap map[string]any
					if err := json.Unmarshal([]byte(item.Arguments), &argsMap); err == nil {
						if pretty, err := json.MarshalIndent(argsMap, "│     ", "  "); err == nil {
							argsPretty = string(pretty)
						}
					}
					if argsPretty == "" {
						argsPretty = item.Arguments
					}
					for _, line := range strings.Split(argsPretty, "\n") {
						sb.WriteString(fmt.Sprintf("│     %s\n", line))
					}
				}
				if i < len(items)-1 {
					sb.WriteString("│\n")
				}
			}
		}
	}

	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

func extractToolsFromResponsesRequest(req responsesRequest) string {
	if len(req.Tools) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 🔧 TOOL DEFINITIONS\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	for i, tool := range req.Tools {
		name := tool.Name
		if name == "" {
			name = tool.Type
		}
		sb.WriteString(fmt.Sprintf("│ 📌 Tool #%d: %s (type: %s)\n", i+1, name, tool.Type))
		if tool.Description != "" {
			descLines := strings.Split(strings.TrimSpace(tool.Description), "\n")
			sb.WriteString("│   📝 Description:\n")
			for _, line := range descLines {
				sb.WriteString(fmt.Sprintf("│     %s\n", line))
			}
		}
		if tool.Parameters != nil {
			sb.WriteString("│   📋 Parameters:\n")
			var paramsPretty string
			if pretty, err := json.MarshalIndent(tool.Parameters, "│     ", "  "); err == nil {
				paramsPretty = string(pretty)
			}
			if paramsPretty != "" {
				for _, line := range strings.Split(paramsPretty, "\n") {
					sb.WriteString(fmt.Sprintf("│     %s\n", line))
				}
			}
		}
		if i < len(req.Tools)-1 {
			sb.WriteString("│\n")
		}
	}

	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

// responseInfo 保存从响应中提取的结构化信息
type responseInfo struct {
	Content          string // 模型返回的正式文本内容
	ReasoningContent string // 模型的思考过程（OpenAI的reasoning_content、Anthropic的thinking）
	Refusal          string // OpenAI 的 refusal 字段（内容审核拒绝时返回）
	ToolCalls        []toolCallInfo
	FinishReason     string
	Usage            *usageInfo
	RawContent       string // 原始响应文本（用于保存到文件）
	ParseError       string // 解析失败时的诊断信息
}

type usageInfo struct {
	PromptTokens             int `json:"prompt_tokens"`
	CompletionTokens         int `json:"completion_tokens"`
	TotalTokens              int `json:"total_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

type openAIChoiceState struct {
	Content      strings.Builder
	Reasoning    strings.Builder
	Refusal      strings.Builder
	FinishReason string
	ToolCalls    map[int]*toolCallInfo
}

func appendTextSegment(builder *strings.Builder, segment string) {
	if segment == "" {
		return
	}
	if builder.Len() > 0 {
		builder.WriteString("\n")
	}
	builder.WriteString(segment)
}

func trimLeadingNoise(s string) string {
	// \u8df3\u8fc7\u524d\u5bfc\u7a7a\u767d\u3001BOM \u548c SSE \u6ce8\u91ca\u884c\uff08\u4ee5 : \u5f00\u5934\u7684\u884c\uff09
	for {
		s = strings.TrimLeftFunc(s, func(r rune) bool {
			return unicode.IsSpace(r) || r == '\ufeff'
		})
		if !strings.HasPrefix(s, ":") {
			return s
		}
		// \u8df3\u8fc7\u6574\u4e2a SSE \u6ce8\u91ca\u884c
		if idx := strings.IndexByte(s, '\n'); idx >= 0 {
			s = s[idx+1:]
		} else {
			return ""
		}
	}
}

// extractResponseInfo 从响应body中提取模型返回的完整信息，包括工具调用、token用量等
func extractResponseInfo(body, provider string) responseInfo {
	info := responseInfo{RawContent: body}

	if body == "" {
		info.ParseError = "response body is empty"
		return info
	}

	if provider == providerOpenAIResponses {
		trimmed := trimLeadingNoise(body)
		if strings.Contains(body, "\nevent:") || strings.HasPrefix(trimmed, "event:") {
			info.extractResponsesFromStream(body)
		} else {
			info.extractResponsesFromNonStream(body)
		}
		if info.Content == "" && len(info.ToolCalls) == 0 && info.Usage == nil {
			info.ParseError = fmt.Sprintf("failed to parse responses format, body preview: %s", truncateString(body, 200))
		}
		return info
	}

	if provider == providerAnthropic {
		trimmed := trimLeadingNoise(body)
		if strings.Contains(body, "\nevent:") || strings.HasPrefix(trimmed, "event:") {
			info.extractAnthropicFromStream(body)
		} else {
			info.extractAnthropicFromNonStream(body)
		}
		if info.Content == "" && info.ReasoningContent == "" && len(info.ToolCalls) == 0 && info.Usage == nil && info.Refusal == "" {
			if strings.HasPrefix(trimLeadingNoise(body), "data: ") {
				info.extractOpenAIFromStream(body)
			} else {
				info.extractOpenAIFromNonStream(body)
			}
		}
		if info.Content == "" && info.ReasoningContent == "" && len(info.ToolCalls) == 0 && info.Usage == nil && info.Refusal == "" {
			info.ParseError = fmt.Sprintf("failed to parse as anthropic or openai format, body preview: %s", truncateString(body, 200))
		}
		return info
	}

	// OpenAI 流式响应
	if strings.HasPrefix(trimLeadingNoise(body), "data: ") {
		info.extractOpenAIFromStream(body)
	} else {
		info.extractOpenAIFromNonStream(body)
	}
	if info.Content == "" && info.ReasoningContent == "" && len(info.ToolCalls) == 0 && info.Usage == nil && info.Refusal == "" {
		if strings.Contains(body, "\nevent:") || strings.HasPrefix(trimLeadingNoise(body), "event:") {
			info.extractAnthropicFromStream(body)
		} else {
			info.extractAnthropicFromNonStream(body)
		}
	}
	if info.Content == "" && info.ReasoningContent == "" && len(info.ToolCalls) == 0 && info.Usage == nil && info.Refusal == "" {
		info.ParseError = fmt.Sprintf("failed to parse as openai or anthropic format, body preview: %s", truncateString(body, 200))
	}

	return info
}

// streamToolCallDelta 用于流式响应中解析 delta.tool_calls，包含 index 字段
type streamToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// extractFromStream 从 SSE 流式响应中提取信息
func (info *responseInfo) extractOpenAIFromStream(body string) {
	contentBuilder := strings.Builder{}
	reasoningBuilder := strings.Builder{}
	var lastUsage *usageInfo

	lines := strings.Split(body, "\n")
	choiceStates := make(map[int]*openAIChoiceState)

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
				Index int `json:"index"`
				Delta struct {
					Content          string                `json:"content"`
					ReasoningContent string                `json:"reasoning_content"`
					Reasoning        string                `json:"reasoning"`
					Refusal          string                `json:"refusal"`
					ToolCalls        []streamToolCallDelta `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *usageInfo `json:"usage"`
		}
		if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}

		for _, choice := range chunk.Choices {
			state, ok := choiceStates[choice.Index]
			if !ok {
				state = &openAIChoiceState{ToolCalls: make(map[int]*toolCallInfo)}
				choiceStates[choice.Index] = state
			}
			if choice.Delta.Content != "" {
				state.Content.WriteString(choice.Delta.Content)
			}
			if choice.Delta.ReasoningContent != "" {
				state.Reasoning.WriteString(choice.Delta.ReasoningContent)
			} else if choice.Delta.Reasoning != "" {
				state.Reasoning.WriteString(choice.Delta.Reasoning)
			}
			if choice.Delta.Refusal != "" {
				state.Refusal.WriteString(choice.Delta.Refusal)
			}
			// 合并流式 tool_calls（每个 chunk 只包含部分信息）
			for _, tc := range choice.Delta.ToolCalls {
				idx := tc.Index
				if existing, exists := state.ToolCalls[idx]; exists {
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
					state.ToolCalls[idx] = &toolCallInfo{
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
				state.FinishReason = choice.FinishReason
			}
		}
	}

	var refusalBuilder strings.Builder
	if len(choiceStates) > 0 {
		choiceIndexes := make([]int, 0, len(choiceStates))
		for idx := range choiceStates {
			choiceIndexes = append(choiceIndexes, idx)
		}
		sort.Ints(choiceIndexes)
		for _, idx := range choiceIndexes {
			state := choiceStates[idx]
			appendTextSegment(&contentBuilder, state.Content.String())
			appendTextSegment(&reasoningBuilder, state.Reasoning.String())
			if state.Refusal.Len() > 0 {
				appendTextSegment(&refusalBuilder, state.Refusal.String())
			}
			if state.FinishReason != "" {
				info.FinishReason = state.FinishReason
			}
			if len(state.ToolCalls) > 0 {
				toolIndexes := make([]int, 0, len(state.ToolCalls))
				for k := range state.ToolCalls {
					toolIndexes = append(toolIndexes, k)
				}
				sort.Ints(toolIndexes)
				for _, k := range toolIndexes {
					info.ToolCalls = append(info.ToolCalls, *state.ToolCalls[k])
				}
			}
		}
	}
	info.Content = contentBuilder.String()
	info.ReasoningContent = reasoningBuilder.String()
	info.Refusal = refusalBuilder.String()
	info.Usage = lastUsage
}

// extractFromNonStream 从普通（非流式）响应中提取信息
func (info *responseInfo) extractOpenAIFromNonStream(body string) {
	var resp struct {
		Choices []struct {
			Message struct {
				Content          string         `json:"content"`
				ReasoningContent string         `json:"reasoning_content"`
				Reasoning        string         `json:"reasoning"`
				Refusal          string         `json:"refusal"`
				ToolCalls        []toolCallInfo `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usageInfo `json:"usage"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return
	}
	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	for _, choice := range resp.Choices {
		if choice.Message.Content != "" {
			appendTextSegment(&contentBuilder, choice.Message.Content)
		}
		if choice.Message.ReasoningContent != "" {
			appendTextSegment(&reasoningBuilder, choice.Message.ReasoningContent)
		} else if choice.Message.Reasoning != "" {
			appendTextSegment(&reasoningBuilder, choice.Message.Reasoning)
		}
		if choice.Message.Refusal != "" {
			info.Refusal = choice.Message.Refusal
		}
		if len(choice.Message.ToolCalls) > 0 {
			info.ToolCalls = append(info.ToolCalls, choice.Message.ToolCalls...)
		}
		if choice.FinishReason != "" {
			info.FinishReason = choice.FinishReason
		}
	}
	info.Content = contentBuilder.String()
	info.ReasoningContent = reasoningBuilder.String()
	info.Usage = resp.Usage
}

// responsesOutputItem 描述 OpenAI Responses API 输出项
type responsesOutputItem struct {
	Type      string `json:"type"`
	ID        string `json:"id,omitempty"`
	Role      string `json:"role,omitempty"`
	Status    string `json:"status,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Content   []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"content,omitempty"`
}

func (info *responseInfo) extractResponsesFromNonStream(body string) {
	var resp struct {
		Output []responsesOutputItem `json:"output"`
		Status string                `json:"status"`
		Usage  *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return
	}

	var contentBuilder strings.Builder
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					appendTextSegment(&contentBuilder, part.Text)
				}
			}
			if item.Status != "" {
				info.FinishReason = item.Status
			}
		case "function_call":
			tool := toolCallInfo{
				ID:   item.CallID,
				Type: "function",
			}
			tool.Function.Name = item.Name
			tool.Function.Arguments = item.Arguments
			info.ToolCalls = append(info.ToolCalls, tool)
		}
	}
	info.Content = contentBuilder.String()
	if resp.Status != "" {
		info.FinishReason = resp.Status
	}
	if resp.Usage != nil {
		info.Usage = &usageInfo{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
	}
}

func (info *responseInfo) extractResponsesFromStream(body string) {
	var contentBuilder strings.Builder
	streamToolCalls := make(map[string]*toolCallInfo)
	itemIDToCallID := make(map[string]string)
	var finishReason string
	var usage *usageInfo
	currentEvent := ""

	lines := strings.Split(body, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		switch currentEvent {
		case "response.output_text.delta":
			var event struct {
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &event); err == nil {
				contentBuilder.WriteString(event.Delta)
			}

		case "response.function_call_arguments.delta":
			var event struct {
				ItemID     string `json:"item_id"`
				CallID     string `json:"call_id"`
				Delta      string `json:"delta"`
				PartialJSON string `json:"partial_json"`
			}
			if err := json.Unmarshal([]byte(data), &event); err == nil {
				key := event.CallID
				if key == "" {
					key = event.ItemID
				}
				if mapped, ok := itemIDToCallID[key]; ok {
					key = mapped
				}
				delta := event.Delta
				if delta == "" {
					delta = event.PartialJSON
				}
				if existing, ok := streamToolCalls[key]; ok {
					existing.Function.Arguments += delta
				} else {
					tc := &toolCallInfo{
						ID:   key,
						Type: "function",
					}
					tc.Function.Arguments = delta
					streamToolCalls[key] = tc
				}
			}

		case "response.output_item.added":
			var event struct {
				Item responsesOutputItem `json:"item"`
			}
			if err := json.Unmarshal([]byte(data), &event); err == nil {
				if event.Item.Type == "function_call" {
					key := event.Item.CallID
					if key == "" {
						key = event.Item.ID
					}
					tc := &toolCallInfo{
						ID:   key,
						Type: "function",
					}
					tc.Function.Name = event.Item.Name
					streamToolCalls[key] = tc
					if event.Item.ID != "" && event.Item.CallID != "" {
						itemIDToCallID[event.Item.ID] = event.Item.CallID
					}
				}
			}

		case "response.completed":
			var event struct {
				Response struct {
					Status string                `json:"status"`
					Output []responsesOutputItem  `json:"output"`
					Usage  *struct {
						InputTokens  int `json:"input_tokens"`
						OutputTokens int `json:"output_tokens"`
					} `json:"usage"`
				} `json:"response"`
			}
			if err := json.Unmarshal([]byte(data), &event); err == nil {
				if event.Response.Status != "" {
					finishReason = event.Response.Status
				}
				if event.Response.Usage != nil {
					usage = &usageInfo{
						PromptTokens:     event.Response.Usage.InputTokens,
						CompletionTokens: event.Response.Usage.OutputTokens,
						TotalTokens:      event.Response.Usage.InputTokens + event.Response.Usage.OutputTokens,
					}
				}
			}
		}
	}

	info.Content = contentBuilder.String()
	if len(streamToolCalls) > 0 {
		keys := make([]string, 0, len(streamToolCalls))
		for k := range streamToolCalls {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			info.ToolCalls = append(info.ToolCalls, *streamToolCalls[k])
		}
	}
	info.FinishReason = finishReason
	info.Usage = usage
}

func (info *responseInfo) extractAnthropicFromNonStream(body string) {
	var resp struct {
		Content    []anthropicContentBlock `json:"content"`
		StopReason string                  `json:"stop_reason"`
		Usage      *struct {
			InputTokens               int `json:"input_tokens"`
			OutputTokens              int `json:"output_tokens"`
			CreationCacheInputTokens  int `json:"cache_creation_input_tokens"`
			ReadCacheInputTokens      int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		return
	}

	var contentBuilder strings.Builder
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			contentBuilder.WriteString(block.Text)
		case "thinking":
			// Claude 3.7+ 新的思考过程块
			info.ReasoningContent += string(block.Content)
		case "redacted_thinking":
			// 加密的思考过程，记录标记即可
			info.ReasoningContent += "[加密的思考过程，无法读取]"
		case "tool_use":
			tool := toolCallInfo{ID: block.ID, Type: block.Type}
			tool.Function.Name = block.Name
			tool.Function.Arguments = strings.TrimSpace(string(block.Input))
			info.ToolCalls = append(info.ToolCalls, tool)
		}
	}

	info.Content = contentBuilder.String()
	info.FinishReason = resp.StopReason
	if resp.Usage != nil {
		info.Usage = &usageInfo{
			PromptTokens:             resp.Usage.InputTokens,
			CompletionTokens:         resp.Usage.OutputTokens,
			TotalTokens:              resp.Usage.InputTokens + resp.Usage.OutputTokens,
			CacheCreationInputTokens: resp.Usage.CreationCacheInputTokens,
			CacheReadInputTokens:     resp.Usage.ReadCacheInputTokens,
		}
	}
}

func (info *responseInfo) extractAnthropicFromStream(body string) {
	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	reasoningBlocks := make(map[int]string)
	streamToolCalls := make(map[int]*toolCallInfo)
	toolCallHasDelta := make(map[int]bool)
	var finishReason string
	var usage *usageInfo
	currentEvent := ""

	lines := strings.Split(body, "\n")
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		switch currentEvent {
		case "message_start":
			var event struct {
				Message struct {
					Usage struct {
						InputTokens              int `json:"input_tokens"`
						CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
						CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					} `json:"usage"`
				} `json:"message"`
			}
			if err := json.Unmarshal([]byte(data), &event); err == nil {
				usage = &usageInfo{
					PromptTokens:             event.Message.Usage.InputTokens,
					CacheCreationInputTokens: event.Message.Usage.CacheCreationInputTokens,
					CacheReadInputTokens:     event.Message.Usage.CacheReadInputTokens,
				}
			}
		case "content_block_start":
			var event struct {
				Index        int                   `json:"index"`
				ContentBlock anthropicContentBlock `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}
			if event.ContentBlock.Type == "text" {
				contentBuilder.WriteString(event.ContentBlock.Text)
				continue
			}
			if event.ContentBlock.Type == "thinking" {
				continue
			}
			if event.ContentBlock.Type == "redacted_thinking" {
				reasoningBlocks[event.Index] = "[加密的思考过程，无法读取]"
				continue
			}
			if event.ContentBlock.Type == "tool_use" {
				tool := &toolCallInfo{ID: event.ContentBlock.ID, Type: event.ContentBlock.Type}
				tool.Function.Name = event.ContentBlock.Name
				tool.Function.Arguments = strings.TrimSpace(string(event.ContentBlock.Input))
				streamToolCalls[event.Index] = tool
			}
		case "content_block_delta":
			var event struct {
				Index int `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					Thinking    string `json:"thinking"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}
			switch event.Delta.Type {
			case "text_delta":
				contentBuilder.WriteString(event.Delta.Text)
			case "thinking_delta":
				if event.Delta.Thinking != "" {
					reasoningBlocks[event.Index] += event.Delta.Thinking
				} else {
					reasoningBlocks[event.Index] += event.Delta.Text
				}
			case "input_json_delta":
				if _, ok := streamToolCalls[event.Index]; !ok {
					streamToolCalls[event.Index] = &toolCallInfo{Type: "tool_use"}
				}
				if !toolCallHasDelta[event.Index] {
					streamToolCalls[event.Index].Function.Arguments = ""
					toolCallHasDelta[event.Index] = true
				}
				streamToolCalls[event.Index].Function.Arguments += event.Delta.PartialJSON
			}
		case "message_delta":
			var event struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
				Usage struct {
					InputTokens              int `json:"input_tokens"`
					OutputTokens             int `json:"output_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}
			if event.Delta.StopReason != "" {
				finishReason = event.Delta.StopReason
			}
			if usage == nil {
				usage = &usageInfo{}
			}
			if event.Usage.InputTokens != 0 {
				usage.PromptTokens = event.Usage.InputTokens
			}
			usage.CompletionTokens = event.Usage.OutputTokens
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
			if event.Usage.CacheCreationInputTokens != 0 {
				usage.CacheCreationInputTokens = event.Usage.CacheCreationInputTokens
			}
			if event.Usage.CacheReadInputTokens != 0 {
				usage.CacheReadInputTokens = event.Usage.CacheReadInputTokens
			}
		}
	}

	info.Content = contentBuilder.String()
	if len(reasoningBlocks) > 0 {
		indices := make([]int, 0, len(reasoningBlocks))
		for idx := range reasoningBlocks {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		for _, idx := range indices {
			appendTextSegment(&reasoningBuilder, reasoningBlocks[idx])
		}
	}
	info.ReasoningContent = reasoningBuilder.String()
	if len(streamToolCalls) > 0 {
		indices := make([]int, 0, len(streamToolCalls))
		for idx := range streamToolCalls {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		for _, idx := range indices {
			info.ToolCalls = append(info.ToolCalls, *streamToolCalls[idx])
		}
	}
	info.FinishReason = finishReason
	info.Usage = usage
}

// formatResponseContent 格式化响应内容为可读文本
func formatResponseContent(info responseInfo) string {
	if info.Content == "" && info.ReasoningContent == "" && info.Refusal == "" && len(info.ToolCalls) == 0 && info.Usage == nil {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("┌──────────────────────────────────────────────────────────────────────────────────────────────────\n")
	sb.WriteString("│ 🤖 ASSISTANT RESPONSE\n")
	sb.WriteString("├──────────────────────────────────────────────────────────────────────────────────────────────────\n")

	// 思考过程（如果有）
	if info.ReasoningContent != "" {
		sb.WriteString("│ 🧠 思考过程:\n")
		lines := strings.Split(strings.TrimSpace(info.ReasoningContent), "\n")
		for _, line := range lines {
			if line == "" {
				sb.WriteString("│\n")
				continue
			}
			sb.WriteString(fmt.Sprintf("│   %s\n", line))
		}
		sb.WriteString("│\n")
		sb.WriteString("│ 📝 正式回答:\n")
	}

	// 正式文本内容
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

	// 内容审核拒绝
	if info.Refusal != "" {
		if info.Content != "" || info.ReasoningContent != "" {
			sb.WriteString("│\n")
		}
		sb.WriteString("│ 🚫 REFUSAL:\n")
		for _, line := range strings.Split(strings.TrimSpace(info.Refusal), "\n") {
			if line == "" {
				sb.WriteString("│\n")
				continue
			}
			sb.WriteString(fmt.Sprintf("│   %s\n", line))
		}
	}

	// 工具调用
	if len(info.ToolCalls) > 0 {
		if info.Content != "" || info.Refusal != "" {
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
		if info.Usage.CacheCreationInputTokens > 0 || info.Usage.CacheReadInputTokens > 0 {
			sb.WriteString("│\n")
			sb.WriteString("│   🗄️  Cache Details:\n")
			if info.Usage.CacheCreationInputTokens > 0 {
				sb.WriteString(fmt.Sprintf("│      Cache Created:     %d tokens\n", info.Usage.CacheCreationInputTokens))
			}
			if info.Usage.CacheReadInputTokens > 0 {
				sb.WriteString(fmt.Sprintf("│      Cache Read:        %d tokens\n", info.Usage.CacheReadInputTokens))
			}
		}
	}

	sb.WriteString("└──────────────────────────────────────────────────────────────────────────────────────────────────")
	return sb.String()
}

func (s *proxyServer) logExchange(entry exchangeLog, provider string) {
	model, messages, tools, toolCount := extractRequestInfo(entry.RequestBody, provider)
	// 提取响应信息（包括工具调用、token用量等）
	responseInfo := extractResponseInfo(entry.ResponseBody, provider)
	// 格式化响应内容
	responseContent := formatResponseContent(responseInfo)

	// 基本信息放在最前面
	attrs := []any{
		slog.Uint64("requestId", entry.RequestID),
		slog.String("provider", provider),
		slog.Int("statusCode", entry.StatusCode),
		slog.String("model", model),
		slog.Duration("duration", entry.Duration),
		slog.String("remoteAddr", entry.RemoteAddr),
	}

	// 核心内容放在中间，最容易看到
	if !s.disableFullLogging {
		if tools != "" {
			attrs = append(attrs, slog.String("tools", truncateString(tools, s.maxLogFieldSize)))
		}
		if messages != "" {
			attrs = append(attrs, slog.String("messages", truncateString(messages, s.maxLogFieldSize)))
		}
		if responseContent != "" {
			attrs = append(attrs, slog.String("responseContent", truncateString(responseContent, s.maxLogFieldSize)))
		} else if entry.StatusCode == 200 && entry.Error == "" {
			// DEBUG: 响应解析为空时记录原始响应体
			attrs = append(attrs, slog.Int("responseBodyLen", len(entry.ResponseBody)))
			attrs = append(attrs, slog.String("responseBodyPreview", truncateString(entry.ResponseBody, 500)))
		}
		if responseInfo.ParseError != "" {
			attrs = append(attrs, slog.String("parseError", responseInfo.ParseError))
		}
		if responseInfo.Refusal != "" {
			attrs = append(attrs, slog.String("refusal", truncateString(responseInfo.Refusal, s.maxLogFieldSize)))
		}
	} else {
		attrs = append(attrs, slog.String("logging", "full content logging disabled for security"))
	}

	// 将消息保存到单独的文件中
	if s.msgDir != "" && !s.disableFullLogging {
		s.saveMessageToFile(entry.RequestID, provider, entry.Path, entry.UpstreamURL, model, toolCount, tools, messages, responseContent, responseInfo)
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
		// 客户端断开的错误打Warn级别
		if strings.Contains(entry.Error, "context canceled") ||
			strings.Contains(entry.Error, "client disconnected") ||
			strings.Contains(entry.Error, "broken pipe") ||
			strings.Contains(entry.Error, "connection reset by peer") {
			s.logger.Warn("⚠️ 客户端断开连接", attrs...)
		} else {
			s.logger.Error("❌ 请求失败", attrs...)
		}
		return
	}
	s.logger.Info("✅ 请求成功", attrs...)
}

func extractRequestInfo(body, provider string) (model, messages, tools string, toolCount int) {
	if provider == providerOpenAIResponses {
		req := parseResponsesRequest(body)
		return req.Model, extractMessagesFromResponsesRequest(req), extractToolsFromResponsesRequest(req), len(req.Tools)
	}
	if provider == providerAnthropic {
		req := parseAnthropicRequest(body)
		return req.Model, extractMessagesFromAnthropicRequest(req), extractToolsFromAnthropicRequest(req), len(req.Tools)
	}

	req := parseChatRequest(body)
	return req.Model, extractMessagesFromRequest(req), extractToolsFromRequest(req), len(req.Tools)
}

func (s *proxyServer) saveMessageToFile(reqID uint64, provider, requestPath, upstreamURL, model string, toolCount int, tools, messages, response string, respInfo responseInfo) {
	now := time.Now()
	timestamp := now.Format("20060102_150405")
	// 文件名包含model，方便搜索
	safeModel := strings.ReplaceAll(model, "/", "_")
	filename := fmt.Sprintf("%s_%s_%04d_%s.txt", timestamp, provider, reqID, safeModel)

	var filePath string
	providerDir := filepath.Join(s.msgDir, provider)
	if s.createDateSubdirs {
		// 按年月/年月日创建子目录
		monthDir := now.Format("200601")
		dayDir := now.Format("20060102")
		fullDir := filepath.Join(providerDir, monthDir, dayDir)
		if err := os.MkdirAll(fullDir, 0o700); err != nil {
			s.logger.Error("failed to create date subdirectory", "path", fullDir, "error", err)
			// 降级到provider目录
			filePath = filepath.Join(providerDir, filename)
		} else {
			filePath = filepath.Join(fullDir, filename)
		}
	} else {
		if err := os.MkdirAll(providerDir, 0o700); err != nil {
			s.logger.Error("failed to create provider directory", "path", providerDir, "error", err)
			filePath = filepath.Join(s.msgDir, filename)
		} else {
			filePath = filepath.Join(providerDir, filename)
		}
	}

	var content strings.Builder
	content.WriteString("╔══════════════════════════════════════════════════════════════════════════════════════════════════\n")
	content.WriteString(fmt.Sprintf("║ 🆔 Request ID: %d\n", reqID))
	content.WriteString(fmt.Sprintf("║ ⏰ Timestamp:  %s\n", time.Now().Format(time.RFC3339)))
	content.WriteString(fmt.Sprintf("║ 🧭 Provider:   %s\n", provider))
	content.WriteString(fmt.Sprintf("║ 🛣️  Path:       %s\n", requestPath))
	if upstreamURL != "" {
		content.WriteString(fmt.Sprintf("║ 🔗 Upstream:   %s\n", upstreamURL))
	}
	content.WriteString(fmt.Sprintf("║ 🧠 Model:      %s\n", model))
	if respInfo.FinishReason != "" {
		content.WriteString(fmt.Sprintf("║ 🏁 Finish:     %s\n", respInfo.FinishReason))
	}
	if respInfo.Usage != nil {
		content.WriteString(fmt.Sprintf("║ 📊 Tokens:     prompt=%d, completion=%d, total=%d\n",
			respInfo.Usage.PromptTokens, respInfo.Usage.CompletionTokens, respInfo.Usage.TotalTokens))
		if respInfo.Usage.CacheCreationInputTokens > 0 || respInfo.Usage.CacheReadInputTokens > 0 {
			if respInfo.Usage.CacheCreationInputTokens > 0 {
				content.WriteString(fmt.Sprintf("║ 🗄️  Cache:      created=%d tokens\n", respInfo.Usage.CacheCreationInputTokens))
			}
			if respInfo.Usage.CacheReadInputTokens > 0 {
				content.WriteString(fmt.Sprintf("║ 🗄️  Cache:      read=%d tokens\n", respInfo.Usage.CacheReadInputTokens))
			}
		}
	}
	if respInfo.Refusal != "" {
		content.WriteString(fmt.Sprintf("║ 🚫 Refusal:    %s\n", truncateString(respInfo.Refusal, 200)))
	}
	if len(respInfo.ToolCalls) > 0 {
		content.WriteString(fmt.Sprintf("║ 🔧 Tool Calls: %d\n", len(respInfo.ToolCalls)))
		for i, tc := range respInfo.ToolCalls {
			content.WriteString(fmt.Sprintf("║    #%d %s (id: %s)\n", i+1, tc.Function.Name, tc.ID))
		}
	}
	if toolCount > 0 {
		content.WriteString(fmt.Sprintf("║ 🛠️  Tools:      %d defined\n", toolCount))
	}
	if respInfo.ParseError != "" {
		content.WriteString(fmt.Sprintf("║ ⚠️  Parse Error: %s\n", respInfo.ParseError))
	}
	content.WriteString("╚══════════════════════════════════════════════════════════════════════════════════════════════════\n\n")

	if tools != "" {
		content.WriteString(tools)
		content.WriteString("\n\n")
	}

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

func copyResponseBody(dst io.Writer, src io.Reader, buf *strings.Builder, maxLogBytes int) error {
	data := make([]byte, 32*1024)
	var dstErr error
	dstBroken := false
	for {
		n, err := src.Read(data)
		if n > 0 {
			chunk := data[:n]
			if buf != nil {
				if maxLogBytes <= 0 {
					_, _ = buf.Write(chunk)
				} else {
					remaining := maxLogBytes - buf.Len()
					if remaining > 0 {
						if remaining > len(chunk) {
							remaining = len(chunk)
						}
						_, _ = buf.Write(chunk[:remaining])
					}
				}
			}
			if !dstBroken {
				written, writeErr := dst.Write(chunk)
				if writeErr != nil {
					dstErr = writeErr
					dstBroken = true
				} else if written != len(chunk) {
					dstErr = io.ErrShortWrite
					dstBroken = true
				}
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return dstErr
			}
			return err
		}
	}
}

type limitedReadCloser struct {
	*io.LimitedReader
	closers []io.Closer
}

func newLimitedReadCloser(reader io.Reader, maxSize int64, closers ...io.Closer) io.ReadCloser {
	return &limitedReadCloser{
		LimitedReader: &io.LimitedReader{R: reader, N: maxSize},
		closers:       closers,
	}
}

func (l *limitedReadCloser) Close() error {
	var err error
	for _, closer := range l.closers {
		if closer == nil {
			continue
		}
		err = errors.Join(err, closer.Close())
	}
	return err
}

// decompressResponseBody 解压gzip/deflate压缩的响应体，以便日志正确记录
func decompressResponseBody(resp *http.Response) (io.ReadCloser, bool) {
	encoding := strings.ToLower(resp.Header.Get("Content-Encoding"))
	switch encoding {
	case "gzip":
		gr, err := gzip.NewReader(resp.Body)
		if err != nil {
			return resp.Body, false
		}
		return gr, true
	case "deflate":
		return flate.NewReader(resp.Body), true
	default:
		return resp.Body, false
	}
}

// decompressRequestBody 解压gzip/deflate压缩的请求体，防止zip bomb
func decompressRequestBody(r *http.Request, maxSize int64) (io.ReadCloser, error) {
	encoding := strings.ToLower(r.Header.Get("Content-Encoding"))
	switch encoding {
	case "gzip":
		gr, err := gzip.NewReader(r.Body)
		if err != nil {
			return nil, err
		}
		return newLimitedReadCloser(gr, maxSize, gr, r.Body), nil
	case "deflate":
		fr := flate.NewReader(r.Body)
		return newLimitedReadCloser(fr, maxSize, fr, r.Body), nil
	default:
		return newLimitedReadCloser(r.Body, maxSize, r.Body), nil
	}
}

// truncateString 截断过长的字符串，避免日志过大
func truncateString(s string, maxLen int) string {
	if maxLen <= 0 || len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + fmt.Sprintf("...[truncated, total %d bytes]", len(s))
}

// parseOpenAIContent 解析OpenAI的内容字段，支持字符串和多模态数组
func parseOpenAIContent(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	// 尝试解析为字符串
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}
	// 尝试解析为内容块数组
	var blocks []map[string]any
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return string(raw)
	}
	var lines []string
	for _, block := range blocks {
		blockType, _ := block["type"].(string)
		switch blockType {
		case "text":
			if text, ok := block["text"].(string); ok && strings.TrimSpace(text) != "" {
				lines = append(lines, text)
			}
		case "image_url":
			imageUrl, ok := block["image_url"].(map[string]any)
			if ok {
				url, _ := imageUrl["url"].(string)
				if strings.HasPrefix(url, "data:") {
					// base64图片，只记录类型和大小
					parts := strings.SplitN(url, ";", 2)
					mediaType := "unknown"
					if len(parts) > 0 {
						mediaType = strings.TrimPrefix(parts[0], "data:")
					}
					lines = append(lines, fmt.Sprintf("[image] type: %s, size: %d bytes", mediaType, len(url)))
				} else {
					lines = append(lines, fmt.Sprintf("[image] url: %s", url))
				}
			} else {
				lines = append(lines, "[image]")
			}
		default:
			lines = append(lines, fmt.Sprintf("[%s] size: %d bytes", blockType, len(raw)))
		}
	}
	return strings.Join(lines, "\n")
}

func selectedHeaders(h http.Header) map[string]string {
	selected := make(map[string]string)
	for _, key := range []string{"Content-Type", "Transfer-Encoding", "Content-Length", "OpenAI-Model", "X-Request-Id", "Anthropic-Version"} {
		if value := h.Get(key); value != "" {
			selected[key] = value
		}
	}
	if len(selected) == 0 {
		return nil
	}
	return selected
}
