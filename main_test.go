package main

import (
	"bytes"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
)

func TestBuildUpstreamURLByProvider(t *testing.T) {
	openaiURL, err := buildUpstreamURL("https://api.openai.com/v1", providerOpenAI)
	if err != nil {
		t.Fatalf("build openai upstream url: %v", err)
	}
	if openaiURL != "https://api.openai.com/v1/chat/completions" {
		t.Fatalf("unexpected openai upstream url: %s", openaiURL)
	}

	anthropicURL, err := buildUpstreamURL("https://api.anthropic.com", providerAnthropic)
	if err != nil {
		t.Fatalf("build anthropic upstream url: %v", err)
	}
	if anthropicURL != "https://api.anthropic.com/v1/messages" {
		t.Fatalf("unexpected anthropic upstream url: %s", anthropicURL)
	}
}

func TestBuildRouteTargetsSupportsSingleProvider(t *testing.T) {
	tests := []struct {
		name         string
		cfg          proxyConfig
		wantPath     string
		wantProvider string
		wantUpstream string
		wantCount    int
	}{
		{
			name: "openai only",
			cfg: proxyConfig{
				openaiUpstreamBaseURL: "https://api.openai.com/v1",
				openaiAPIKey:          "openai-key",
			},
			wantPath:     "/v1/chat/completions",
			wantProvider: providerOpenAI,
			wantUpstream: "https://api.openai.com/v1/chat/completions",
			wantCount:    2, // chat/completions + responses
		},
		{
			name: "anthropic only",
			cfg: proxyConfig{
				anthropicUpstreamBaseURL: "https://api.anthropic.com",
				anthropicAPIKey:          "anthropic-key",
			},
			wantPath:     "/v1/messages",
			wantProvider: providerAnthropic,
			wantUpstream: "https://api.anthropic.com/v1/messages",
			wantCount:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			targets, err := buildRouteTargets(tt.cfg)
			if err != nil {
				t.Fatalf("build route targets: %v", err)
			}
			if len(targets) != tt.wantCount {
				t.Fatalf("expected %d route target(s), got %d", tt.wantCount, len(targets))
			}
			target, ok := targets[tt.wantPath]
			if !ok {
				t.Fatalf("missing target for path %s: %+v", tt.wantPath, targets)
			}
			if target.provider != tt.wantProvider {
				t.Fatalf("unexpected provider: %s", target.provider)
			}
			if target.upstreamURL != tt.wantUpstream {
				t.Fatalf("unexpected upstream url: %s", target.upstreamURL)
			}
		})
	}
}

func TestNewProxyMuxRoutesAlwaysAvailable(t *testing.T) {
	cfg := proxyConfig{
		healthzPath: "/healthz",
	}
	// 添加两个空的上游，让路由存在
	server := &proxyServer{routeTargets: map[string]upstreamTarget{
		"/v1/chat/completions": {provider: providerOpenAI},
		"/v1/messages":         {provider: providerAnthropic},
	}}
	mux := newProxyMux(server, cfg)

	resp := httptest.NewRecorder()
	mux.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil))
	if resp.Code != http.StatusMethodNotAllowed {
		t.Fatalf("/v1/chat/completions should exist and reject GET with 405, got %d", resp.Code)
	}

	resp = httptest.NewRecorder()
	mux.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/v1/messages", nil))
	if resp.Code != http.StatusMethodNotAllowed {
		t.Fatalf("/v1/messages should exist and reject GET with 405, got %d", resp.Code)
	}

	resp = httptest.NewRecorder()
	mux.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/v1/unknown", nil))
	if resp.Code != http.StatusNotFound {
		t.Fatalf("unknown path should return 404, got %d", resp.Code)
	}
}

func TestNewProxyMuxEnableCORS(t *testing.T) {
	cfg := proxyConfig{
		healthzPath: "/healthz",
		enableCORS:  true,
	}
	server := &proxyServer{routeTargets: map[string]upstreamTarget{
		"/v1/chat/completions": {provider: providerOpenAI},
	}}
	mux := newProxyMux(server, cfg)

	preflight := httptest.NewRecorder()
	mux.ServeHTTP(preflight, httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil))
	if preflight.Code != http.StatusNoContent {
		t.Fatalf("expected preflight 204, got %d", preflight.Code)
	}
	if preflight.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Fatalf("expected CORS allow origin header, got %q", preflight.Header().Get("Access-Control-Allow-Origin"))
	}

	healthz := httptest.NewRecorder()
	mux.ServeHTTP(healthz, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if healthz.Code != http.StatusOK {
		t.Fatalf("expected healthz 200, got %d", healthz.Code)
	}
	if healthz.Header().Get("Access-Control-Allow-Methods") == "" {
		t.Fatalf("expected CORS allow methods header")
	}
}

func TestDualUpstreamProxyForwardsByPathAndAuthHeaders(t *testing.T) {
	var receivedPath string
	var receivedAuth string
	var receivedAPIKey string

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedAuth = r.Header.Get("Authorization")
		receivedAPIKey = r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer openaiUpstream.Close()

	var anthropicPath string
	var anthropicAuth string
	var anthropicAPIKey string
	var anthropicVersion string
	var anthropicBeta string

	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		anthropicPath = r.URL.Path
		anthropicAuth = r.Header.Get("Authorization")
		anthropicAPIKey = r.Header.Get("X-Api-Key")
		anthropicVersion = r.Header.Get("anthropic-version")
		anthropicBeta = r.Header.Get("anthropic-beta")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"type":"message","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`)
	}))
	defer anthropicUpstream.Close()

	openaiUpstreamURL, err := buildUpstreamURL(openaiUpstream.URL, providerOpenAI)
	if err != nil {
		t.Fatalf("build openai upstream url: %v", err)
	}
	anthropicUpstreamURL, err := buildUpstreamURL(anthropicUpstream.URL, providerAnthropic)
	if err != nil {
		t.Fatalf("build anthropic upstream url: %v", err)
	}

	tmpDir := t.TempDir()

	proxy := &proxyServer{
		routeTargets: map[string]upstreamTarget{
			"/v1/chat/completions": {
				provider:    providerOpenAI,
				upstreamURL: openaiUpstreamURL,
				apiKey:      "openai-proxy-key",
			},
			"/v1/messages": {
				provider:    providerAnthropic,
				upstreamURL: anthropicUpstreamURL,
				apiKey:      "anthropic-proxy-key",
			},
		},
		logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		client: &http.Client{},
		msgDir: tmpDir,
	}
	cfg := proxyConfig{
		healthzPath: "/healthz",
	}
	mux := newProxyMux(proxy, cfg)

	openaiBody := `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello-openai"}]}`
	openaiReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(openaiBody))
	openaiReq.Header.Set("Content-Type", "application/json")
	openaiReq.Header.Set("X-Api-Key", "client-x-api-key")
	openaiResp := httptest.NewRecorder()
	mux.ServeHTTP(openaiResp, openaiReq)

	if openaiResp.Code != http.StatusOK {
		t.Fatalf("unexpected openai proxy status: %d", openaiResp.Code)
	}
	if receivedPath != "/chat/completions" {
		t.Fatalf("unexpected openai upstream path: %s", receivedPath)
	}
	if receivedAuth != "Bearer openai-proxy-key" {
		t.Fatalf("openai proxy should inject Authorization, got: %s", receivedAuth)
	}
	if receivedAPIKey != "" {
		t.Fatalf("openai upstream request should not include X-Api-Key, got: %s", receivedAPIKey)
	}

	anthropicBody := `{"model":"claude-3-5-sonnet","system":"system-prompt","messages":[{"role":"user","content":[{"type":"text","text":"hello-anthropic"}]}]}`
	anthropicReq := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(anthropicBody))
	anthropicReq.Header.Set("Content-Type", "application/json")
	anthropicReq.Header.Set("Authorization", "Bearer client-should-be-removed")
	anthropicReq.Header.Set("anthropic-version", "2023-06-01")
	anthropicReq.Header.Set("anthropic-beta", "tools-2024-04-04")
	anthropicResp := httptest.NewRecorder()
	mux.ServeHTTP(anthropicResp, anthropicReq)

	if anthropicResp.Code != http.StatusOK {
		t.Fatalf("unexpected anthropic proxy status: %d", anthropicResp.Code)
	}
	if anthropicPath != "/v1/messages" {
		t.Fatalf("unexpected anthropic upstream path: %s", anthropicPath)
	}
	if anthropicAPIKey != "anthropic-proxy-key" {
		t.Fatalf("anthropic proxy should inject X-Api-Key, got: %s", anthropicAPIKey)
	}
	if anthropicAuth != "" {
		t.Fatalf("anthropic upstream request should not include Authorization, got: %s", anthropicAuth)
	}
	if anthropicVersion != "2023-06-01" {
		t.Fatalf("proxy should pass through anthropic-version, got: %s", anthropicVersion)
	}
	if anthropicBeta != "tools-2024-04-04" {
		t.Fatalf("proxy should pass through anthropic-beta, got: %s", anthropicBeta)
	}

	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("read msg dir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 provider subdirs, got %d", len(entries))
	}

	var providerDirs []string
	for _, entry := range entries {
		providerDirs = append(providerDirs, entry.Name())
	}
	sort.Strings(providerDirs)

	if providerDirs[0] != "anthropic" || providerDirs[1] != "openai" {
		t.Fatalf("expected provider subdirs [anthropic, openai], got: %v", providerDirs)
	}

	openaiFiles, err := os.ReadDir(filepath.Join(tmpDir, "openai"))
	if err != nil {
		t.Fatalf("read openai dir: %v", err)
	}
	if len(openaiFiles) != 1 {
		t.Fatalf("expected 1 openai message file, got %d", len(openaiFiles))
	}
	anthropicFiles, err := os.ReadDir(filepath.Join(tmpDir, "anthropic"))
	if err != nil {
		t.Fatalf("read anthropic dir: %v", err)
	}
	if len(anthropicFiles) != 1 {
		t.Fatalf("expected 1 anthropic message file, got %d", len(anthropicFiles))
	}

	openaiContentBytes, err := os.ReadFile(filepath.Join(tmpDir, "openai", openaiFiles[0].Name()))
	if err != nil {
		t.Fatalf("read openai message file: %v", err)
	}
	anthropicContentBytes, err := os.ReadFile(filepath.Join(tmpDir, "anthropic", anthropicFiles[0].Name()))
	if err != nil {
		t.Fatalf("read anthropic message file: %v", err)
	}

	openaiContent := string(openaiContentBytes)
	anthropicContent := string(anthropicContentBytes)

	if !strings.Contains(openaiContent, "Provider:   openai") || !strings.Contains(openaiContent, "Path:       /v1/chat/completions") {
		t.Fatalf("openai file should include provider/path markers, content: %s", openaiContent)
	}
	if !strings.Contains(anthropicContent, "Provider:   anthropic") || !strings.Contains(anthropicContent, "Path:       /v1/messages") {
		t.Fatalf("anthropic file should include provider/path markers, content: %s", anthropicContent)
	}
	if !strings.Contains(openaiContent, "Finish:     stop") {
		t.Fatalf("openai extraction should include finish reason stop, content: %s", openaiContent)
	}
	if !strings.Contains(anthropicContent, "Finish:     end_turn") {
		t.Fatalf("anthropic extraction should include finish reason end_turn, content: %s", anthropicContent)
	}
	if strings.Contains(openaiContent, "Finish:     end_turn") {
		t.Fatalf("openai extraction should not use anthropic parser, content: %s", openaiContent)
	}
	if strings.Contains(anthropicContent, "Finish:     stop") {
		t.Fatalf("anthropic extraction should not use openai parser, content: %s", anthropicContent)
	}
}

func TestParseFlagsRequiresDualUpstreamConfig(t *testing.T) {
	// 测试同时配置两个上游，应该成功
	cfg, err := runParseFlagsWithArgs(t, []string{
		"--openai-upstream-base-url", "https://api.openai.com/v1",
		"--openai-api-key", "openai-key",
		"--anthropic-upstream-base-url", "https://api.anthropic.com",
		"--anthropic-api-key", "anthropic-key",
	})
	if err != nil {
		t.Fatalf("expected parseFlags success with complete dual config: %v", err)
	}
	if cfg.openaiAPIKey != "openai-key" || cfg.anthropicAPIKey != "anthropic-key" {
		t.Fatalf("unexpected parsed api keys: %+v", cfg)
	}

	// 测试只配置openai，应该成功
	cfg, err = runParseFlagsWithArgs(t, []string{
		"--openai-upstream-base-url", "https://api.openai.com/v1",
		"--openai-api-key", "openai-key",
	})
	if err != nil {
		t.Fatalf("expected parseFlags success with only openai config: %v", err)
	}
	if cfg.openaiAPIKey != "openai-key" || cfg.anthropicAPIKey != "" {
		t.Fatalf("unexpected parsed api keys for single openai config: %+v", cfg)
	}

	// 测试只配置anthropic，应该成功
	cfg, err = runParseFlagsWithArgs(t, []string{
		"--anthropic-upstream-base-url", "https://api.anthropic.com",
		"--anthropic-api-key", "anthropic-key",
	})
	if err != nil {
		t.Fatalf("expected parseFlags success with only anthropic config: %v", err)
	}
	if cfg.anthropicAPIKey != "anthropic-key" || cfg.openaiAPIKey != "" {
		t.Fatalf("unexpected parsed api keys for single anthropic config: %+v", cfg)
	}

	// 测试anthropic配置不完整且没有其他上游，应该失败
	_, err = runParseFlagsWithArgs(t, []string{
		"--anthropic-upstream-base-url", "https://api.anthropic.com",
	})
	if err == nil {
		t.Fatalf("expected error when anthropic config is incomplete and no other upstream, got nil")
	}

	// 测试openai配置不完整且没有其他上游，应该失败
	_, err = runParseFlagsWithArgs(t, []string{
		"--openai-api-key", "openai-key",
	})
	if err == nil {
		t.Fatalf("expected error when openai config is incomplete and no other upstream, got nil")
	}

	// 测试两个都不配置，应该失败
	_, err = runParseFlagsWithArgs(t, []string{})
	if err == nil || !strings.Contains(err.Error(), "at least one upstream") {
		t.Fatalf("expected error when no upstream configured, got: %v", err)
	}
}

func runParseFlagsWithArgs(t *testing.T, args []string) (proxyConfig, error) {
	t.Helper()
	originalArgs := os.Args
	originalCommandLine := flag.CommandLine

	os.Args = append([]string{"llm_recorder"}, args...)
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	flag.CommandLine.SetOutput(io.Discard)

	cfg, err := parseFlags()

	os.Args = originalArgs
	flag.CommandLine = originalCommandLine

	return cfg, err
}

func TestExtractResponseInfoAnthropicNonStream(t *testing.T) {
	body := `{"type":"message","stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":8},"content":[{"type":"text","text":"hello"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Beijing"}}]}`
	info := extractResponseInfo(body, providerAnthropic)

	if info.Content != "hello" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.FinishReason != "end_turn" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 12 || info.Usage.CompletionTokens != 8 || info.Usage.TotalTokens != 20 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
	if len(info.ToolCalls) != 1 {
		t.Fatalf("unexpected tool call count: %d", len(info.ToolCalls))
	}
	if info.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected tool name: %s", info.ToolCalls[0].Function.Name)
	}
	if strings.TrimSpace(info.ToolCalls[0].Function.Arguments) != `{"city":"Beijing"}` {
		t.Fatalf("unexpected tool args: %s", info.ToolCalls[0].Function.Arguments)
	}
}

func TestExtractResponseInfoAnthropicStream(t *testing.T) {
	body := strings.Join([]string{
		"event: message_start",
		"data: {\"message\":{\"usage\":{\"input_tokens\":10}}}",
		"",
		"event: content_block_start",
		"data: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Hi\"}}",
		"",
		"event: content_block_delta",
		"data: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}",
		"",
		"event: content_block_start",
		"data: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_123\",\"name\":\"get_weather\"}}",
		"",
		"event: content_block_delta",
		"data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Bei\"}}",
		"",
		"event: content_block_delta",
		"data: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"jing\\\"}\"}}",
		"",
		"event: message_delta",
		"data: {\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":4}}",
	}, "\n")

	info := extractResponseInfo(body, providerAnthropic)
	if info.Content != "Hi there" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.FinishReason != "tool_use" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 10 || info.Usage.CompletionTokens != 4 || info.Usage.TotalTokens != 14 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
	if len(info.ToolCalls) != 1 {
		t.Fatalf("unexpected tool call count: %d", len(info.ToolCalls))
	}
	if info.ToolCalls[0].ID != "toolu_123" || info.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected tool call: %+v", info.ToolCalls[0])
	}
	if info.ToolCalls[0].Function.Arguments != `{"city":"Beijing"}` {
		t.Fatalf("unexpected tool args: %s", info.ToolCalls[0].Function.Arguments)
	}
}

func TestExtractResponseInfoOpenAINonStream(t *testing.T) {
	body := `{"choices":[{"index":0,"message":{"content":"","reasoning_content":"step one","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"alpha\"}"}}]},"finish_reason":"tool_calls"},{"index":1,"message":{"content":"final answer","reasoning_content":"step two"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}`
	info := extractResponseInfo(body, providerOpenAI)

	if info.Content != "final answer" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.ReasoningContent != "step one\nstep two" {
		t.Fatalf("unexpected reasoning content: %q", info.ReasoningContent)
	}
	if info.FinishReason != "stop" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 5 || info.Usage.CompletionTokens != 7 || info.Usage.TotalTokens != 12 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
	if len(info.ToolCalls) != 1 || info.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("unexpected tool calls: %+v", info.ToolCalls)
	}
}

func TestExtractResponseInfoOpenAIStream(t *testing.T) {
	body := strings.Join([]string{
		"  data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"alpha\\\"}\"}}]}}]}",
		"data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}",
		"data: [DONE]",
	}, "\n")

	info := extractResponseInfo(body, providerOpenAI)
	if info.Content != "Hello" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.ReasoningContent != "think" {
		t.Fatalf("unexpected reasoning content: %q", info.ReasoningContent)
	}
	if info.FinishReason != "stop" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if len(info.ToolCalls) != 1 || info.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("unexpected tool calls: %+v", info.ToolCalls)
	}
}

func TestExtractResponseInfoAnthropicStreamThinking(t *testing.T) {
	body := strings.Join([]string{
		"event: message_start",
		"data: {\"message\":{\"usage\":{\"input_tokens\":3}}}",
		"",
		"event: content_block_start",
		"data: {\"index\":0,\"content_block\":{\"type\":\"thinking\"}}",
		"",
		"event: content_block_delta",
		"data: {\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"First thought\"}}",
		"",
		"event: content_block_start",
		"data: {\"index\":1,\"content_block\":{\"type\":\"redacted_thinking\"}}",
		"",
		"event: content_block_start",
		"data: {\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"Answer\"}}",
		"",
		"event: message_delta",
		"data: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}",
	}, "\n")

	info := extractResponseInfo(body, providerAnthropic)
	if info.Content != "Answer" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if !strings.Contains(info.ReasoningContent, "First thought") {
		t.Fatalf("unexpected reasoning content: %q", info.ReasoningContent)
	}
	if !strings.Contains(info.ReasoningContent, "加密的思考过程") {
		t.Fatalf("expected redacted thinking marker, got: %q", info.ReasoningContent)
	}
	if info.FinishReason != "end_turn" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 3 || info.Usage.CompletionTokens != 2 || info.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
}

func TestExtractResponseInfoAnthropicStreamPromptTokensFromMessageDelta(t *testing.T) {
	body := strings.Join([]string{
		"event: content_block_start",
		"data: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Answer\"}}",
		"",
		"event: message_delta",
		"data: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":9,\"output_tokens\":4}}",
	}, "\n")

	info := extractResponseInfo(body, providerAnthropic)
	if info.Content != "Answer" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.FinishReason != "end_turn" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 9 || info.Usage.CompletionTokens != 4 || info.Usage.TotalTokens != 13 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
}

func TestExtractResponseInfoAnthropicProviderFallsBackToOpenAIStream(t *testing.T) {
	body := strings.Join([]string{
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}",
		"data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":2,\"total_tokens\":8}}",
		"data: [DONE]",
	}, "\n")

	info := extractResponseInfo(body, providerAnthropic)
	if info.Content != "Hello" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.FinishReason != "stop" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 6 || info.Usage.CompletionTokens != 2 || info.Usage.TotalTokens != 8 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
}

type countingReadCloser struct {
	io.ReadCloser
	closeCalls int
}

func (c *countingReadCloser) Close() error {
	c.closeCalls++
	return c.ReadCloser.Close()
}

type countingCloser struct {
	closeCalls int
}

func (c *countingCloser) Close() error {
	c.closeCalls++
	return nil
}

func TestDecompressRequestBodyCloseClosesRequestBody(t *testing.T) {
	var gzipped bytes.Buffer
	zw := gzip.NewWriter(&gzipped)
	if _, err := zw.Write([]byte(`{"message":"hello"}`)); err != nil {
		t.Fatalf("write gzip body: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close gzip writer: %v", err)
	}

	reqBody := &countingReadCloser{ReadCloser: io.NopCloser(bytes.NewReader(gzipped.Bytes()))}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Body = reqBody
	req.Header.Set("Content-Encoding", "gzip")

	bodyReader, err := decompressRequestBody(req, maxRequestBodySize)
	if err != nil {
		t.Fatalf("decompress request body: %v", err)
	}

	if _, err := io.ReadAll(bodyReader); err != nil {
		t.Fatalf("read decompressed body: %v", err)
	}
	if err := bodyReader.Close(); err != nil {
		t.Fatalf("close body reader: %v", err)
	}
	if reqBody.closeCalls != 1 {
		t.Fatalf("expected request body to be closed once, got %d", reqBody.closeCalls)
	}
}

func TestLimitedReadCloserCloseClosesAllClosers(t *testing.T) {
	readerCloser := &countingCloser{}
	bodyCloser := &countingCloser{}

	bodyReader := newLimitedReadCloser(strings.NewReader("hello"), 5, readerCloser, bodyCloser)
	if _, err := io.ReadAll(bodyReader); err != nil {
		t.Fatalf("read body reader: %v", err)
	}
	if err := bodyReader.Close(); err != nil {
		t.Fatalf("close body reader: %v", err)
	}
	if readerCloser.closeCalls != 1 {
		t.Fatalf("expected reader closer to be called once, got %d", readerCloser.closeCalls)
	}
	if bodyCloser.closeCalls != 1 {
		t.Fatalf("expected body closer to be called once, got %d", bodyCloser.closeCalls)
	}
}

func TestCopyResponseBodyLimitsLoggedBuffer(t *testing.T) {
	var dst bytes.Buffer
	var logged strings.Builder

	err := copyResponseBody(&dst, strings.NewReader("abcdefgh"), &logged, 5)
	if err != nil {
		t.Fatalf("copy response body: %v", err)
	}
	if dst.String() != "abcdefgh" {
		t.Fatalf("unexpected copied response body: %q", dst.String())
	}
	if logged.String() != "abcde" {
		t.Fatalf("unexpected logged response body: %q", logged.String())
	}
}

// failAfterWriter 写入 failAfter 字节后返回错误，模拟客户端中途断开
type failAfterWriter struct {
	buf       bytes.Buffer
	failAfter int
	written   int
}

func (w *failAfterWriter) Write(p []byte) (int, error) {
	remaining := w.failAfter - w.written
	if remaining <= 0 {
		return 0, fmt.Errorf("broken pipe")
	}
	if len(p) > remaining {
		n, _ := w.buf.Write(p[:remaining])
		w.written += n
		return n, fmt.Errorf("broken pipe")
	}
	n, _ := w.buf.Write(p)
	w.written += n
	return n, nil
}

func TestCopyResponseBodyContinuesAfterDstError(t *testing.T) {
	// 模拟上游返回大量 SSE 数据，客户端在收到前 10 字节后断开
	upstreamData := "data: {\"choices\":[{\"delta\":{\"content\":\"hello world\"}}]}\n\ndata: [DONE]\n\n"
	dst := &failAfterWriter{failAfter: 10}
	var logged strings.Builder

	err := copyResponseBody(dst, strings.NewReader(upstreamData), &logged, 0)
	if err == nil {
		t.Fatal("expected error from broken dst, got nil")
	}
	if err.Error() != "broken pipe" {
		t.Fatalf("expected broken pipe error, got: %v", err)
	}
	// 关键：buffer 应该包含完整的上游响应，即使 dst 写入失败
	if logged.String() != upstreamData {
		t.Fatalf("buffer should contain full upstream response even when dst fails\ngot:  %q\nwant: %q", logged.String(), upstreamData)
	}
	// dst 只收到了前 10 字节
	if dst.buf.Len() != 10 {
		t.Fatalf("dst should only have 10 bytes, got %d", dst.buf.Len())
	}
}

func TestBuildUpstreamURLByProviderResponses(t *testing.T) {
responsesURL, err := buildUpstreamURL("https://api.openai.com/v1", providerOpenAIResponses)
	if err != nil {
		t.Fatalf("build responses upstream url: %v", err)
	}
	if responsesURL != "https://api.openai.com/v1/responses" {
		t.Fatalf("unexpected responses upstream url: %s", responsesURL)
	}
}

func TestBuildRouteTargetsIncludesResponses(t *testing.T) {
	cfg := proxyConfig{
		openaiUpstreamBaseURL: "https://api.openai.com/v1",
		openaiAPIKey:          "openai-key",
	}
	targets, err := buildRouteTargets(cfg)
	if err != nil {
		t.Fatalf("build route targets: %v", err)
	}
	if len(targets) != 2 {
		t.Fatalf("expected 2 route targets (chat/completions + responses), got %d", len(targets))
	}
	respTarget, ok := targets["/v1/responses"]
	if !ok {
		t.Fatal("missing /v1/responses target")
	}
	if respTarget.provider != providerOpenAIResponses {
		t.Fatalf("unexpected provider: %s", respTarget.provider)
	}
	if respTarget.upstreamURL != "https://api.openai.com/v1/responses" {
		t.Fatalf("unexpected upstream url: %s", respTarget.upstreamURL)
	}
	if respTarget.apiKey != "openai-key" {
		t.Fatalf("unexpected api key: %s", respTarget.apiKey)
	}
}

func TestBuildRouteTargetsResponsesIndependentConfig(t *testing.T) {
	cfg := proxyConfig{
		openaiUpstreamBaseURL:          "https://api.openai.com/v1",
		openaiAPIKey:                   "openai-key",
		openaiResponsesUpstreamBaseURL: "https://responses.example.com/v1",
		openaiResponsesAPIKey:          "responses-key",
	}
	targets, err := buildRouteTargets(cfg)
	if err != nil {
		t.Fatalf("build route targets: %v", err)
	}
	if len(targets) != 2 {
		t.Fatalf("expected 2 route targets, got %d", len(targets))
	}

	chatTarget := targets["/v1/chat/completions"]
	if chatTarget.upstreamURL != "https://api.openai.com/v1/chat/completions" {
		t.Fatalf("unexpected chat upstream: %s", chatTarget.upstreamURL)
	}
	if chatTarget.apiKey != "openai-key" {
		t.Fatalf("unexpected chat api key: %s", chatTarget.apiKey)
	}

	respTarget := targets["/v1/responses"]
	if respTarget.upstreamURL != "https://responses.example.com/v1/responses" {
		t.Fatalf("unexpected responses upstream: %s", respTarget.upstreamURL)
	}
	if respTarget.apiKey != "responses-key" {
		t.Fatalf("unexpected responses api key: %s", respTarget.apiKey)
	}
}

func TestBuildRouteTargetsResponsesOnly(t *testing.T) {
	cfg := proxyConfig{
		openaiResponsesUpstreamBaseURL: "https://responses.example.com/v1",
		openaiResponsesAPIKey:          "responses-key",
	}
	targets, err := buildRouteTargets(cfg)
	if err != nil {
		t.Fatalf("build route targets: %v", err)
	}
	if len(targets) != 1 {
		t.Fatalf("expected 1 route target, got %d", len(targets))
	}
	respTarget, ok := targets["/v1/responses"]
	if !ok {
		t.Fatal("missing /v1/responses target")
	}
	if respTarget.upstreamURL != "https://responses.example.com/v1/responses" {
		t.Fatalf("unexpected upstream: %s", respTarget.upstreamURL)
	}
}

func TestNewProxyMuxRoutesResponses(t *testing.T) {
	cfg := proxyConfig{
		healthzPath: "/healthz",
	}
	server := &proxyServer{routeTargets: map[string]upstreamTarget{
		"/v1/chat/completions": {provider: providerOpenAI},
		"/v1/messages":         {provider: providerAnthropic},
		"/v1/responses":        {provider: providerOpenAIResponses},
	}}
	mux := newProxyMux(server, cfg)

	resp := httptest.NewRecorder()
	mux.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/v1/responses", nil))
	if resp.Code != http.StatusMethodNotAllowed {
		t.Fatalf("/v1/responses should exist and reject GET with 405, got %d", resp.Code)
	}
}

func TestExtractResponseInfoResponsesNonStream(t *testing.T) {
	body := `{"id":"resp_123","object":"response","status":"completed","model":"gpt-4o","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello world"}]},{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`
	info := extractResponseInfo(body, providerOpenAIResponses)

	if info.Content != "Hello world" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.FinishReason != "completed" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 10 || info.Usage.CompletionTokens != 5 || info.Usage.TotalTokens != 15 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
	if len(info.ToolCalls) != 1 {
		t.Fatalf("unexpected tool call count: %d", len(info.ToolCalls))
	}
	if info.ToolCalls[0].ID != "call_1" || info.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected tool call: %+v", info.ToolCalls[0])
	}
	if strings.TrimSpace(info.ToolCalls[0].Function.Arguments) != `{"city":"Beijing"}` {
		t.Fatalf("unexpected tool args: %s", info.ToolCalls[0].Function.Arguments)
	}
}

func TestExtractResponseInfoResponsesStream(t *testing.T) {
	body := strings.Join([]string{
		"event: response.created",
		`data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`,
		"",
		"event: response.output_item.added",
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}`,
		"",
		"event: response.content_part.added",
		`data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`,
		"",
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}`,
		"",
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":" world"}`,
		"",
		"event: response.output_text.done",
		`data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello world"}`,
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
	}, "\n")

	info := extractResponseInfo(body, providerOpenAIResponses)
	if info.Content != "Hello world" {
		t.Fatalf("unexpected content: %q", info.Content)
	}
	if info.FinishReason != "completed" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 10 || info.Usage.CompletionTokens != 5 || info.Usage.TotalTokens != 15 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
}

func TestExtractResponseInfoResponsesStreamWithToolCalls(t *testing.T) {
	body := strings.Join([]string{
		"event: response.output_item.added",
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}`,
		"",
		"event: response.function_call_arguments.delta",
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"content_index":0,"partial_json":"{\"lo"}`,
		"",
		"event: response.function_call_arguments.delta",
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"content_index":0,"partial_json":"cation\":\"SF\"}"}`,
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`,
	}, "\n")

	info := extractResponseInfo(body, providerOpenAIResponses)
	if len(info.ToolCalls) != 1 {
		t.Fatalf("unexpected tool call count: %d", len(info.ToolCalls))
	}
	if info.ToolCalls[0].ID != "call_1" {
		t.Fatalf("unexpected tool call ID: %s", info.ToolCalls[0].ID)
	}
	if info.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected tool name: %s", info.ToolCalls[0].Function.Name)
	}
	if info.ToolCalls[0].Function.Arguments != `{"location":"SF"}` {
		t.Fatalf("unexpected tool args: %s", info.ToolCalls[0].Function.Arguments)
	}
	if info.FinishReason != "completed" {
		t.Fatalf("unexpected finish reason: %s", info.FinishReason)
	}
	if info.Usage == nil || info.Usage.PromptTokens != 5 || info.Usage.CompletionTokens != 3 {
		t.Fatalf("unexpected usage: %+v", info.Usage)
	}
}

func TestDualUpstreamProxyForwardsResponsesEndpoint(t *testing.T) {
	var receivedPath string
	var receivedAuth string
	var receivedAPIKey string

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedAuth = r.Header.Get("Authorization")
		receivedAPIKey = r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}`)
	}))
	defer upstream.Close()

	upstreamURL, err := buildUpstreamURL(upstream.URL, providerOpenAIResponses)
	if err != nil {
		t.Fatalf("build responses upstream url: %v", err)
	}

	tmpDir := t.TempDir()

	proxy := &proxyServer{
		routeTargets: map[string]upstreamTarget{
			"/v1/responses": {
				provider:    providerOpenAIResponses,
				upstreamURL: upstreamURL,
				apiKey:      "responses-proxy-key",
			},
		},
		logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
		client:  &http.Client{},
		msgDir:  tmpDir,
		nextID:  atomic.Uint64{},
	}
	cfg := proxyConfig{
		healthzPath: "/healthz",
	}
	mux := newProxyMux(proxy, cfg)

	body := `{"model":"gpt-4o","input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Api-Key", "client-x-api-key")
	resp := httptest.NewRecorder()
	mux.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", resp.Code)
	}
	if receivedPath != "/responses" {
		t.Fatalf("unexpected upstream path: %s", receivedPath)
	}
	if receivedAuth != "Bearer responses-proxy-key" {
		t.Fatalf("proxy should inject Authorization, got: %s", receivedAuth)
	}
	if receivedAPIKey != "" {
		t.Fatalf("upstream should not include X-Api-Key, got: %s", receivedAPIKey)
	}

	entries, err := os.ReadDir(filepath.Join(tmpDir, "openai-responses"))
	if err != nil {
		t.Fatalf("read provider dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 message file, got %d", len(entries))
	}

	contentBytes, err := os.ReadFile(filepath.Join(tmpDir, "openai-responses", entries[0].Name()))
	if err != nil {
		t.Fatalf("read message file: %v", err)
	}
	content := string(contentBytes)
	if !strings.Contains(content, "Provider:   openai-responses") {
		t.Fatalf("file should include provider marker, content: %s", content)
	}
	if !strings.Contains(content, "Finish:     completed") {
		t.Fatalf("file should include finish reason, content: %s", content)
	}
}

func TestParseResponsesRequest(t *testing.T) {
	t.Run("string input", func(t *testing.T) {
		body := `{"model":"gpt-4o","input":"What is 2+2?","instructions":"You are a math tutor."}`
		req := parseResponsesRequest(body)
		if req.Model != "gpt-4o" {
			t.Fatalf("unexpected model: %s", req.Model)
		}
		if req.Instructions != "You are a math tutor." {
			t.Fatalf("unexpected instructions: %s", req.Instructions)
		}
		msgs := extractMessagesFromResponsesRequest(req)
		if !strings.Contains(msgs, "INSTRUCTIONS") {
			t.Fatalf("messages should contain instructions section: %s", msgs)
		}
		if !strings.Contains(msgs, "What is 2+2?") {
			t.Fatalf("messages should contain input text: %s", msgs)
		}
	})

	t.Run("array input", func(t *testing.T) {
		body := `{"model":"gpt-4o","input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"Be helpful."}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"Hi"}]}]}`
		req := parseResponsesRequest(body)
		if req.Model != "gpt-4o" {
			t.Fatalf("unexpected model: %s", req.Model)
		}
		msgs := extractMessagesFromResponsesRequest(req)
		if !strings.Contains(msgs, "DEVELOPER") {
			t.Fatalf("messages should contain developer role: %s", msgs)
		}
		if !strings.Contains(msgs, "Be helpful.") {
			t.Fatalf("messages should contain developer text: %s", msgs)
		}
		if !strings.Contains(msgs, "Hi") {
			t.Fatalf("messages should contain user text: %s", msgs)
		}
	})

	t.Run("with tools", func(t *testing.T) {
		body := `{"model":"gpt-4o","input":"test","tools":[{"type":"function","name":"lookup","description":"Search","parameters":{"type":"object"}},{"type":"web_search"},{"type":"image_generation"}]}`
		req := parseResponsesRequest(body)
		tools := extractToolsFromResponsesRequest(req)
		if !strings.Contains(tools, "lookup") {
			t.Fatalf("tools should contain function name: %s", tools)
		}
		if !strings.Contains(tools, "Search") {
			t.Fatalf("tools should contain description: %s", tools)
		}
		if !strings.Contains(tools, "web_search") {
			t.Fatalf("tools should contain web_search type: %s", tools)
		}
		if !strings.Contains(tools, "image_generation") {
			t.Fatalf("tools should contain image_generation type: %s", tools)
		}
	})
}
