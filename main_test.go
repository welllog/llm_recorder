package main

import (
	"flag"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

func TestNewProxyMuxRoutesAlwaysAvailable(t *testing.T) {
	server := &proxyServer{routeTargets: map[string]upstreamTarget{}}
	mux := newProxyMux(server)

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
	mux := newProxyMux(proxy)

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
		t.Fatalf("expected 2 message files, got %d", len(entries))
	}

	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	sort.Strings(names)

	if !strings.Contains(names[0], "_anthropic_") && !strings.Contains(names[1], "_anthropic_") {
		t.Fatalf("expected anthropic message file in names: %v", names)
	}
	if !strings.Contains(names[0], "_openai_") && !strings.Contains(names[1], "_openai_") {
		t.Fatalf("expected openai message file in names: %v", names)
	}

	var openaiFile string
	var anthropicFile string
	for _, name := range names {
		if strings.Contains(name, "_openai_") {
			openaiFile = filepath.Join(tmpDir, name)
		}
		if strings.Contains(name, "_anthropic_") {
			anthropicFile = filepath.Join(tmpDir, name)
		}
	}

	openaiContentBytes, err := os.ReadFile(openaiFile)
	if err != nil {
		t.Fatalf("read openai message file: %v", err)
	}
	anthropicContentBytes, err := os.ReadFile(anthropicFile)
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

	_, err = runParseFlagsWithArgs(t, []string{
		"--openai-upstream-base-url", "https://api.openai.com/v1",
		"--openai-api-key", "openai-key",
		"--anthropic-upstream-base-url", "https://api.anthropic.com",
	})
	if err == nil || !strings.Contains(err.Error(), "--anthropic-api-key") {
		t.Fatalf("expected missing anthropic api key error, got: %v", err)
	}

	_, err = runParseFlagsWithArgs(t, []string{
		"--openai-api-key", "openai-key",
		"--anthropic-upstream-base-url", "https://api.anthropic.com",
		"--anthropic-api-key", "anthropic-key",
	})
	if err == nil || !strings.Contains(err.Error(), "--openai-upstream-base-url") {
		t.Fatalf("expected missing openai upstream base url error, got: %v", err)
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
