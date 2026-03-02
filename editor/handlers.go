package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

var allowedFiles = []string{"skills", "projects", "experience", "certifications"}

func isAllowed(name string) bool {
	for _, f := range allowedFiles {
		if f == name {
			return true
		}
	}
	return false
}

func filePath(indexDir, name string) string {
	return filepath.Join(indexDir, name+".yaml")
}

func handleListFiles(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(allowedFiles)
}

type fileResponse struct {
	Raw   string      `json:"raw"`
	Data  interface{} `json:"data"`
	Mtime int64       `json:"mtime"`
}

func handleGetFile(w http.ResponseWriter, r *http.Request, indexDir string) {
	name := r.PathValue("name")
	if !isAllowed(name) {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}

	path := filePath(indexDir, name)
	info, err := os.Stat(path)
	if err != nil {
		http.Error(w, `{"error":"file not found"}`, http.StatusNotFound)
		return
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, `{"error":"read error"}`, http.StatusInternalServerError)
		return
	}

	var data interface{}
	if err := yaml.Unmarshal(raw, &data); err != nil {
		data = nil
	}

	resp := fileResponse{
		Raw:   string(raw),
		Data:  data,
		Mtime: info.ModTime().UnixMilli(),
	}
	json.NewEncoder(w).Encode(resp)
}

type putRequest struct {
	Raw string `json:"raw"`
}

type putResponse struct {
	OK     bool     `json:"ok"`
	Errors []string `json:"errors"`
	Mtime  int64    `json:"mtime"`
}

func handlePutFile(w http.ResponseWriter, r *http.Request, indexDir string, watcher *Watcher) {
	name := r.PathValue("name")
	if !isAllowed(name) {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"read body error"}`, http.StatusBadRequest)
		return
	}

	var req putRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}

	// Parse into yaml.Node tree (preserves comments and ordering)
	var rootNode yaml.Node
	if err := yaml.Unmarshal([]byte(req.Raw), &rootNode); err != nil {
		resp := putResponse{OK: false, Errors: []string{fmt.Sprintf("YAML parse error: %v", err)}}
		json.NewEncoder(w).Encode(resp)
		return
	}

	errors := validateNode(name, &rootNode)
	if len(errors) > 0 {
		resp := putResponse{OK: false, Errors: errors}
		json.NewEncoder(w).Encode(resp)
		return
	}

	path := filePath(indexDir, name)
	tmpPath := path + ".tmp"

	watcher.SuppressNext(name)

	if err := os.WriteFile(tmpPath, []byte(req.Raw), 0644); err != nil {
		http.Error(w, `{"error":"write error"}`, http.StatusInternalServerError)
		return
	}

	if err := os.Rename(tmpPath, path); err != nil {
		http.Error(w, `{"error":"rename error"}`, http.StatusInternalServerError)
		return
	}

	info, _ := os.Stat(path)
	var mtime int64
	if info != nil {
		mtime = info.ModTime().UnixMilli()
	} else {
		mtime = time.Now().UnixMilli()
	}

	resp := putResponse{OK: true, Errors: []string{}, Mtime: mtime}
	json.NewEncoder(w).Encode(resp)
}

type validateResponse struct {
	Valid  bool     `json:"valid"`
	Errors []string `json:"errors"`
}

func handleValidateFile(w http.ResponseWriter, r *http.Request, indexDir string) {
	name := r.PathValue("name")
	if !isAllowed(name) {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"read body error"}`, http.StatusBadRequest)
		return
	}

	var req putRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}

	var rootNode yaml.Node
	if err := yaml.Unmarshal([]byte(req.Raw), &rootNode); err != nil {
		resp := validateResponse{Valid: false, Errors: []string{fmt.Sprintf("YAML parse error: %v", err)}}
		json.NewEncoder(w).Encode(resp)
		return
	}

	errors := validateNode(name, &rootNode)
	resp := validateResponse{Valid: len(errors) == 0, Errors: errors}
	if resp.Errors == nil {
		resp.Errors = []string{}
	}
	json.NewEncoder(w).Encode(resp)
}

func handleSSE(w http.ResponseWriter, r *http.Request, watcher *Watcher) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Override the application/json set by middleware
	w.Header().Set("Content-Type", "text/event-stream")

	ch := watcher.Subscribe()
	defer watcher.Unsubscribe(ch)

	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case evt := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", evt)
			flusher.Flush()
		case <-heartbeat.C:
			fmt.Fprintf(w, ": heartbeat\n\n")
			flusher.Flush()
		}
	}
}

// nodeToMap extracts a yaml.Node (document or mapping) into a map for key lookup
func nodeMapping(node *yaml.Node) map[string]*yaml.Node {
	result := make(map[string]*yaml.Node)
	if node.Kind == yaml.DocumentNode && len(node.Content) > 0 {
		node = node.Content[0]
	}
	if node.Kind != yaml.MappingNode {
		return result
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		result[node.Content[i].Value] = node.Content[i+1]
	}
	return result
}

func nodeStringList(node *yaml.Node, path string) []string {
	var errs []string
	if node.Kind != yaml.SequenceNode {
		errs = append(errs, fmt.Sprintf("%s: expected list, got %s", path, kindName(node.Kind)))
		return errs
	}
	for i, item := range node.Content {
		if item.Kind != yaml.ScalarNode {
			errs = append(errs, fmt.Sprintf("%s[%d]: expected string, got %s", path, i, kindName(item.Kind)))
		}
	}
	return errs
}

func kindName(k yaml.Kind) string {
	switch k {
	case yaml.DocumentNode:
		return "document"
	case yaml.SequenceNode:
		return "list"
	case yaml.MappingNode:
		return "map"
	case yaml.ScalarNode:
		return "scalar"
	default:
		return "unknown"
	}
}

func requireKeys(m map[string]*yaml.Node, path string, keys []string) []string {
	var errs []string
	for _, k := range keys {
		if _, ok := m[k]; !ok {
			errs = append(errs, fmt.Sprintf("%s: missing required key '%s'", path, k))
		}
	}
	return errs
}

func enumCheck(node *yaml.Node, path string, allowed []string) []string {
	for _, v := range allowed {
		if node.Value == v {
			return nil
		}
	}
	return []string{fmt.Sprintf("%s: must be one of [%s], got '%s'", path, strings.Join(allowed, ", "), node.Value)}
}
