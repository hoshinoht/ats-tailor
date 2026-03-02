package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

func main() {
	port := flag.Int("port", 8070, "HTTP server port")
	dir := flag.String("dir", "", "Path to index directory (default: <git-root>/index)")
	noBrowser := flag.Bool("no-browser", false, "Don't open browser on start")
	flag.Parse()

	absDir, err := resolveIndexDir(*dir)
	if err != nil {
		log.Fatalf("cannot resolve index dir: %v", err)
	}

	watcher, err := newWatcher(absDir)
	if err != nil {
		log.Fatalf("watcher init failed: %v", err)
	}
	defer watcher.Close()

	go watcher.Run()

	mux := buildMux(absDir, watcher)

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("yamledit listening on http://localhost%s (index: %s)", addr, absDir)

	if !*noBrowser {
		go func() {
			time.Sleep(500 * time.Millisecond)
			openBrowser(fmt.Sprintf("http://localhost%s", addr))
		}()
	}

	log.Fatal(http.ListenAndServe(addr, mux))
}

// resolveIndexDir finds the index directory:
//  1. Explicit -dir flag if provided
//  2. git repo root + /index
//  3. Fall back to ./index
func resolveIndexDir(explicit string) (string, error) {
	if explicit != "" {
		return filepath.Abs(explicit)
	}

	// Try git root
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err == nil {
		root := strings.TrimSpace(string(out))
		dir := filepath.Join(root, "index")
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			return dir, nil
		}
	}

	// Fall back to ./index
	if info, err := os.Stat("index"); err == nil && info.IsDir() {
		return filepath.Abs("index")
	}

	return "", fmt.Errorf("no index/ directory found — pass -dir explicitly or run from within the git repo")
}

func openBrowser(url string) {
	var cmd string
	var args []string
	switch runtime.GOOS {
	case "darwin":
		cmd = "open"
		args = []string{url}
	case "linux":
		cmd = "xdg-open"
		args = []string{url}
	default:
		cmd = "cmd"
		args = []string{"/c", "start", url}
	}
	if err := exec.Command(cmd, args...).Start(); err != nil {
		log.Printf("could not open browser: %v", err)
	}
}
