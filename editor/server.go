package main

import (
	"embed"
	"io/fs"
	"mime"
	"net/http"
)

//go:embed static
var staticFiles embed.FS

func init() {
	// Ensure .js files are served with correct MIME type (Alpine/embed may lack it)
	mime.AddExtensionType(".js", "application/javascript")
	mime.AddExtensionType(".mjs", "application/javascript")
	mime.AddExtensionType(".css", "text/css")
}

func buildMux(indexDir string, w *Watcher) http.Handler {
	mux := http.NewServeMux()

	staticSub, _ := fs.Sub(staticFiles, "static")
	mux.Handle("/", http.FileServer(http.FS(staticSub)))

	mux.HandleFunc("GET /api/files", handleListFiles)
	mux.HandleFunc("GET /api/files/{name}", func(rw http.ResponseWriter, r *http.Request) {
		handleGetFile(rw, r, indexDir)
	})
	mux.HandleFunc("PUT /api/files/{name}", func(rw http.ResponseWriter, r *http.Request) {
		handlePutFile(rw, r, indexDir, w)
	})
	mux.HandleFunc("POST /api/files/{name}/validate", func(rw http.ResponseWriter, r *http.Request) {
		handleValidateFile(rw, r, indexDir)
	})
	mux.HandleFunc("GET /api/events", func(rw http.ResponseWriter, r *http.Request) {
		handleSSE(rw, r, w)
	})

	return corsMiddleware(jsonAPIMiddleware(mux))
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, PUT, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func jsonAPIMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(r.URL.Path) > 4 && r.URL.Path[:5] == "/api/" {
			w.Header().Set("Content-Type", "application/json")
		}
		next.ServeHTTP(w, r)
	})
}
